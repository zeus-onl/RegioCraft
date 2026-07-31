r"""
RegioCraft  —  ComfyUI custom node (Krea2 / Flux.2-Klein single-stream DiT)
================================================================================
The "best of three" merge of:
  - regional_character_lora          (base engine + Attention Isolation, WIP)
  - regional_character_lora_sparse   (sparse-token hook, ramp/warmup, mask outs)
  - Krea2-Multi-Character-Lora-Node  (unlimited regions, LoKr, Reference Lock)

WHAT'S IN HERE
  1. Unlimited regions (2, 3, 10... draw a box, add a row).
  2. Standard LoRA (kohya lora_down/up, diffusers lora_A/B) AND LoKr
     (lokr_w1/w2, incl. decomposed w1_a@w1_b / w2_a@w2_b) support.
  3. Sparse-token forward hook: only the tokens whose regional mask is above
     `sparse_threshold` pay for the LoRA/LoKr matmul, not the whole sequence.
  4. Attention Isolation: additive cross-region attention-logit bias so
     region A's query tokens attend less to region B's key/value tokens
     (dampens identity bleed at the attention level, not just the LoRA-delta
     level). Multi-architecture registry (Krea2, Flux.1); text tokens always
     keep full unrestricted attention.
  5. Reference Lock: optional per-region reference image. VAE-encoded once,
     resized into its box on the latent grid ("the mold"), then a post-CFG
     hook nudges the denoised prediction toward it every step inside a
     scheduled sigma window. Anchors identity across seeds/generations,
     something LoRA-delta masking alone cannot do.
  6. steps_without_applying + lora_ramp_calls: let the base composition
     establish itself for N model calls before regional LoRAs kick in, then
     ramp them up instead of slamming full strength on call 1.
  7. Debug outputs: per-region mask preview (rainbow-coded IMAGE) + a data
     dict describing what got armed.

Engine notes carried over from the three parents:
  - LoRA deltas are never merged into weights; they're injected as forward-
    hook activation deltas, masked to a spatial region. fp8/quantized Krea2
    checkpoints are untouched (we only read the quantized Linear's input/
    output activations), which is why this stays fp8-safe. See the original
    regional_character_lora README for the "why this works vs. a stack"
    explanation; unchanged here.
  - The image-token grid assumption is the Krea2 one: VAE f8 then patch=2,
    so image tokens = (H//16) * (W//16), trailing the sequence
    ([text | image]). Nothing about 512 or a fixed resolution is hardcoded;
    the grid is read from the live latent shape at first forward call.

INSTALL
  <ComfyUI>/custom_nodes/RegioCraft/__init__.py   (this file)
  Restart ComfyUI -> Add Node -> conditioning/regional -> "RegioCraft".

CREDITS
  RegioCraft stands on the shoulders of the people who built the three parent
  nodes it merges:
    - Gorecheese, who created the original idea and node -- regional_character_lora,
      the activation-delta masking engine this whole approach is built on.
    - Shy, who forked and extended that work (the sparse-token hook, Attention
      Isolation, and the ramp/warmup scheduling).
    - Fedor, who built Krea2-Multi-Character-Lora-Node-w-bounding-box
      (unlimited regions, LoKr support, Reference Lock / latent-mold identity
      anchoring).
  Thanks to all three for the groundwork -- none of this exists without it.
"""

import os
import re
import json
import math
import logging
import importlib

import numpy as np
import torch
import safetensors.torch

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    import comfy.model_management
except Exception:
    comfy = None

try:
    import comfy.patcher_extension as _pext
    _WRAPPER_ENUM = _pext.WrappersMP.DIFFUSION_MODEL
except Exception:
    _pext = None
    _WRAPPER_ENUM = "diffusion_model"

WRAPPER_KEY = "regiocraft"
__version__ = "0.1.0"

_COMPUTE_DTYPE = torch.bfloat16


# ============================================================================
# generic helpers (shared across all three parents, unchanged)
# ============================================================================
def _lora_dir_list():
    if folder_paths is not None:
        try:
            return folder_paths.get_filename_list("loras")
        except Exception:
            pass
    return []


def _resolve_lora_path(name):
    if folder_paths is not None:
        try:
            p = folder_paths.get_full_path("loras", name)
            if p:
                return p
        except Exception:
            pass
    return name


def _norm(s):
    s = s.lower()
    for pre in ("lora_unet_", "lora_te_", "lora_", "diffusion_model.",
                "diffusion_model_", "transformer.", "model.diffusion_model.",
                "model.", "base_model."):
        if s.startswith(pre):
            s = s[len(pre):]
    return s.replace(".", "").replace("_", "")


def _iter_named_linears(module):
    for name, sub in module.named_modules():
        if isinstance(sub, torch.nn.Linear) or hasattr(sub, "weight"):
            yield name, sub


# ============================================================================
# LoRA + LoKr loading  (standard A/B pairs, and Kronecker-factor LoKr)
# ============================================================================
def _load_lora_matrices(path):
    """Return { module_sig: entry } where entry is either:
       {'kind':'lora', 'down':T, 'up':T, 'scale':float}
    or {'kind':'lokr', 'w1':T, 'w2':T, 'scale':float}
    LoKr tucker/conv-decomposed variants are skipped (rare, conv-only)."""
    sd = safetensors.torch.load_file(path)
    lora_groups, lora_alphas = {}, {}
    lokr_groups, lokr_alphas = {}, {}

    for k, v in sd.items():
        if k.endswith(".alpha") or k.endswith("alpha"):
            base = re.sub(r"\.?alpha$", "", k)
            try:
                a = float(v.flatten()[0].item())
            except Exception:
                continue
            lora_alphas[base] = a
            lokr_alphas[base] = a
            continue

        m = re.search(r"(.*?)\.(lora_down|lora_A)\.weight$", k)
        if m:
            lora_groups.setdefault(m.group(1), {})["down"] = v.float()
            continue
        m = re.search(r"(.*?)\.(lora_up|lora_B)\.weight$", k)
        if m:
            lora_groups.setdefault(m.group(1), {})["up"] = v.float()
            continue

        m = re.search(r"(.*?)\.lokr_w1\.weight$", k)
        if m:
            lokr_groups.setdefault(m.group(1), {})["w1"] = v.float()
            continue
        m = re.search(r"(.*?)\.lokr_w1$", k)
        if m:
            lokr_groups.setdefault(m.group(1), {})["w1"] = v.float()
            continue
        m = re.search(r"(.*?)\.lokr_w2\.weight$", k)
        if m:
            lokr_groups.setdefault(m.group(1), {})["w2"] = v.float()
            continue
        m = re.search(r"(.*?)\.lokr_w2$", k)
        if m:
            lokr_groups.setdefault(m.group(1), {})["w2"] = v.float()
            continue
        # decomposed factors: w1_a @ w1_b, w2_a @ w2_b
        for side in ("w1", "w2"):
            m = re.search(r"(.*?)\.lokr_%s_a\.weight$" % side, k)
            if m:
                lokr_groups.setdefault(m.group(1), {}).setdefault(side + "_parts", {})["a"] = v.float()
                break
            m = re.search(r"(.*?)\.lokr_%s_b\.weight$" % side, k)
            if m:
                lokr_groups.setdefault(m.group(1), {}).setdefault(side + "_parts", {})["b"] = v.float()
                break

    out = {}
    for base, mats in lora_groups.items():
        if "down" not in mats or "up" not in mats:
            continue
        down, up = mats["down"], mats["up"]
        rank = down.shape[0]
        alpha = lora_alphas.get(base, float(rank))
        out[_norm(base)] = {
            "kind": "lora", "down": down, "up": up,
            "scale": float(alpha) / float(rank), "_dbg": base,
        }

    skipped_lokr = 0
    for base, mats in lokr_groups.items():
        w1 = mats.get("w1")
        w2 = mats.get("w2")
        if w1 is None and "w1_parts" in mats:
            p = mats["w1_parts"]
            if "a" in p and "b" in p:
                w1 = p["a"] @ p["b"]
        if w2 is None and "w2_parts" in mats:
            p = mats["w2_parts"]
            if "a" in p and "b" in p:
                w2 = p["a"] @ p["b"]
        if w1 is None or w2 is None:
            skipped_lokr += 1
            continue
        # tucker/conv variants have >2 dims left over; skip those, standard
        # linear LoKr factors are 2D.
        if w1.dim() != 2 or w2.dim() != 2:
            skipped_lokr += 1
            continue
        rank = min(w1.shape[0], w1.shape[1], w2.shape[0], w2.shape[1])
        alpha = lokr_alphas.get(base, float(rank))
        out[_norm(base)] = {
            "kind": "lokr", "w1": w1, "w2": w2,
            "scale": float(alpha) / float(max(1, rank)), "_dbg": base,
        }
    if skipped_lokr:
        logging.info("[RegioCraft] skipped %d LoKr entr(y/ies) with tucker/conv "
                      "factors (unsupported) in %s", skipped_lokr, path)
    return out


def _materialize_delta_fn(entry, dev, cdt):
    """Return (compute_fn(x_sel) -> delta, effective_scale) for one loaded
    entry, with weights pre-moved to device/dtype once. Standard LoRA does
    x @ down.T @ up.T; LoKr materializes kron(w1, w2) once (linear factors
    are typically small enough that this is cheap and simple) and does
    x @ W.T. Numerically this is the same identity ComfyUI's own LoKr
    adapter uses."""
    scale = entry["scale"]
    if entry["kind"] == "lora":
        down_d = entry["down"].to(dev, cdt)
        up_d = (entry["up"].to(dev, cdt)) * scale

        def fn(x_sel):
            return (x_sel @ down_d.t()) @ up_d.t()
        return fn
    else:  # lokr
        w1 = entry["w1"].to(dev, torch.float32)
        w2 = entry["w2"].to(dev, torch.float32)
        kron = torch.kron(w1, w2) * scale
        kron_d = kron.to(cdt)

        def fn(x_sel):
            return x_sel @ kron_d.t()
        return fn


# ============================================================================
# region parsing: unlimited rows, each {lora, strength, enable, ref_image}
# ============================================================================
DEFAULT_REGIONS_JSON = (
    "[\n"
    '  {"lora": "None", "strength": 1.2, "enable": true, "ref_image": "",'
    ' "prompt": "", "trigger": "", "x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0},\n'
    '  {"lora": "None", "strength": 1.2, "enable": true, "ref_image": "",'
    ' "prompt": "", "trigger": "", "x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0}\n'
    "]"
)


def _default_box(i, n):
    """Equal left->right columns, used when a region has no stored x/y/w/h yet
    (freshly added row before the user drags it)."""
    n = max(1, n)
    return (i / n, 0.0, (i + 1) / n, 1.0)


def _parse_regions(regions_json):
    try:
        raw = json.loads(regions_json)
    except Exception:
        raw = []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        raw = []
    out = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            continue
        has_box = all(k in r for k in ("x", "y", "w", "h"))
        if has_box:
            bx, by, bw, bh = float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])
        else:
            bx, by, x1, y1 = _default_box(i, len(raw))
            bw, bh = x1 - bx, y1 - by
        out.append({
            "name": r.get("name") or f"region_{i}",
            "lora": str(r.get("lora", "None") or "None"),
            "strength": float(r.get("strength", 1.0)),
            "enable": bool(r.get("enable", True)),
            "ref_image": str(r.get("ref_image", "") or "").strip(),
            "prompt": str(r.get("prompt", "") or "").strip(),
            "trigger": str(r.get("trigger", "") or "").strip(),
            "box": (max(0.0, min(1.0, bx)), max(0.0, min(1.0, by)),
                    max(0.0, min(1.0, bx + bw)), max(0.0, min(1.0, by + bh))),
        })
    return out


def _flatten_bboxes(bboxes):
    if not bboxes:
        return []
    try:
        first = bboxes[0]
    except Exception:
        return []
    if isinstance(first, (list, tuple)):
        return list(first)
    return list(bboxes)


def _coerce_bbox_norm(box, w, h):
    vals = list(box) if not isinstance(box, dict) else [
        box.get("x", box.get("x0", 0)), box.get("y", box.get("y0", 0)),
        box.get("x1", box.get("x", 0) + box.get("w", box.get("width", 0))),
        box.get("y1", box.get("y", 0) + box.get("h", box.get("height", 0)))]
    x0, y0, x1, y1 = [float(v) for v in vals[:4]]
    if max(x0, y0, x1, y1) > 1.0:          # pixels -> normalised
        x0, x1 = x0 / w, x1 / w
        y0, y1 = y0 / h, y1 / h
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return (max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1))


def _auto_split_norm(n, mode):
    """Equal strips fallback when there's no bbox wire. mode: auto_vertical
    (left->right columns) or auto_horizontal (top->bottom rows)."""
    n = max(1, n)
    boxes = []
    for i in range(n):
        if mode == "auto_horizontal":
            boxes.append((0.0, i / n, 1.0, (i + 1) / n))
        else:
            boxes.append((i / n, 0.0, (i + 1) / n, 1.0))
    return boxes


def _rect_token_mask(rows, cols, nx0, ny0, nx1, ny1, feather):
    c0, c1 = nx0 * cols, nx1 * cols
    r0, r1 = ny0 * rows, ny1 * rows
    fc = max(1e-3, feather * cols)
    fr = max(1e-3, feather * rows)
    cc = torch.arange(cols, dtype=torch.float32).unsqueeze(0)
    rr = torch.arange(rows, dtype=torch.float32).unsqueeze(1)
    in_x = torch.sigmoid((cc - c0) / fc) * torch.sigmoid((c1 - cc) / fc)
    in_y = torch.sigmoid((rr - r0) / fr) * torch.sigmoid((r1 - rr) / fr)
    return (in_y * in_x).reshape(-1).clamp(0.0, 1.0)


def _build_token_grid(w, h):
    return max(1, h // 16), max(1, w // 16)   # rows, cols


# ============================================================================
# attention isolation: multi-architecture registry (from regional_character_lora)
# ============================================================================
_ATTN_TARGETS = [
    ("comfy.ldm.krea2.model", "optimized_attention_masked", "comfy.ldm.krea2"),
    ("comfy.ldm.flux.layers", "attention", "comfy.ldm.flux"),
]
_ACTIVE_ISOLATION_SESSION = None
_WARNED_UNSUPPORTED_ARCHS = set()


def _resolve_attn_target(dm):
    dm_module = type(dm).__module__
    for mod_path, attr, needle in _ATTN_TARGETS:
        if needle in dm_module:
            try:
                mod = importlib.import_module(mod_path)
                return mod, attr, getattr(mod, attr)
            except Exception:
                return None
    if dm_module not in _WARNED_UNSUPPORTED_ARCHS:
        _WARNED_UNSUPPORTED_ARCHS.add(dm_module)
        logging.info("[RegioCraft] attention_isolation: no patch target for "
                     "architecture '%s' yet; regional LoRA masking still "
                     "applies as usual, just without attention-level dampening.",
                     dm_module)
    return None


def _make_isolation_wrapper(orig_fn):
    def wrapper(*args, **kwargs):
        sess = _ACTIVE_ISOLATION_SESSION
        q = args[0] if args else kwargs.get("q")
        if sess is not None and sess._img_bias is not None and torch.is_tensor(q):
            seq = q.shape[-2]
            bias = sess._full_attn_bias(seq, q.device, q.dtype)
            mask = kwargs.get("mask")
            if mask is None:
                kwargs["mask"] = bias
            elif mask.dtype != torch.bool:
                kwargs["mask"] = mask + bias.to(mask.dtype)
        return orig_fn(*args, **kwargs)
    return wrapper


# ============================================================================
# the session: N regions, sparse hook, ramp/warmup, attention isolation
# ============================================================================
class _RegioCraftSession:
    def __init__(self, patcher, active_regions, norm_boxes, seam_feather,
                 blend_override, sparse_threshold, steps_without_applying,
                 lora_ramp_calls, attention_isolation):
        self.patcher = patcher
        self.active = active_regions           # list of dicts with 'name','lora_path','strength'
        self.norm_boxes = norm_boxes            # list of (x0,y0,x1,y1) normalised
        self.seam_feather = float(seam_feather)
        self.blend_override = float(blend_override)
        self.sparse_threshold = max(0.0, float(sparse_threshold))
        self.steps_without_applying = max(0, int(steps_without_applying))
        self.lora_ramp_calls = max(0, int(lora_ramp_calls))
        self.attention_isolation = float(attention_isolation)

        self._model_call_index = 0
        self._ramp_multiplier = 1.0
        self.n_img = 0
        self._layer_map = None      # name -> (module, {region_idx: {fn, sig}})
        self._prepared = False
        self._region_masks = None   # list[Tensor] normalised token masks, one per region
        self._region_masks_d = None
        self._active_token_cache = {}
        self._img_bias = None
        self._attn_bias_cache = {}
        self._dev = None

    def _diffusion_model(self):
        m = self.patcher.model
        return getattr(m, "diffusion_model", m)

    def _build_layer_map(self, dm, dev, cdt):
        # entry: {sig: {region_idx: {'fn': compute_fn}}}
        sig_to_region_fns = {}
        for i, r in enumerate(self.active):
            mats = r["mats"]
            for sig, entry in mats.items():
                fn = _materialize_delta_fn(entry, dev, cdt)
                sig_to_region_fns.setdefault(sig, {})[i] = fn

        layer_map = {}
        matched = 0
        for name, mod in _iter_named_linears(dm):
            sig = _norm(name)
            if sig in sig_to_region_fns:
                layer_map[name] = (mod, sig_to_region_fns[sig])
                matched += 1
        logging.info("[RegioCraft] matched %d layers across %d region(s).",
                     matched, len(self.active))
        if matched == 0:
            logging.warning("[RegioCraft] 0 layers matched - check your LoRA/LoKr "
                            "files are trained for this model architecture.")
        return layer_map

    def _resolve_grid(self, x):
        if torch.is_tensor(x) and x.dim() >= 4:
            H, W = int(x.shape[-2]), int(x.shape[-1])
            rows, cols = H // 2, W // 2
            if rows > 0 and cols > 0:
                return rows, cols
        return _build_token_grid(1024, 1536)

    def _build_masks_now(self, rows, cols):
        masks = []
        for box in self.norm_boxes:
            x0, y0, x1, y1 = box
            masks.append(_rect_token_mask(rows, cols, x0, y0, x1, y1, self.seam_feather))
        blend = self.blend_override
        if blend > 0.0 and masks:
            # 0 -> clean split ; ->0.5 lets everything mix a bit more evenly
            avg = sum(masks) / len(masks)
            masks = [(1.0 - blend) * m + blend * avg for m in masks]
        return masks

    def _prepare(self, dev, x):
        cdt = _COMPUTE_DTYPE
        self._dev = dev
        self._layer_map = self._build_layer_map(self._diffusion_model(), dev, cdt)
        rows, cols = self._resolve_grid(x)
        self.n_img = rows * cols
        self._region_masks = self._build_masks_now(rows, cols)
        self._region_masks_d = [m.to(dev, cdt) for m in self._region_masks]
        self._active_token_cache = {}
        self._build_attn_bias(dev, cdt)
        self._prepared = True
        logging.info("[RegioCraft] prepared | grid=%dx%d n_img=%d regions=%d "
                     "sparse_threshold=%.3f ramp_calls=%d attn_iso=%.1f",
                     rows, cols, self.n_img, len(self.active),
                     self.sparse_threshold, self.lora_ramp_calls,
                     self.attention_isolation)

    # ---- attention isolation (generalised to N regions) --------------
    def _build_attn_bias(self, dev, cdt):
        strength = self.attention_isolation
        self._attn_bias_cache = {}
        if not strength or self.n_img <= 0 or not self._region_masks_d:
            self._img_bias = None
            return
        # coverage[p] = how much token p belongs to ANY region;
        # same[p,q] = how much p and q share the SAME region(s);
        # cross = coverage[p]*coverage[q] - same[p,q]  (>=0), penalise it.
        coverage = torch.zeros(self.n_img, device=dev, dtype=torch.float32)
        same = torch.zeros(self.n_img, self.n_img, device=dev, dtype=torch.float32)
        for m in self._region_masks_d:
            mf = m.float()
            coverage += mf
            same += torch.outer(mf, mf)
        cross = (torch.outer(coverage, coverage) - same).clamp(min=0.0)
        self._img_bias = (-strength * cross).to(cdt)

    def _full_attn_bias(self, seq, device, dtype):
        key = (seq, device, dtype)
        cached = self._attn_bias_cache.get(key)
        if cached is not None:
            return cached
        n_img = self.n_img
        full = torch.zeros(seq, seq, device=device, dtype=dtype)
        if n_img > 0 and n_img <= seq and self._img_bias is not None:
            off = seq - n_img
            full[off:, off:] = self._img_bias.to(device, dtype)
        out = full.view(1, 1, seq, seq)
        self._attn_bias_cache[key] = out
        return out

    # ---- sparse token selection ---------------------------------------
    def _active_tokens(self, region_idx, seq):
        key = (region_idx, int(seq))
        cached = self._active_token_cache.get(key)
        if cached is not None:
            return cached
        mv = self._region_masks_d[region_idx]
        n_img = self.n_img
        if n_img <= 0 or n_img > seq:
            idx = torch.arange(seq, device=self._dev, dtype=torch.long)
            weight = torch.full((seq,), float(mv.mean().item()),
                                 device=self._dev, dtype=_COMPUTE_DTYPE)
        else:
            keep = torch.nonzero(mv.abs() > self.sparse_threshold,
                                  as_tuple=False).flatten().to(torch.long)
            idx = keep + (seq - n_img)
            weight = mv[keep]
        self._active_token_cache[key] = (idx, weight)
        return idx, weight

    # ---- ramp / warmup ---------------------------------------------------
    def _compute_ramp_multiplier(self):
        n = self.lora_ramp_calls
        if n <= 0:
            return 1.0
        active_call = self._model_call_index - self.steps_without_applying
        if active_call <= 0:
            return 0.0
        if active_call <= n:
            return float(active_call) / float(n + 1)
        return 1.0

    def _make_hook(self, region_fns):
        # region_fns: {region_idx: compute_fn}
        def hook(module, inp, out):
            if not torch.is_tensor(out) or out.dim() < 2:
                return out
            x = inp[0]
            if not torch.is_tensor(x) or x.dim() < 2:
                return out
            seq = x.shape[-2]
            xf = x.to(_COMPUTE_DTYPE)
            res = None
            ramp = self._ramp_multiplier
            for region_idx, fn in region_fns.items():
                idx, weight = self._active_tokens(region_idx, seq)
                if idx.numel() == 0:
                    continue
                x_sel = torch.index_select(xf, dim=-2, index=idx)
                delta = fn(x_sel)
                delta = delta * weight.view(*([1] * (delta.dim() - 2)), -1, 1)
                if ramp != 1.0:
                    delta = delta * ramp
                if res is None:
                    res = torch.zeros_like(out, dtype=_COMPUTE_DTYPE)
                res.index_add_(dim=-2, index=idx, source=delta)
            if res is None:
                return out
            return out + res.to(out.dtype)
        return hook

    def run(self, executor, *args, **kwargs):
        self._model_call_index += 1
        if self.steps_without_applying > 0 and self._model_call_index <= self.steps_without_applying:
            return executor(*args, **kwargs)
        self._ramp_multiplier = self._compute_ramp_multiplier()

        dm = self._diffusion_model()
        if not self._prepared:
            dev = args[0].device if args and torch.is_tensor(args[0]) else \
                next(dm.parameters()).device
            self._prepare(dev, args[0] if args else None)
        if not self._layer_map:
            return executor(*args, **kwargs)

        global _ACTIVE_ISOLATION_SESSION
        target = None
        if self.attention_isolation and self._img_bias is not None:
            target = _resolve_attn_target(dm)
        prev_session = _ACTIVE_ISOLATION_SESSION
        handles = []
        try:
            for name, (mod, region_fns) in self._layer_map.items():
                handles.append(mod.register_forward_hook(self._make_hook(region_fns)))
            if target is not None:
                mod, attr, orig_fn = target
                _ACTIVE_ISOLATION_SESSION = self
                setattr(mod, attr, _make_isolation_wrapper(orig_fn))
            return executor(*args, **kwargs)
        finally:
            for h in handles:
                h.remove()
            if target is not None:
                mod, attr, orig_fn = target
                setattr(mod, attr, orig_fn)
                _ACTIVE_ISOLATION_SESSION = prev_session


# ============================================================================
# reference lock helpers (from Krea2-Multi-Character-Lora-Node v3, unchanged idea)
# ============================================================================
def _load_ref_image_tensor(name):
    from PIL import Image, ImageOps
    path = folder_paths.get_annotated_filepath(name)
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None]


def _encode_reference(model, vae, image_tensor):
    img = image_tensor.movedim(-1, 1)  # [1,3,H,W]
    latent = vae.encode(img.movedim(1, -1))
    if hasattr(model.model, "process_latent_in"):
        latent = model.model.process_latent_in(latent)
    return latent


def _sigma_window(model, start_pct, end_pct):
    try:
        sampling = model.get_model_object("model_sampling")
        s_start = float(sampling.percent_to_sigma(1.0 - start_pct))
        s_end = float(sampling.percent_to_sigma(1.0 - end_pct))
    except Exception:
        s_start, s_end = 1000.0, 0.0
    return s_start, s_end


def _in_window(sigma, s_start, s_end):
    s = float(sigma.max().item()) if torch.is_tensor(sigma) else float(sigma)
    return s_end <= s <= s_start


def _build_mold(ref_latent, box, C, H, W, feather, device):
    x0, y0, x1, y1 = box
    bx0, by0 = int(round(x0 * W)), int(round(y0 * H))
    bx1, by1 = int(round(x1 * W)), int(round(y1 * H))
    bx1, by1 = max(bx0 + 1, bx1), max(by0 + 1, by1)
    bw, bh = bx1 - bx0, by1 - by0
    ref = ref_latent.to(device)
    if ref.shape[1] != C:
        return None
    ref_resized = torch.nn.functional.interpolate(
        ref.float(), size=(bh, bw), mode="bilinear", align_corners=False)
    mold = torch.zeros((1, C, H, W), device=device, dtype=torch.float32)
    mold[:, :, by0:by1, bx0:bx1] = ref_resized
    mask = _rect_token_mask_pixel(H, W, x0, y0, x1, y1, feather).to(device)
    mask = mask.view(1, 1, H, W)
    return mold, mask


def _rect_token_mask_pixel(H, W, nx0, ny0, nx1, ny1, feather):
    c0, c1 = nx0 * W, nx1 * W
    r0, r1 = ny0 * H, ny1 * H
    fc = max(1e-3, feather * W)
    fr = max(1e-3, feather * H)
    cc = torch.arange(W, dtype=torch.float32).unsqueeze(0)
    rr = torch.arange(H, dtype=torch.float32).unsqueeze(1)
    in_x = torch.sigmoid((cc - c0) / fc) * torch.sigmoid((c1 - cc) / fc)
    in_y = torch.sigmoid((rr - r0) / fr) * torch.sigmoid((r1 - rr) / fr)
    return (in_y * in_x).clamp(0.0, 1.0)


# ============================================================================
# per-region text conditioning (named-character style prompting, Gorecheese/
# Fedor idea): each region can carry its own text alongside its LoRA. The
# base_prompt is encoded once as a full-image (unmasked) conditioning so
# overall scene/composition still comes from the main prompt; each region
# with its own text gets ADDITIONALLY encoded (base + ", " + region text) and
# masked to that region's box via ComfyUI's standard area-conditioning fields
# (mask / mask_strength / set_area_to_bounds) -- the same mechanism the
# built-in ConditioningSetMask node uses, so it works with any KSampler and
# any model, not just Krea2.
# ============================================================================
def _encode_text(clip, text):
    tokens = clip.tokenize(text or "")
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def _mask_conditioning(cond_list, mask, strength=1.0):
    m = mask.unsqueeze(0) if mask.dim() == 2 else mask
    out = []
    for c, extra in cond_list:
        n_extra = extra.copy()
        n_extra["mask"] = m
        n_extra["set_area_to_bounds"] = False
        n_extra["mask_strength"] = float(strength)
        out.append([c, n_extra])
    return out


def _build_region_conditioning(clip, base_prompt, prepared_regions, norm_boxes,
                                cw, ch, text_strength):
    """Whole-image base conditioning + one masked regional conditioning per
    region that has its own prompt text. Regions without text contribute
    nothing here -- their identity comes purely from the LoRA/LoKr masking."""
    base_prompt = base_prompt or ""
    combined = _encode_text(clip, base_prompt)
    for r, box in zip(prepared_regions, norm_boxes):
        text = r.get("prompt", "")
        if not text:
            continue
        full_text = f"{base_prompt}, {text}" if base_prompt else text
        region_cond = _encode_text(clip, full_text)
        x0, y0, x1, y1 = box
        mask = _rect_token_mask_pixel(ch, cw, x0, y0, x1, y1, 0.02)
        combined += _mask_conditioning(region_cond, mask, text_strength)
    return combined


# ============================================================================
# the node
# ============================================================================
class RegioCraft:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16}),
                "canvas_height": ("INT", {"default": 1024, "min": 64, "max": 16384, "step": 16}),
                "regions_json": ("STRING", {
                    "multiline": True, "default": DEFAULT_REGIONS_JSON,
                    "tooltip": 'JSON array, in box order: {"lora":"file.safetensors",'
                               ' "strength":1.2, "enable":true, "ref_image":"uploaded.png",'
                               ' "prompt":"optional per-region text"}. '
                               "Unlimited rows. LoRA and LoKr files both work.",
                }),
                "base_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Overall scene prompt (composition, setting, lighting). "
                               "Combined with each region's own text (if any) for that "
                               "region's masked conditioning. Also scanned for trigger "
                               "names if auto_activate_from_prompt is on.",
                }),
                "text_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Blend strength of each region's masked text conditioning."}),
                "auto_activate_from_prompt": ("BOOLEAN", {"default": False,
                    "tooltip": "If a region has a 'trigger' name set, that region is only "
                               "active when its trigger word appears (case-insensitive) in "
                               "base_prompt -- the region's own enable toggle is ignored in "
                               "that case. Regions with no trigger set keep using their "
                               "enable toggle as normal (handy for LoRAs that don't need a "
                               "trigger word to activate)."}),
                "split_mode": (["manual", "bbox", "auto_vertical", "auto_horizontal"], {"default": "manual"}),
                "seam_feather": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
                "blend_override": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sparse_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.2, "step": 0.005,
                    "tooltip": "Skip LoRA/LoKr compute on tokens below this mask value. 0=safest/slowest."}),
                "steps_without_applying": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Let the base model establish composition for N model calls first."}),
                "lora_ramp_calls": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "0=off. 1=50->100%, 2=33->66->100%, relative to selected strength."}),
                "attention_isolation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 12.0, "step": 0.5,
                    "tooltip": "0=off. >0 dampens cross-region attention (not just the LoRA delta). Try 4-8."}),
                "ref_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-step pull toward each region's reference image. 0=off."}),
                "ref_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ref_end_percent": ("FLOAT", {"default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ref_feather": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01}),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX",),
                "vae": ("VAE", {"tooltip": "Required to enable reference-image regions."}),
                "base_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "region_prompt_1": ("STRING", {"forceInput": True, "multiline": True,
                    "tooltip": "Connect an external text node here to drive region 1's prompt "
                               "dynamically (e.g. a wildcard/random prompt generator). Overrides "
                               "whatever's typed into region 1's own prompt field when connected."}),
                "region_prompt_2": ("STRING", {"forceInput": True, "multiline": True}),
                "region_prompt_3": ("STRING", {"forceInput": True, "multiline": True}),
                "region_prompt_4": ("STRING", {"forceInput": True, "multiline": True}),
                "region_prompt_5": ("STRING", {"forceInput": True, "multiline": True}),
                "region_prompt_6": ("STRING", {"forceInput": True, "multiline": True}),
                "region_prompt_7": ("STRING", {"forceInput": True, "multiline": True}),
                "region_prompt_8": ("STRING", {"forceInput": True, "multiline": True,
                    "tooltip": "Region 9+ have no dedicated input socket (fixed cap of 8) -- "
                               "type their prompts directly into each region's own prompt field."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "IMAGE", "STRING", "CONDITIONING")
    RETURN_NAMES = ("model", "clip", "mask_preview", "info", "conditioning")
    FUNCTION = "apply"
    CATEGORY = "RegioCraft"
    DESCRIPTION = ("RegioCraft: unlimited-region character LoRA/LoKr masking, sparse-token "
                   "hook, ramp/warmup scheduling, attention isolation, reference-image "
                   "identity lock, and optional per-region text conditioning, all in one node.")

    def apply(self, model, clip, canvas_width, canvas_height, regions_json, split_mode,
              seam_feather, blend_override, sparse_threshold, steps_without_applying,
              lora_ramp_calls, attention_isolation, ref_strength, ref_start_percent,
              ref_end_percent, ref_feather, base_prompt="", text_strength=1.0,
              auto_activate_from_prompt=False, bboxes=None, vae=None, base_strength=1.0,
              **kwargs):

        regions = _parse_regions(regions_json)
        cw, ch = int(canvas_width), int(canvas_height)

        # Connected region_prompt_N inputs (N = 1-based position in the JSON /
        # canvas, matching the "region N" labels) override that region's typed-
        # in prompt text. A region with a live prompt connection but no LoRA and
        # no manually-typed text still counts (text alone is a valid region).
        for i, r in enumerate(regions):
            connected = kwargs.get(f"region_prompt_{i + 1}")
            if connected:
                r["prompt"] = connected

        def has_lora(r):
            return r["lora"] not in ("None", "") and (r["strength"] * base_strength) != 0.0

        def has_ref(r):
            return bool(r["ref_image"])

        def is_enabled(r):
            # A region with a trigger name is auto-activated purely by whether
            # that word appears in base_prompt when auto_activate is on -- its
            # own enable toggle is ignored in that case (Gorecheese's "named
            # character" idea). Regions without a trigger (many LoRAs need no
            # trigger word at all) keep working exactly as before: the manual
            # enable toggle decides.
            if auto_activate_from_prompt and r["trigger"]:
                return r["trigger"].lower() in (base_prompt or "").lower()
            return r["enable"]

        active = [r for r in regions if is_enabled(r) and (has_lora(r) or has_ref(r) or r["prompt"])]
        if not active:
            logging.warning("[RegioCraft] no active regions; passing model through unchanged.")
            blank = torch.zeros((1, 64, 64, 3))
            base_cond = _encode_text(clip, base_prompt)
            return (model, clip, blank, "RegioCraft: no active regions.", base_cond)

        if split_mode == "manual":
            # boxes come from the in-node visual editor, stored per-region
            # in regions_json (x/y/w/h), already parsed into 'box' above.
            norm_boxes = [r["box"] for r in active]
        elif split_mode == "bbox":
            frame = _flatten_bboxes(bboxes)
            if frame:
                norm_boxes = [_coerce_bbox_norm(frame[i], cw, ch) if i < len(frame)
                              else (0.0, 0.0, 1.0, 1.0) for i in range(len(active))]
            else:
                logging.warning("[RegioCraft] split_mode=bbox but no boxes wired; "
                                "falling back to auto_vertical.")
                norm_boxes = _auto_split_norm(len(active), "auto_vertical")
        else:
            norm_boxes = _auto_split_norm(len(active), split_mode)

        # -- load LoRA/LoKr matrices per active region -----------------
        file_cache = {}
        prepared_regions = []
        for r in active:
            mats = {}
            if has_lora(r):
                path = _resolve_lora_path(r["lora"])
                if path not in file_cache:
                    file_cache[path] = _load_lora_matrices(path)
                base_mats = file_cache[path]
                s = r["strength"] * float(base_strength)
                mats = {sig: {**d, "scale": d["scale"] * s} for sig, d in base_mats.items()}
                if not mats:
                    logging.warning("[RegioCraft] '%s' matched 0 LoRA/LoKr layers.", r["lora"])
            prepared_regions.append({"name": r["name"], "lora_path": r["lora"],
                                     "strength": r["strength"], "mats": mats,
                                     "ref_image": r["ref_image"], "prompt": r["prompt"],
                                     "trigger": r["trigger"]})

        patched = model.clone()
        session = _RegioCraftSession(
            patched, prepared_regions, norm_boxes, seam_feather, blend_override,
            sparse_threshold, steps_without_applying, lora_ramp_calls, attention_isolation)

        def wrapper(executor, *args, **kwargs):
            return session.run(executor, *args, **kwargs)

        if hasattr(patched, "add_wrapper_with_key"):
            patched.add_wrapper_with_key(_WRAPPER_ENUM, WRAPPER_KEY, wrapper)
        elif hasattr(patched, "add_wrapper"):
            patched.add_wrapper(_WRAPPER_ENUM, wrapper)
        else:
            raise RuntimeError("This ComfyUI build lacks model wrapper support. Update ComfyUI.")

        # -- reference lock (post-cfg latent mold), same idea as v3 --------
        ref_entries = []
        n_refs_wanted = sum(1 for r in prepared_regions if r["ref_image"])
        if n_refs_wanted and vae is None:
            logging.warning("[RegioCraft] %d region(s) have reference images but no VAE "
                            "wired; reference guidance skipped.", n_refs_wanted)
        elif n_refs_wanted and float(ref_strength) > 0.0 and float(ref_end_percent) > float(ref_start_percent):
            for i, r in enumerate(prepared_regions):
                if not r["ref_image"]:
                    continue
                try:
                    img = _load_ref_image_tensor(r["ref_image"])
                    ref_latent = _encode_reference(patched, vae, img)
                except Exception as e:
                    logging.warning("[RegioCraft] could not load/encode ref '%s': %s",
                                    r["ref_image"], e)
                    continue
                ref_entries.append((norm_boxes[i], ref_latent, r["name"]))

        if ref_entries:
            sigma_start, sigma_end = _sigma_window(patched, ref_start_percent, ref_end_percent)
            w, fth = float(ref_strength), float(ref_feather)
            state = {"key": None, "built": []}

            def post_cfg(args):
                denoised = args["denoised"]
                if denoised.dim() != 4 or not _in_window(args["sigma"], sigma_start, sigma_end):
                    return denoised
                C, H, W = denoised.shape[1], denoised.shape[2], denoised.shape[3]
                if state["key"] != (C, H, W):
                    built = []
                    for box, ref_ms, _name in ref_entries:
                        mm = _build_mold(ref_ms, box, C, H, W, fth, denoised.device)
                        if mm is not None:
                            built.append(mm)
                    state["built"] = built
                    state["key"] = (C, H, W)
                if not state["built"]:
                    return denoised
                d32 = denoised.float()
                for mold, mask in state["built"]:
                    d32 = d32 + (w * mask) * (mold - d32)
                return d32.to(denoised.dtype)

            patched.set_model_sampler_post_cfg_function(post_cfg)

        # -- rainbow mask preview -------------------------------------------
        latent_w = max(4, int(math.ceil(cw / 16)))
        latent_h = max(4, int(math.ceil(ch / 16)))
        preview = torch.zeros((1, ch, cw, 3), dtype=torch.float32)
        for i, box in enumerate(norm_boxes):
            hue = (i / max(1, len(norm_boxes))) * 360.0
            color = _hsv_to_rgb(hue, 0.75, 0.9)
            x0, y0, x1, y1 = box
            px0, py0 = int(x0 * cw), int(y0 * ch)
            px1, py1 = max(px0 + 1, int(x1 * cw)), max(py0 + 1, int(y1 * ch))
            for c in range(3):
                preview[0, py0:py1, px0:px1, c] = torch.maximum(
                    preview[0, py0:py1, px0:px1, c],
                    torch.tensor(color[c] * 0.5))

        info = json.dumps({
            "regions": [{"name": r["name"], "lora": r["lora_path"], "strength": r["strength"],
                        "has_ref": bool(r["ref_image"]), "layers_matched": len(r["mats"])}
                       for r in prepared_regions],
            "engine": "activation_delta(sparse)+attention_isolation+latent_mold",
            "n_regions": len(active), "n_refs": len(ref_entries),
        }, indent=2)

        logging.info("[RegioCraft] armed %d region(s), %d with reference lock.",
                     len(active), len(ref_entries))

        region_conditioning = _build_region_conditioning(
            clip, base_prompt, prepared_regions, norm_boxes, cw, ch, text_strength)

        return (patched, clip, preview, info, region_conditioning)


def _hsv_to_rgb(h, s, v):
    import colorsys
    return colorsys.hsv_to_rgb(h / 360.0, s, v)


WEB_DIRECTORY = "./web"
NODE_CLASS_MAPPINGS = {
    "RegioCraft": RegioCraft,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RegioCraft": "RegioCraft (Regional Multi-LoRA + Ref Lock)",
}
