"""
identity_provider.py -- model-agnostic Identity-Provider interface for RegioCraft,
plus a Krea2Edit adapter around comfyui-krea2edit's krea2_edit_forward().

STATUS (2026-08-02): Phase 1+2 ONLY. Standalone, independently importable and
testable. NOT YET wired into _RegioCraftSession / the DIFFUSION_MODEL wrapper.
Diagnostic logging (see __init__.py [RegioCraft][DEBUG] prints) confirmed that
stacking Krea2EditModelPatch's own wrapper underneath/above RegioCraft's wrapper
on WrappersMP.DIFFUSION_MODEL silently kills RegioCraft's session (hooks never
registered, attention-isolation never applied, reference-lock never fires) --
Krea2EditModelPatch's wrapper never calls its own `executor`, so whichever
wrapper it would have delegated to just never runs. Two wrappers cannot both
own the full diffusion forward. This module is the first step toward RegioCraft
owning that forward alone, with identity-preservation as swappable logic
instead of a second competing patch.

DESIGN NOTES (agreed with human + GPT cross-check, 2026-08-02):
  - A provider replaces the FULL transformer forward ONCE per model call --
    never once per region. A DiT should only run once per denoising step;
    running it per-region would be prohibitively expensive and could produce
    inconsistent attention state between regions.
  - The interface is deliberately NOT Krea2-specific (no `source_latent` /
    `source_latent_b` naming at this layer) so a future Flux-Edit / Qwen-Image-
    Edit / Krea3-Edit provider can be swapped in without touching RegioCraft.
  - Krea2EditProvider caps at max_refs=2 (scene+subject) because that's the
    ONLY reference-count the krea2_identity_edit LoRA was actually trained on.
    Do not raise this cap without new training evidence -- more source frames
    is untested territory, not a documented capability.
"""
import os
import sys
import importlib
import importlib.util
import logging


def _grounding_template(n_images, system_prompt=""):
    """Same template comfyui-krea2edit's Krea2EditGroundedEncode uses -- the
    krea2_edit LoRA was trained with the instruction encoded TOGETHER with the
    source image through Qwen3-VL (vision tokens + instruction, one shared
    system prompt). Duplicated here (not imported) so this keeps working even
    if comfyui-krea2edit reshuffles its internal class layout -- only the
    template STRING itself has to stay in sync, and that's frozen by the
    LoRA's training data, not something that changes casually.
    (2026-08-18, edit-instruction grounding for RegioCraft ref+prompt regions.)
    """
    sp = (system_prompt or "").strip() or (
        "Describe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
    )
    vis = "<|vision_start|><|image_pad|><|vision_end|>" * n_images
    return ("<|im_start|>system\n" + sp + "<|im_end|>\n<|im_start|>user\n"
            + vis + "{}<|im_end|>\n<|im_start|>assistant\n")


def _prep_grounding_image(img, grounding_px=768):
    """Cap the longest side fed to the VLM, same as Krea2EditGroundedEncode's
    own _prep(). img is an IMAGE tensor [1,H,W,3]."""
    h, w = img.shape[1], img.shape[2]
    if grounding_px and max(h, w) > grounding_px:
        import comfy.utils
        s = grounding_px / max(h, w)
        samples = img.movedim(-1, 1)
        samples = comfy.utils.common_upscale(samples, round(w * s), round(h * s), "area", "disabled")
        img = samples.movedim(1, -1)
    return img[:, :, :, :3]


def build_grounded_edit_context(clip, prompt, images, grounding_px=768, system_prompt=""):
    """Encode an edit instruction grounded on reference image(s) -- the
    training-matched SEMANTIC path of krea2_edit (mirrors comfyui-krea2edit's
    Krea2EditGroundedEncode node exactly: same template, same tokenize(images=,
    llama_template=) call).

    Returns the raw cond tensor (the same format RegioCraft's own
    `_encode_text()` in __init__.py already produces), NOT a CONDITIONING list.
    This is injected directly as the diffusion model's `context` argument
    from INSIDE the DIFFUSION_MODEL wrapper -- there is no hook to influence
    ComfyUI's normal CONDITIONING->context batching from in here (that step
    already ran, on whatever was wired into KSampler, before the wrapper is
    ever called), so grounding on a per-region ref+prompt combo means
    reproducing that one encode step manually and swapping it in.
    """
    imgs = [_prep_grounding_image(im, grounding_px) for im in images]
    template = _grounding_template(len(imgs), system_prompt)
    tokens = clip.tokenize(prompt or "", images=imgs, llama_template=template)
    cond, _pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return cond


class IdentityProvider:
    """Minimal, model-agnostic identity-forward interface.

    refs: list of dicts, each:
        {
          "latent": <LATENT dict {"samples": tensor} or a raw tensor>,
          "image":  <IMAGE tensor, optional -- pixel-path, Phase 3 scope>,
          "boost":  <float, optional -- reference-fidelity dial, provider-specific>,
          "boost_mask": <MASK tensor, optional -- e.g. restrict boost to a face>,
          "role":   <str, optional, provider-specific hint e.g. "scene"/"subject">,
          "prompt": <str, optional (2026-08-18) -- per-ref edit instruction, e.g.
                     "give him a hooked nose". Grounded on this ref's own image
                     via build_grounded_edit_context() and used as the model's
                     `context` for this call, REPLACING whatever conditioning
                     was wired into KSampler. Empty/absent = old behavior,
                     unchanged: pure identity/face-lock, no edit applied.>,
        }
    """

    #: hard cap on simultaneous refs this provider was actually trained/tested
    #: for. Callers (RegioCraft) must not exceed this even if they themselves
    #: manage more regions internally -- raise ValueError instead of silently
    #: truncating, so the caller has to make an explicit choice.
    max_refs = 1

    def forward(self, model, x, timesteps, context, *, transformer_options, refs, vae=None, clip=None):
        raise NotImplementedError


def _load_krea2edit_module():
    """Locate comfyui-krea2edit's package independent of how ComfyUI's
    custom-node loader registered it in sys.modules (a folder name with a
    hyphen isn't guaranteed reachable via a normal `import` statement, so we
    don't rely on that)."""
    for candidate in ("comfyui-krea2edit", "comfyui_krea2edit", "ComfyUI-Krea2Edit"):
        mod = sys.modules.get(candidate)
        if mod is not None and hasattr(mod, "krea2_edit_forward"):
            return mod

    here = os.path.dirname(os.path.abspath(__file__))       # .../custom_nodes/RegioCraft
    custom_nodes_dir = os.path.dirname(here)                # .../custom_nodes
    for folder in ("comfyui-krea2edit", "ComfyUI-Krea2Edit", "comfyui_krea2edit"):
        init_path = os.path.join(custom_nodes_dir, folder, "__init__.py")
        if os.path.isfile(init_path):
            try:
                spec = importlib.util.spec_from_file_location(
                    "_regiocraft_krea2edit_dep", init_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "krea2_edit_forward"):
                    return mod
            except Exception as e:
                logging.info("[RegioCraft][identity_provider] could not load %s "
                             "standalone (%s) -- trying next candidate.",
                             init_path, e)
                continue
    return None


class Krea2EditProvider(IdentityProvider):
    """Adapter around comfyui-krea2edit's krea2_edit_forward(). Deliberately
    capped at 2 refs (scene=frame1, subject=frame2) -- that's the only
    reference count the krea2_identity_edit LoRA was trained on."""

    max_refs = 2

    def __init__(self):
        mod = _load_krea2edit_module()
        if mod is None:
            raise RuntimeError(
                "[RegioCraft] Krea2EditProvider: could not locate comfyui-krea2edit's "
                "krea2_edit_forward(). Is the comfyui-krea2edit custom node installed "
                "and loaded?")
        self._krea2_edit_forward = mod.krea2_edit_forward
        self._mod = mod
        # Persistent per-target-resolution pixel-encode cache (2026-08-02, blur fix).
        # Mirrors Krea2EditModelPatch's own px_cache: encodes the reference image
        # ONCE per (resolution, fit_mode) combo, reused across every sampling step
        # for the life of this provider instance (one per RegioCraft session/generation).
        self._px_cache = {}
        # 2026-08-20 perf fix: grounded-edit context (Qwen3-VL vision-language
        # encode of ref image(s) + instruction) does NOT depend on x/timesteps --
        # only on (clip, prompt, ref images), all fixed for the life of one
        # generation. Without caching, forward() re-ran the full 4B-parameter
        # VLM encode on EVERY sampling step (x2 for cond/uncond CFG passes) --
        # e.g. 10 steps = 20 redundant VLM forward passes instead of 1.
        self._grounded_context_cache = None
        self._grounded_context_cache_key = None

    def forward(self, model, x, timesteps, context, *, transformer_options, refs, vae=None, clip=None):
        if not refs:
            raise ValueError("[RegioCraft] Krea2EditProvider.forward() called with no refs.")
        if len(refs) > self.max_refs:
            raise ValueError(
                f"[RegioCraft] Krea2EditProvider supports at most {self.max_refs} "
                f"reference(s) (scene+subject, per its training); got {len(refs)}. "
                f"More refs is untested territory -- see 2026-08-02 architecture note "
                f"before raising this cap.")

        # -- edit-instruction grounding (2026-08-18) ------------------------
        # A ref with BOTH an image and non-empty 'prompt' text wants an actual
        # edit ("give him a hooked nose"), not just identity lock. That needs
        # the instruction encoded TOGETHER with the ref image through the
        # training-matched grounding template -- plain identity/face-lock
        # (image + boost, no prompt) is completely unaffected by this branch.
        edit_prompts = [(r.get("prompt") or "").strip() for r in refs]
        wants_edit = any(edit_prompts)
        if wants_edit and clip is not None:
            ref_images = [r["image"] for r in refs if r.get("image") is not None]
            # Multiple simultaneous refs each with their own instruction get
            # concatenated into one combined instruction -- krea2_edit_forward
            # runs the diffusion model ONCE per call with ONE context, there is
            # no per-ref instruction slot at that level (see identity_provider.py
            # module docstring / 2026-08-18 RegioCraft README note).
            combined_prompt = " ".join(p for p in edit_prompts if p)
            cache_key = (id(clip), combined_prompt, tuple(id(im) for im in ref_images))
            try:
                if self._grounded_context_cache is not None and self._grounded_context_cache_key == cache_key:
                    context = self._grounded_context_cache
                else:
                    context = build_grounded_edit_context(clip, combined_prompt, ref_images)
                    self._grounded_context_cache = context
                    self._grounded_context_cache_key = cache_key
                # 2026-08-20 device fix: clip.encode_from_tokens' output device isn't
                # guaranteed to match the diffusion model's compute device (x) -- krea2edit's
                # own torch.cat([context] + src_imgs + [tgt_img]) requires all three to
                # match, and this is the first real end-to-end path this branch has run.
                context = context.to(x.device, x.dtype)
                logging.info("[RegioCraft] identity_provider='krea2edit': grounded edit "
                            "instruction active (%d ref image(s), prompt=%r).",
                            len(ref_images), combined_prompt)
            except Exception as e:
                logging.warning("[RegioCraft] grounded edit-instruction encode failed (%s); "
                                "falling back to the model's normal conditioning for this "
                                "call (identity lock, if any, still applies).", e)
        elif wants_edit and clip is None:
            logging.warning("[RegioCraft] a region has a ref_image + edit prompt but no "
                            "'clip' reached the identity provider; the edit instruction is "
                            "being ignored this run. (Should not happen via RegioCraft's own "
                            "node -- clip is wired automatically; only relevant if calling "
                            "this provider directly.)")

        # PIXEL PATH (2026-08-02, blur fix): if every ref carries a raw 'image' and a
        # vae is connected, use krea2_edit_forward's own '_fit_encode_image' -- the
        # blur-proof pixel-space resample-then-encode the LoRA was actually trained
        # against (ref_native=True, pos_mode='stride1'). This matches
        # Krea2EditModelPatch's own fit_mode='fit' behavior exactly.
        use_pixel_path = vae is not None and all(r.get("image") is not None for r in refs)

        if use_pixel_path:
            Hh, Ww = x.shape[-2], x.shape[-1]
            src_list = []
            for i, r in enumerate(refs):
                cache_key = (id(r["image"]), Hh, Ww, i)
                lat = self._mod._fit_encode_image(
                    r["image"], vae, Hh, Ww, self._px_cache, cache_key, "fit")
                src_list.append(model.model.process_latent_in(lat))
            src = src_list[0] if len(src_list) == 1 else src_list
            ref_native, pos_mode = True, "stride1"
        else:
            src_list = []
            for r in refs:
                lat = r.get("latent")
                if lat is None:
                    raise ValueError(
                        "[RegioCraft] Krea2EditProvider: each ref needs either an "
                        "'image' (with a vae connected to RegioCraft) or a pre-encoded "
                        "'latent'. Neither was provided for one of the refs.")
                samples = lat["samples"] if isinstance(lat, dict) else lat
                src_list.append(model.model.process_latent_in(samples))
            src = src_list[0] if len(src_list) == 1 else src_list
            ref_native, pos_mode = False, "anchor"

        boosts = [r.get("boost", 1.0) for r in refs]
        ref_boost = boosts[-1] if boosts else 1.0
        ref_boost_a = boosts[0] if len(boosts) > 1 else 1.0
        ref_boost_mask = refs[-1].get("boost_mask") if refs else None

        dm = getattr(model.model, "diffusion_model", model.model)
        return self._krea2_edit_forward(
            dm, x, timesteps, context, src, transformer_options,
            ref_boost=ref_boost, ref_boost_a=ref_boost_a,
            ref_boost_mask=ref_boost_mask, ref_native=ref_native, pos_mode=pos_mode,
        )


_REGISTRY = {"krea2edit": Krea2EditProvider}


def get_default_provider(name="krea2edit"):
    """Small registry indirection so RegioCraft's future integration point
    doesn't hardcode a class name. Returns None (never raises) if the named
    provider's dependency isn't available -- callers must fall back to the
    normal forward path in that case, never hard-fail a whole generation
    because an optional identity provider couldn't load."""
    cls = _REGISTRY.get(name)
    if cls is None:
        logging.info("[RegioCraft] identity provider '%s' not in registry.", name)
        return None
    try:
        return cls()
    except Exception as e:
        logging.info("[RegioCraft] identity provider '%s' unavailable: %s", name, e)
        return None
