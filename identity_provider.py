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


class IdentityProvider:
    """Minimal, model-agnostic identity-forward interface.

    refs: list of dicts, each:
        {
          "latent": <LATENT dict {"samples": tensor} or a raw tensor>,
          "image":  <IMAGE tensor, optional -- pixel-path, Phase 3 scope>,
          "boost":  <float, optional -- reference-fidelity dial, provider-specific>,
          "boost_mask": <MASK tensor, optional -- e.g. restrict boost to a face>,
          "role":   <str, optional, provider-specific hint e.g. "scene"/"subject">,
        }
    """

    #: hard cap on simultaneous refs this provider was actually trained/tested
    #: for. Callers (RegioCraft) must not exceed this even if they themselves
    #: manage more regions internally -- raise ValueError instead of silently
    #: truncating, so the caller has to make an explicit choice.
    max_refs = 1

    def forward(self, model, x, timesteps, context, *, transformer_options, refs, vae=None):
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

    def forward(self, model, x, timesteps, context, *, transformer_options, refs, vae=None):
        if not refs:
            raise ValueError("[RegioCraft] Krea2EditProvider.forward() called with no refs.")
        if len(refs) > self.max_refs:
            raise ValueError(
                f"[RegioCraft] Krea2EditProvider supports at most {self.max_refs} "
                f"reference(s) (scene+subject, per its training); got {len(refs)}. "
                f"More refs is untested territory -- see 2026-08-02 architecture note "
                f"before raising this cap.")

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
