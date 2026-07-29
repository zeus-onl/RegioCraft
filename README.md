# RegioCraft

**Put unlimited character LoRAs into one image, each locked to its own hand-drawn box — plus reference-image identity lock, cross-region attention dampening, and a sparse-token engine that only pays for the tokens that matter.**

RegioCraft is a ComfyUI custom node for Krea2 / Flux.2-Klein. It's the "best of three" merge of three earlier regional-LoRA nodes, combined into one, with a built-in visual editor so you never touch raw JSON.

## Why this exists

A normal LoRA stack applies every LoRA **everywhere, uniformly**. Load two character LoRAs at once and ask for "Alice on the left, Bob on the right" and you get one face that's a blend of both, in both spots. RegioCraft removes that permission entirely: every regional LoRA's contribution is masked to its own box, and multiplied by zero everywhere else. There's no mathematical path for a LoRA to bleed outside its region — it's not discouraged from leaking, it's structurally incapable of it.

## What it does

- **Unlimited regions.** Two characters or ten — draw a box, hit "+ Add Region", assign a LoRA. No fixed slots.
- **LoRA *and* LoKr support.** Standard kohya/diffusers LoRAs (`lora_down/up`, `lora_A/B`) and Kronecker-factor LoKr files (`lokr_w1/w2`, including decomposed `w1_a@w1_b` variants) both load straight into the same region slot.
- **In-node visual editor.** Drag boxes, resize corners, assign a LoRA + strength per region, toggle regions on/off, upload an optional reference image per region — all without leaving the node.
- **"Load latest output" workflow.** Pull the last generated image straight into the editor as a background (or drop one in, or enable auto-refresh after every run), line your boxes up exactly on the faces that came out, then re-run the same seed at full strength.
- **Sparse-token engine.** Only the image tokens whose regional mask is above `sparse_threshold` pay for the LoRA/LoKr matmul — not the whole sequence. Faster with more regions, not slower.
- **Attention Isolation.** An additive cross-region attention-logit bias, so a region's query tokens attend *less* to another region's key/value tokens — dampening identity bleed at the attention level, not just the LoRA-delta level. Multi-architecture (Krea2, Flux.1); text tokens always keep full, unrestricted attention.
- **Reference Lock.** Give any region a reference image and RegioCraft VAE-encodes it once, resizes it into that region's box on the latent grid (the "mold"), then nudges the denoised prediction toward it every sampling step inside a scheduled window. Anchors identity across seeds and across generations — something LoRA-delta masking alone can't do.
- **Ramp / warmup scheduling.** `steps_without_applying` lets the base composition establish itself for N model calls before regional LoRAs kick in; `lora_ramp_calls` then ramps them up gradually instead of slamming full strength on call 1.
- **fp8-safe.** Works on quantized Krea2 Turbo checkpoints, where the native ComfyUI hook-LoRA path crashes (`'Linear' object has no attribute 'weight_scale'`) — RegioCraft only touches activations, never quantized weights.

## Install

1. Copy the `RegioCraft/` folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.
3. Add node: **RegioCraft → "RegioCraft (Regional Multi-LoRA + Ref Lock)"**.

## Minimal workflow

```
UNETLoader → (global style LoRAs, if any) → RegioCraft → KSampler
```

Character LoRAs go into RegioCraft's own region rows, not into a global LoRA stack before it.

## Using it

- Draw a box per region on the canvas, or drag existing ones around.
- Each region row: **enable** toggle, **LoRA** dropdown, **strength**, and an optional **reference image** (click to upload, click again to replace, ✕ to clear).
- **"↻ Load latest output"** pulls your last generated image into the editor as a background so you can line boxes up precisely on where faces actually landed; **"auto after each run"** does this automatically after every generation.
- `split_mode`: `manual` (the boxes you draw, default) / `bbox` (wire in an external `BOUNDING_BOX` source) / `auto_vertical` / `auto_horizontal` (equal strips, no boxes needed).
- `seam_feather`: softness of the border between regions.
- `blend_override`: `0` = clean regional split (recommended); raising it lets regions bleed toward a shared average.
- `sparse_threshold`: skip near-zero masked tokens for speed. `0` = safest/slowest, `0.01` = practical default.
- `steps_without_applying` / `lora_ramp_calls`: warm-up scheduling — let the base composition settle before regional identities are enforced.
- `attention_isolation`: `0` = off. Try `4`–`8` if you're seeing identity bleed that the LoRA masking alone doesn't fully stop; push too high (`10`+) and expressions can go flat.
- `ref_strength` / `ref_start_percent` / `ref_end_percent` / `ref_feather`: Reference Lock controls. `0` strength = off. A VAE must be wired for reference images to take effect.

## The one rule that matters

The box marks **where the LoRA is injected**, not just "where the character is." Most character LoRAs are face/portrait-trained, so the box needs to cover where the head/face lands — err generous, not tight. A box that misses the face gives a weak identity.

## Troubleshooting

**Characters still look merged** — boxes probably overlap; shrink them or lower `seam_feather`.

**Console logs "0 layers matched"** — that LoRA/LoKr's key format doesn't map onto the loaded model; it was likely trained on a different architecture, or the checkpoint changed layer names (common with heavy fine-tune merges).

**One character is right, the other generic** — check region order matches box draw order, and that the box actually covers the face.

## Credits

RegioCraft is a merge of three earlier nodes, and stands entirely on the work of the people who built them:

- **Gorecheese** — created the original idea and node, `regional_character_lora`: the activation-delta masking engine this whole approach is built on.
- **Shy** — forked and extended that work: the sparse-token hook, Attention Isolation, and the ramp/warmup scheduling.
- **Fedor** — built `Krea2-Multi-Character-Lora-Node-w-bounding-box`: unlimited regions, LoKr support, and Reference Lock (latent-mold identity anchoring).

Thanks to all three — none of this exists without the groundwork they laid.

## License

MIT, in keeping with the parent nodes it merges.
