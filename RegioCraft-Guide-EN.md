# RegioCraft — Guide

**Unlimited character LoRAs in one image, each locked to its own box — no identity blend.**

RegioCraft is a ComfyUI custom node for Krea2 / Flux.2-Klein. You draw boxes on the canvas, assign a LoRA to each box, and each LoRA only takes effect inside its own box. No merged faces, no half-identities.

## What you need

- **Krea2 / Flux.2-Klein** diffusion model (fp8 Turbo works)
- **One trained character LoRA (or LoKr) per character** — this is mandatory. RegioCraft doesn't generate an identity on its own; it only makes sure whatever LoRA you load stays cleanly confined to its box. Without a LoRA in a region, that box either has no locked identity at all, or only whatever the text prompt alone produces.
- Standard format: kohya (`lora_up/down`) or diffusers (`lora_A/B`), plus LoKr (`lokr_w1/w2`)

## How it works (short version)

A normal LoRA stack affects the **entire image at once**. Load two character LoRAs together and ask for "left" and "right," and both faces blend into a mismatch regardless of where you place them in the prompt.

RegioCraft fixes this structurally: each LoRA is never merged into the model — it's injected as its own activation delta at runtime, and only into the image tokens that fall inside its assigned box. Outside the box, the effect is exactly zero. There's no computational path for a LoRA to bleed outside its region.

## Installation

1. Copy the `RegioCraft/` folder into `ComfyUI/custom_nodes/`
2. Restart ComfyUI
3. Add node: **RegioCraft → "RegioCraft (Regional Multi-LoRA + Ref Lock)"**

## Minimal workflow

```
UNETLoader ─┐
CLIPLoader ─┴─► RegioCraft ──► model ──► KSampler
                          └──► conditioning ──► KSampler (positive)
```

Character LoRAs go into RegioCraft's own region rows, not into a global LoRA stack upstream of it. The `conditioning` output must go to KSampler's positive input as soon as you use `base_prompt` or region prompts/triggers — a plain `CLIPTextEncode` upstream isn't enough on its own.

## Using it

1. Draw a box per character on the canvas (or move existing ones)
2. Per region row, set: **enable** toggle, **LoRA** dropdown, **strength**, optional **prompt** text, optional **trigger** name, optional **reference image** upload
3. Wherever the box sits, that's where the LoRA takes effect — order/numbering never matters

**The one rule that matters:** The box marks where the LoRA gets injected — not just "roughly where the character is." Most character LoRAs are face-trained, so the box needs to cover the head/face. Err generous, not tight — a box that misses the face produces a weak identity.

**Unlimited regions, with one detail to know:** "+ Add Region" lets you draw as many boxes as you want, no cap. The only limit at 8: external, live-connectable `region_prompt_N` sockets (e.g. for wildcard generators) only exist for regions 1–8. From region 9 onward, you just type the prompt directly into that region's own prompt field — LoRA, reference image, and everything else work the same regardless, with no limit.

## All settings explained

| Setting | What it does |
|---|---|
| `base_prompt` | Overall scene (composition, setting, lighting). Combined with region text and masked for `conditioning`. |
| `auto_activate_from_prompt` | Off by default. When on: regions with a `trigger` name set switch on automatically as soon as that name appears in `base_prompt`. |
| `split_mode` | `manual` (drawn boxes, default) / `bbox` (external BOUNDING_BOX source) / `auto_vertical` / `auto_horizontal` (equal strips, no boxes needed) |
| `seam_feather` | Softness of the border between regions |
| `blend_override` | `0` = clean split (recommended); higher = regions blur toward a shared average |
| `sparse_threshold` | Skips near-zero tokens for speed. `0` = safest/slowest, `0.01` = practical default |
| `steps_without_applying` / `lora_ramp_calls` | Warmup scheduling — lets the base composition settle before regional identities are enforced |
| `attention_isolation` | `0` = off. Try `4`–`8` for identity bleed that LoRA masking alone doesn't stop. Past `10`, expressions can go flat. |
| `ref_strength` / `ref_start_percent` / `ref_end_percent` / `ref_feather` | Reference Lock controls. `0` strength = off. Needs a connected VAE. |
| `identity_provider` | `none` (default) or `krea2edit`. See dedicated section below. |
| `identity_ref_boost` | Only relevant when `identity_provider = krea2edit`. Reference-fidelity dial: `1.0` = off, `~4.0` = strong likeness (recommended default), `>10` = risk of over-copying. |

## Identity Provider (`krea2edit`) — reference image + prompt instruction

This is the mode you see in your test workflow: feed in a reference image, describe via prompt what should happen to that person ("recolor the jacket to red," "the person on the left waving") — genuine instruction-based identity editing, not just identity locking.

**Important to understand — this is a different mechanism from Reference Lock:**
- **Reference Lock** (`ref_strength` etc.) — VAE-encodes the reference image once, gently nudges the prediction toward it during sampling. No text instruction involved.
- **`identity_provider = krea2edit`** — for regions with a reference image set, this takes over the **entire diffusion forward pass** via the separate `comfyui-krea2edit` node pack. This is the mechanism that understands text instructions and applies them to the person.

**Requirements for `identity_provider = krea2edit`:**
- The separate custom node **`comfyui-krea2edit`** must be installed (`ComfyUI/custom_nodes/comfyui-krea2edit`)
- The **[`krea2_identity_edit` LoRA](https://civitai.com/models/2761113/krea-2-identity-edit)** must be present and loaded
- A **VAE** must be connected to the RegioCraft node (without it, the identity path stays disabled and RegioCraft falls back to normal forward automatically)
- Maximum **2 simultaneous reference images** (scene + subject) — that's the LoRA's own trained limit, not a RegioCraft restriction. Extra active reference regions beyond that are logged and ignored, not an error.

Regions using LoRA masking and Attention Isolation stay active during the `krea2edit` pass — RegioCraft remains the sole owner of the model wrapper, so no two competing patches run at once (that was exactly the bug the earlier version had).

## Reference Lock — what it adds

Give a region a reference image, and RegioCraft VAE-encodes it once, fits it into that region's box on the latent grid, and gently pulls the prediction toward it during a scheduled window. Anchors identity across seeds and generations — something pure LoRA-delta masking can't do on its own.

## Outputs

| Output | What it is |
|---|---|
| `model` | The patched model — wire to KSampler as usual |
| `clip` | Passed through unchanged |
| `mask_preview` | Rainbow-coded image showing where each region's box currently sits |
| `info` | JSON summary of what was active this run |
| `conditioning` | Combined base + region text conditioning |

## Troubleshooting

- **Characters still look merged** → boxes are probably overlapping; shrink them or lower `seam_feather`
- **Console shows "0 layers matched"** → the LoRA/LoKr's key format doesn't match the loaded model (different architecture, or layer names changed by a heavy merge)
- **One character is right, the other generic** → that region's box probably doesn't cover its face; move the box, don't reassign it
- **"Failed to validate prompt" after an update** → delete the old node instance and drag in a fresh one, since ComfyUI stores values positionally rather than by name
