# RegioCraft

**Put unlimited character LoRAs into one image, each locked to its own hand-drawn box — plus reference-image identity lock, cross-region attention dampening, per-region text conditioning, trigger-name auto-activation, and a sparse-token engine that only pays for the tokens that matter.**

RegioCraft is a ComfyUI custom node for Krea2 / Flux.2-Klein. It's the "best of three" merge of three earlier regional-LoRA nodes, combined into one, with a built-in visual editor so you never touch raw JSON.

## Why this exists

A normal LoRA stack applies every LoRA **everywhere, uniformly**. Load two character LoRAs at once and ask for "Alice on the left, Bob on the right" and you get one face that's a blend of both, in both spots. RegioCraft removes that permission entirely: every regional LoRA's contribution is masked to its own box, and multiplied by zero everywhere else. There's no mathematical path for a LoRA to bleed outside its region — it's not discouraged from leaking, it's structurally incapable of it.

## What it does

- **Unlimited regions.** Two characters or ten — draw a box, hit "+ Add Region", assign a LoRA. No fixed slots.
- **LoRA *and* LoKr support.** Standard kohya/diffusers LoRAs (`lora_down/up`, `lora_A/B`) and Kronecker-factor LoKr files (`lokr_w1/w2`, including decomposed `w1_a@w1_b` variants) both load straight into the same region slot.
- **In-node visual editor.** Drag boxes, resize corners, assign a LoRA + strength per region, toggle regions on/off, upload an optional reference image per region — all without leaving the node. **A box's position and its region's settings (LoRA, strength, prompt, trigger) are inseparable** — drag a box anywhere on the canvas and its LoRA moves with it. There's no separate "which region is which slot" concept to get confused by; wherever you drop a box is where that region's LoRA gets injected, full stop.
- **"Load latest output" workflow.** Pull the last generated image straight into the editor as a background (or drop one in, or enable auto-refresh after every run), line your boxes up exactly on the faces that came out, then re-run the same seed at full strength.
- **Sparse-token engine.** Only the image tokens whose regional mask is above `sparse_threshold` pay for the LoRA/LoKr matmul — not the whole sequence. Faster with more regions, not slower.
- **Attention Isolation.** An additive cross-region attention-logit bias, so a region's query tokens attend *less* to another region's key/value tokens — dampening identity bleed at the attention level, not just the LoRA-delta level. Multi-architecture (Krea2, Flux.1); text tokens always keep full, unrestricted attention.
- **Reference Lock.** Give any region a reference image and RegioCraft VAE-encodes it once, resizes it into that region's box on the latent grid (the "mold"), then nudges the denoised prediction toward it every sampling step inside a scheduled window. Anchors identity across seeds and across generations — something LoRA-delta masking alone can't do.
- **Ramp / warmup scheduling.** `steps_without_applying` lets the base composition establish itself for N model calls before regional LoRAs kick in; `lora_ramp_calls` then ramps them up gradually instead of slamming full strength on call 1.
- **Per-region text conditioning.** Give any region its own text (in addition to, or instead of, a LoRA) and RegioCraft encodes it together with the overall scene prompt, then masks that conditioning to the region's box using ComfyUI's standard area-conditioning fields. Lets you combine a LoRA-locked identity with freeform text description in the same box, or describe a character purely in words with no LoRA at all.
- **Auto-activate from trigger names.** Give a region a `trigger` name and turn on `auto_activate_from_prompt`, and that region switches on automatically whenever its name appears in `base_prompt` — write "Alice and Bob talking" and the regions named Alice and Bob activate on their own, no manual toggling. Regions with no trigger set just keep using their manual enable toggle, since plenty of LoRAs don't need a trigger word at all. Inspired by a conversation with Gorecheese about balancing named, language-described characters alongside character LoRAs in the same node.
- **fp8-safe.** Works on quantized Krea2 Turbo checkpoints, where the native ComfyUI hook-LoRA path crashes (`'Linear' object has no attribute 'weight_scale'`) — RegioCraft only touches activations, never quantized weights.

---
☕ **Support the Zeus Project**
If this model's been useful to you and you feel like buying me a coffee: [paypal.me/werk12](https://paypal.me/werk12)
Helps me keep building Zeus Project stuff and pushing out new models/nodes.
---

## Install

1. Copy the `RegioCraft/` folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.
3. Add node: **RegioCraft → "RegioCraft (Regional Multi-LoRA + Ref Lock)"**.

## Minimal workflow

```
UNETLoader ─┐
CLIPLoader ─┴─► (global style LoRAs, if any) ─► RegioCraft ──► model ──► KSampler
                                                          └──► conditioning ──► KSampler (positive)
```

Character LoRAs go into RegioCraft's own region rows, not into a global LoRA stack before it. Wire the `conditioning` output into your KSampler's positive input if you're using `base_prompt` and/or per-region prompts/triggers — a plain `CLIPTextEncode` upstream of RegioCraft is not enough on its own, since the region-masked text conditioning is built inside this node.

## Outputs

| Output | What it is |
|---|---|
| `model` | The patched model — wire this to KSampler as usual. |
| `clip` | Passed through unchanged; wire to anything downstream that still needs raw CLIP. |
| `mask_preview` | A rainbow-coded IMAGE showing exactly where each region's box currently sits. Handy for sanity-checking overlaps and coverage. |
| `info` | A JSON string summarizing what got armed this run — region names, LoRAs, strengths, whether each matched any layers, how many reference locks fired. |
| `conditioning` | The combined base + per-region masked text conditioning. Wire this to KSampler's positive input if you're using `base_prompt`, per-region `prompt` fields, or trigger auto-activation. |

## Using it

- Draw a box per region on the canvas, or drag existing ones around. The box you drop is the box that region uses — order/numbering never matters, only where you physically place it.
- Each region row: **enable** toggle, **LoRA** dropdown, **strength**, an optional **prompt** text field, an optional **trigger name** (for auto-activation), and an optional **reference image** (click to upload, click again to replace, ✕ to clear).
- **`base_prompt`** (top of the node): the overall scene description — composition, setting, lighting. Combined with each region's own text (if any) and masked to that region for the `conditioning` output. Also scanned for trigger names if `auto_activate_from_prompt` is on.
- **`auto_activate_from_prompt`**: off by default. When on, any region with a `trigger` name set is switched on/off purely by whether that name appears in `base_prompt` — its own enable toggle is ignored. Regions with no trigger keep using their manual toggle regardless.
- **"↻ Load latest output"** pulls your last generated image into the editor as a background so you can line boxes up precisely on where faces actually landed; **"auto after each run"** does this automatically after every generation.
- **Dynamic `region_prompt_N` inputs** — real, connectable STRING sockets (up to 8) that appear automatically above the node, one per region currently drawn on the canvas. Wire an external node (a wildcard prompt generator, a character-clothes generator, whatever) into `region_prompt_1` and it overrides that region's typed-in prompt field live. Add or remove a region and the matching socket appears/disappears (auto-disconnecting) to match. For text that's fine applying to the whole scene rather than one character's box specifically, a simpler option is a `Text Concatenate`-style node feeding `base_prompt` directly -- no per-region socket needed for that case.
- `split_mode`: `manual` (the boxes you draw, default) / `bbox` (wire in an external `BOUNDING_BOX` source) / `auto_vertical` / `auto_horizontal` (equal strips, no boxes needed). The little "BoundingBox: Node 2.0 only" hint some ComfyUI frontends show next to the `bboxes` input is just a cosmetic frontend note about a newer node schema RegioCraft doesn't use — it's inert, ignore it, and it never affects anything as long as you're on `split_mode = manual`.
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

**One character is right, the other generic** — the box for that region probably doesn't cover its face; drag the box, not the LoRA — a region's identity always follows wherever its box currently sits, so if the wrong face is getting the wrong LoRA, just move the box, don't try to reorder anything.

**"Failed to validate prompt" errors after updating RegioCraft (values landing in the wrong fields, e.g. a float where a dropdown choice was expected)** — this happens when RegioCraft's input fields were reordered/added in an update and an *existing* node instance on your canvas still holds its old, position-based saved values. ComfyUI stores widget values positionally per node instance, not by name, so a field-order change can shuffle a stale instance's values into the wrong slots. Fix: delete the old RegioCraft node from your workflow and drag in a fresh one, then rewire it. This is a one-time inconvenience per update that touches the field order, not a bug in a given run.

## Changelog

**Core merge (initial release)**
- Merged three parent nodes into one: activation-delta LoRA/LoKr masking engine (Gorecheese), sparse-token hook + Attention Isolation + ramp/warmup scheduling (Shy), unlimited regions + LoKr support + Reference Lock (Fedor)
- fp8-safe LoRA/LoKr injection via forward hooks, no weight merging

**Visual editor**
- In-node canvas: draw/drag/resize boxes directly on the node, rainbow-coded per region
- Per-region control rows: enable toggle, LoRA dropdown, strength, reference image upload with thumbnail preview
- "+ Add Region" button, unlimited regions (not capped at 2)
- "↻ Load latest output" + "auto after each run" -- pulls the last generation into the canvas as a background reference for precise box placement
- Drag-and-drop image support directly onto the canvas

**Per-region text conditioning**
- `base_prompt` input (overall scene prompt, masked-conditioning base)
- `text_strength` control for the masked regional conditioning blend
- New `conditioning` output -- combines base + per-region masked text via ComfyUI's standard area-conditioning mechanism (architecture-agnostic)
- Per-region `prompt` field (optional per-character text)
- Per-region `trigger` field + `auto_activate_from_prompt` toggle -- a region auto-enables when its trigger name appears in `base_prompt`

**Bug fixes**
- Fixed: region data (LoRAs, triggers, prompts) could silently vanish when switching between ComfyUI tabs. The JSON-seeding logic now only resets to defaults when the stored value is truly absent (`undefined`/`null`/`""`), never just because it fails to parse or looks empty -- that state can also happen transiently during multi-tab canvas restore

**Dedicated per-region prompt inputs**
- Added `region_prompt_1` through `region_prompt_8` -- real, connectable STRING inputs (fixed cap of 8) that override a region's typed-in prompt when wired to an external node (e.g. a wildcard/character generator)
- Socket count now auto-syncs to the number of regions actually drawn on the canvas (2 regions = 2 visible sockets, add a 3rd -> `region_prompt_3` appears automatically, remove a region -> its socket disappears and auto-disconnects)

## Credits

RegioCraft is a merge of three earlier nodes, and stands entirely on the work of the people who built them:

- **Gorecheese** — created the original idea and node, `regional_character_lora`: the activation-delta masking engine this whole approach is built on.
- **Shy** — forked and extended that work: the sparse-token hook, Attention Isolation, and the ramp/warmup scheduling.
- **Fedor** — built `Krea2-Multi-Character-Lora-Node-w-bounding-box`: unlimited regions, LoKr support, and Reference Lock (latent-mold identity anchoring).

Thanks to all three — none of this exists without the groundwork they laid.

## License

MIT, in keeping with the parent nodes it merges.
