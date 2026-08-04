# RegioCraft — Anleitung

**Beliebig viele Charakter-LoRAs in einem Bild, jeder fest an seine eigene Box gebunden — ohne Identity-Blend.**

RegioCraft ist ein ComfyUI Custom Node für Krea2 / Flux.2-Klein. Du zeichnest Boxen auf die Canvas, weist jeder Box eine LoRA zu, und jede LoRA wirkt ausschließlich innerhalb ihrer eigenen Box. Kein Vermischen der Gesichter, keine halben Identitäten.

## Was du brauchst

- **Krea2 / Flux.2-Klein** Diffusion Model (fp8 Turbo funktioniert)
- **Pro Charakter eine trainierte Charakter-LoRA (oder LoKr)** — das ist zwingend. RegioCraft erzeugt keine Identität von selbst; es sorgt nur dafür, dass die LoRA, die du reinlädst, sauber auf ihre Box begrenzt bleibt. Ohne LoRA in einer Region bekommst du in dieser Box entweder gar keine feste Identität oder nur das, was der Text-Prompt allein hergibt.
- Standard-Format: kohya (`lora_up/down`) oder diffusers (`lora_A/B`), sowie LoKr (`lokr_w1/w2`)

## Wie es funktioniert (kurz erklärt)

Ein normaler LoRA-Stack wirkt **überall gleichzeitig** auf das ganze Bild. Lädst du zwei Charakter-LoRAs gleichzeitig, verschmelzen beide Gesichter zu einem Mischmasch, egal wo du "links" oder "rechts" hinschreibst.

RegioCraft löst das strukturell: Jede LoRA wird nicht ins Modell gemergt, sondern als eigener Aktivierungs-Delta zur Laufzeit injiziert — und zwar nur in die Bild-Tokens, die innerhalb der zugehörigen Box liegen. Außerhalb der Box ist der Effekt exakt null. Es gibt keinen Rechenweg, über den eine LoRA aus ihrer Box "rausbluten" könnte.

## Installation

1. Ordner `RegioCraft/` nach `ComfyUI/custom_nodes/` kopieren
2. ComfyUI neu starten
3. Node hinzufügen: **RegioCraft → "RegioCraft (Regional Multi-LoRA + Ref Lock)"**

## Minimaler Workflow

```
UNETLoader ─┐
CLIPLoader ─┴─► RegioCraft ──► model ──► KSampler
                          └──► conditioning ──► KSampler (positive)
```

Charakter-LoRAs kommen in die Region-Zeilen von RegioCraft selbst, nicht in einen globalen LoRA-Stack davor. Der `conditioning`-Output muss ans KSampler-positive-Input, sobald du `base_prompt` oder Region-Prompts/Trigger nutzt — ein einfacher `CLIPTextEncode` davor reicht dafür nicht.

## Bedienung

1. Box pro Charakter auf die Canvas zeichnen (oder bestehende verschieben)
2. Pro Region-Zeile einstellen: **enable**-Toggle, **LoRA**-Dropdown, **strength**, optional **prompt**-Text, optional **trigger**-Name, optional **Referenzbild** hochladen
3. Wo die Box liegt, wirkt die LoRA — Reihenfolge oder Nummerierung spielt keine Rolle

**Die eine Regel, die zählt:** Die Box markiert, wo die LoRA injiziert wird — nicht bloß "wo der Charakter ungefähr ist". Die meisten Charakter-LoRAs sind gesichtstrainiert, also muss die Box das Gesicht/den Kopf abdecken. Lieber großzügig als zu eng — eine Box, die das Gesicht verfehlt, gibt eine schwache Identität.

**Beliebig viele Regionen, ein Detail dabei:** Über "+ Add Region" kannst du so viele Boxen zeichnen wie du willst, kein Cap. Die einzige Grenze bei 8: externe, live verbindbare `region_prompt_N`-Sockets (z.B. für Wildcard-Generatoren) gibt's nur für Region 1–8. Ab Region 9 tippst du den Prompt einfach direkt ins Prompt-Feld der Region — LoRA, Referenzbild und alles andere funktioniert unabhängig davon bei jeder Region ohne Limit.

## Alle Settings erklärt

| Setting | Was es macht |
|---|---|
| `base_prompt` | Gesamtszene (Komposition, Setting, Licht). Wird mit Region-Text kombiniert und für `conditioning` maskiert. |
| `auto_activate_from_prompt` | Aus per Default. Wenn an: Regionen mit gesetztem `trigger`-Namen schalten sich automatisch ein, sobald ihr Name in `base_prompt` auftaucht. |
| `split_mode` | `manual` (gezeichnete Boxen, Standard) / `bbox` (externe BOUNDING_BOX-Quelle) / `auto_vertical` / `auto_horizontal` (gleiche Streifen ohne Boxen) |
| `seam_feather` | Weichheit der Grenze zwischen Regionen |
| `blend_override` | `0` = sauberer Split (empfohlen); höher = Regionen verschwimmen Richtung gemeinsamem Durchschnitt |
| `sparse_threshold` | Überspringt Tokens nahe Null für Geschwindigkeit. `0` = sicherste/langsamste Einstellung, `0.01` = praktischer Standard |
| `steps_without_applying` / `lora_ramp_calls` | Warmup-Scheduling — lässt die Basis-Komposition sich erst setzen, bevor regionale Identitäten durchgesetzt werden |
| `attention_isolation` | `0` = aus. `4`–`8` probieren bei Identity-Bleed, den die LoRA-Maskierung allein nicht stoppt. Über `10` können Ausdrücke flach werden. |
| `ref_strength` / `ref_start_percent` / `ref_end_percent` / `ref_feather` | Reference-Lock-Steuerung. `0` Strength = aus. Braucht angeschlossene VAE. |
| `identity_provider` | `none` (Standard) oder `krea2edit`. Siehe eigener Abschnitt unten. |
| `identity_ref_boost` | Nur relevant wenn `identity_provider = krea2edit`. Referenz-Treue-Regler: `1.0` = aus, `~4.0` = starke Ähnlichkeit (empfohlener Standard), `>10` = Risiko dass zu stark kopiert wird. |

## Identity Provider (`krea2edit`) — Referenzbild + Prompt-Instruktion

Das ist der Modus, den du in deinem Test-Workflow siehst: Referenzbild rein, per Prompt beschreiben was mit der Person passieren soll ("recolor the jacket to red", "the person on the left waving") — echtes instruktionsbasiertes Identity-Editing, nicht nur reines Identity-Locking.

**Wichtig zu verstehen — das ist ein anderer Mechanismus als Reference Lock:**
- **Reference Lock** (`ref_strength` etc.) — VAE-encodet das Referenzbild einmal, zieht die Vorhersage während des Samplings sanft in diese Richtung. Keine Text-Instruktion beteiligt.
- **`identity_provider = krea2edit`** — übernimmt für Regionen mit gesetztem Referenzbild den **kompletten Diffusion-Forward-Pass** über das separate `comfyui-krea2edit` Node-Pack. Das ist der Mechanismus, der Text-Instruktionen versteht und die Person entsprechend umsetzt.

**Voraussetzungen für `identity_provider = krea2edit`:**
- Das separate Custom Node **`comfyui-krea2edit`** muss installiert sein (`ComfyUI/custom_nodes/comfyui-krea2edit`)
- Die **[`krea2_identity_edit` LoRA](https://civitai.com/models/2761113/krea-2-identity-edit)** muss vorhanden und geladen sein
- Eine **VAE** muss am RegioCraft-Node angeschlossen sein (ohne VAE bleibt der Identity-Pfad deaktiviert, RegioCraft fällt automatisch auf normales Forward zurück)
- Maximal **2 gleichzeitige Referenzbilder** (Scene + Subject) — das ist die trainierte Obergrenze der LoRA selbst, keine RegioCraft-Beschränkung. Mehr aktive Referenz-Regionen werden geloggt und ignoriert, kein Fehler.

Regionen mit LoRA-Masking und Attention Isolation bleiben während des `krea2edit`-Durchlaufs aktiv — RegioCraft bleibt alleiniger Besitzer des Model-Wrappers, es laufen keine zwei konkurrierenden Patches gleichzeitig (das war genau der Bug, den die frühere Version noch hatte).

## Reference Lock — was das zusätzlich bringt

Gibst du einer Region ein Referenzbild, encodet RegioCraft es einmal per VAE, passt es auf die Box im latenten Grid an und zieht die Vorhersage während eines geplanten Fensters sanft in diese Richtung. Verankert Identität über verschiedene Seeds und Generierungen hinweg — was reines LoRA-Delta-Masking allein nicht schafft.

## Outputs

| Output | Was es ist |
|---|---|
| `model` | Gepatchtes Modell — normal an KSampler |
| `clip` | Unverändert durchgereicht |
| `mask_preview` | Regenbogen-codiertes Bild, zeigt wo jede Region-Box aktuell sitzt |
| `info` | JSON-Zusammenfassung was pro Run aktiv war |
| `conditioning` | Kombinierte Basis- + Region-Text-Konditionierung |

## Troubleshooting

- **Charaktere sehen trotzdem vermischt aus** → Boxen überlappen sich vermutlich; verkleinern oder `seam_feather` senken
- **Konsole zeigt "0 layers matched"** → LoRA/LoKr-Key-Format passt nicht zum geladenen Modell (andere Architektur oder umbenannte Layer durch Merge)
- **Ein Charakter stimmt, der andere ist generisch** → Box für diese Region deckt vermutlich nicht das Gesicht ab; Box verschieben, nicht neu zuordnen
- **"Failed to validate prompt" nach Update** → alte Node-Instanz löschen und frisch reinziehen, da ComfyUI Werte positionsbasiert statt namensbasiert speichert
