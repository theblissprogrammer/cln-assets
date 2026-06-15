# The Delivery Fingerprint — research findings (night of 2026-06-15→16)

**Goal:** make the cloned voice speak Arabic carrying her *way of talking* (Layer 2), not just her
timbre (Layer 1, already solved: Chatterbox ≈0.92). Formalize "way of talking" as an explicit,
auditable, individual operator — not a black-box style embedding.

## What I proved (honest, with numbers)

1. **The delivery fingerprint is real and measurable.** Formalized "way of talking" as timbre-free
   channels: F0 range/dynamism, contour shape, energy/emphasis, rhythm/pausing, voice-quality.
   (`delivery_fingerprint.py`)

2. **Delivery alone identifies her — it's a genuine identity channel separate from timbre.**
   Using ZERO timbre/spectral info, a delivery vector separates her from 19 female impostors at
   **AUC 0.93 / EER ~15%** under a proper leave-one-speaker-out protocol. This matches the
   literature (Gengembre IS2024: prosody-only re-ID at 13.5% EER). So "the way she talks" carries
   her identity — but it's a *soft* cue (~4–10× weaker than timbre): it makes speech sound like her,
   it doesn't alone prove her. The win = Chatterbox timbre (hard anchor) **+** her delivery (her-ness).

3. **The best engine loses her delivery cross-lingually.** Chatterbox-Arabic sits at delivery-sim
   0.12–0.17 vs her 0.46 (her/not-her threshold 0.22) — the clone reads as *not-quite-her* on delivery
   even though its timbre is 0.92. So there's a real gap to close.

4. **Which delivery channels are actually *her* (this reorganized everything):**
   | channel | discriminative AUC |
   |---|---|
   | rhythm/timing | 0.92 *(partly genre-confounded)* |
   | F0 range/dynamism | 0.85 *(genre-robust)* |
   | energy dynamics | 0.73 |
   | **contour SHAPE** | **0.48 = chance** |
   Her identity lives in **range/dynamism + energy + rhythm — NOT contour shape.** Her melody shape
   is *not* individual, and is *not* a learnable function of content (R²≈0, linear & nonlinear).
   Her energy/emphasis *is* an individual learnable function (R²=0.36).

5. **Post-hoc waveform editing is the wrong architecture.** Built faithful PSOLA contour operators
   (imposition r=0.997, clean −0.13 UTMOS tax). Best post-hoc delivery gain was small (+0.04) and
   cost naturalness+identity — and Chatterbox's own `exaggeration` knob beat every post-hoc operator.
   Re-confirms the project rule: **inject delivery at generation, never post-hoc DSP.** (My contour
   transplant chased *shape* — the one channel that isn't even individual.)

## The honest synthesis

The transferable, genuinely-*her* delivery is **narrower than the dream hoped**, because most channels
are confounded once you check honestly:
- **contour melody/shape** — not individual (AUC 0.48) and not learnable (R²≈0) → can't/needn't clone.
- **rhythm/pausing** — strongest raw signal (AUC 0.92) but mostly a *genre* artifact (clean single TTS
  sentence vs her flowing conversation); recoverable generically by multi-phrase generation + pauses,
  but that's genre-matching, not her-identity (her pause profile beats a random one by only +0.03).
- **energy dynamics** — *channel*-confounded: her YouTube audio is dynamic-range-compressed, so
  "matching her energy" just matches her recording compression, not her delivery.
- **F0 range/dynamism** — the one clean, individual, genre-AND-channel-robust lever (AUC 0.85; pitch is
  immune to amplitude compression and additive noise).

**⇒ The honest operator = generate fluent, flowing Arabic with Chatterbox + her-expressive reference,
with `exaggeration` calibrated to match HER measured F0 range/dynamism** — explicit, auditable,
per-speaker, calibrated to her real statistics (not a black-box embedding). Modest but real and honest.
This is exactly what the generation-time sweep (running) quantifies.

## Genuine contribution (vs prior art)
- The **delivery-fingerprint-as-an-identity-channel** measurement (AUC 0.93 timbre-free) + the
  **per-channel individuality ablation** (shape is NOT individual; range/energy/rhythm are) is a
  novel, honest characterization current cross-lingual TTS evaluation lacks.
- The **calibrated, explicit, auditable per-speaker delivery operator** (vs the leaky black-box
  prosody embeddings that Sigurgeirsson-King ICASSP2023 showed don't transfer).

## Open / next
- Generation-time sweep result → best calibrated setting (ear clips for Ahmed).
- Energy-dynamics calibration (target her rms_std exactly, not blind expansion).
- Her real breaths/laughs: blocked — her YouTube audio is denoised, no clean events to extract.
