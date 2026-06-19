# Post-BWE Foley Audio Mastering — Deep Research (2026-06-19)

Training-free / pretrained post-processing methods to add **on top of** UniverSR BWE, to
improve perceived quality of the generated NSFW foley (moaning / wet / skin-slap). Deep-research
harness: 6 angles, 26 sources, 127 claims → 25 adversarially verified (21 confirmed, 4 killed).
Tags per method: **[targets]** which of our problems (fizz / wash / noise-floor / darkness /
weak-transients) and **[risk]** to breath or moan tonality.

## TL;DR — ranked prototype shortlist

| # | Method | Library | Targets | Risk | Verdict |
|---|---|---|---|---|---|
| 1 | **LUFS loudness normalize** | `pyloudnorm` (BS.1770-4) | loudness consistency | low | safe to automate; do per-clip into a true-peak limiter |
| 2 | **Harmonic exciter** (HPF→soft-clip→attenuate→parallel-mix) | `pedalboard.Distortion` | **darkness, wash** | mod (over-drive harshens moans) | cheap brightness *without* BWE synthesis fizz — strongest new lever |
| 3 | **Transient shaper** (fast−slow envelope) | scipy (2 one-pole followers) | **weak transients** (slaps) | low | level-independent punch, doesn't pump moans |
| 4 | **pedalboard DSP backbone** (Comp/Limiter/EQ/Reverb/Distortion + VST3 host) | `pedalboard` (Spotify) | all of dynamics/EQ/spatial | low–mod | the engine everything else rides on |
| 5 | **SAGA-SR neural BWE** (alt/complement to UniverSR) | arXiv 2509.24924 (Sep 2025) | **darkness** | **high** (diffusion hallucinates breath at low SNR) | gate by ear; resample 44.1→48k |
| 6 | **HPSS tonal/transient split** | `librosa.decompose.hpss` | de-fizz (weak), routing | mod | reliable for tonal-vs-transient; **NOT** clean fizz isolation |

**The honest gap:** our #1 problem — **de-fizz** — is the *least* well-answered. HPSS does **not**
cleanly isolate fizz (refuted 0-3 below), and the real tool (soothe-style dynamic resonance
suppression) needs a commercial VST or an unproven Python port. The confident wins are
**exciter (darkness), transient shaper (slaps), LUFS (loudness), and SAGA-SR (BWE upgrade)**.

## By category

### (a) De-fizz / HF grain — WEAKEST coverage, treat as experimental
- **HPSS** (`librosa.decompose.hpss`) reliably separates tonal moans (H) from slap transients (P),
  reconstruction exact (~4e-14). BUT the idea that `margin>1` returns a residual R isolating
  noise-like **fizz** was **REFUTED 0-3** — fizz is broadband and splits across P and R, so you
  can't cleanly route it out. **[targets fizz — unreliable] [risk: aggressive H/P masking dulls breath]**
- **Dynamic resonance suppression ("soothe"-style) / de-essing / dynamic-EQ**: the *correct* tools
  for grain/harshness, but **not individually verified** here and the good ones (soothe2) are
  licensed VSTs. `pedalboard` can host a VST3 de-harsher via `load_plugin`. Open question — no
  validated open-source Python equivalent surfaced.
- Net: no slam-dunk training-free de-fizz. Best bets to *prototype*: (i) a narrow dynamic-EQ
  band on the fizz region, (ii) host a VST de-harsher through pedalboard, (iii) gentle HF-only
  HPSS-H emphasis. All need ear A/B; none proven.

### (b) Dynamics & punch — STRONG
- **Transient shaper** = gain steered by (fast envelope − slow envelope), **level-independent**
  (no threshold), so it boosts slap attack without pumping the moan body. Two one-pole followers
  in scipy, or `pedalboard`. **[targets weak-transients] [risk: low; can exaggerate clicks on grainy HF]**
- **LUFS** via `pyloudnorm` (ITU-R BS.1770-4) — perceptually better than peak-norm; a ComfyUI
  wrapper already exists. **[targets loudness] [risk: none]** Pair with a true-peak limiter
  (note: pedalboard's `Limiter` is threshold/release, **not** true-peak — use a real TP limiter
  or oversample).
- Multiband / parallel compression available in pedalboard. **[targets wash] [risk: over-compression flattens breath]**

### (c) Tonal / corrective EQ — partially covered
- Tilt EQ, dynamic EQ, automatic resonance taming all sensible via pedalboard filters / hosted
  VST, but **none individually verified** as standalone wins. Gentle **tilt EQ** is safe to
  automate; dynamic EQ for dark-vs-harsh needs per-clip tuning.

### (d) Harmonic exciter — STRONG, the key darkness lever
- Classic recipe (AES Eng. Brief / Aphex Aural Exciter / Knoppel 1979): **Butterworth HPF →
  asymmetric soft-clip (odd+even harmonics above a threshold) → attenuate 20–70% → parallel mix**.
  **Synthesizes harmonics from existing content** (unlike BWE which extrapolates a new band) →
  brightness+presence with a *different*, often cleaner character than BWE fizz.
  `pedalboard.Distortion` (tanh) is a ready stage. **[targets darkness, wash] [risk: over-drive
  harshens moans — keep drive low, mix parallel]**. This is the cheapest thing to try against
  the dark-tail problem and may *complement or partially replace* BWE on dark clips.

### (e) Neural restoration — SAGA-SR > AudioSR, both gated by hallucination
- **SAGA-SR** (arXiv 2509.24924, Sep 2025): versatile generative SR (speech/music/SFX), upsamples
  4–32k → 44.1k, **text-conditioned** HF (cuts AudioSR's sibilance). Best fit for our 11–15k dark
  tails. **[targets darkness] [risk HIGH: diffusion hallucinates breath at low SNR; 44.1k→48k resample]**
- **AudioSR** (ICASSP 2024): upsamples 2–16k → 24k @ 48k (matches our rate), **has a ComfyUI node
  already** (`Saganaki22/ComfyUI-AudioSR`). Higher fizz/sibilance risk than SAGA-SR. Fallback only.
- **Structural caveat (confirmed):** generative BWE hallucination is *inherent* — missing HF is
  ill-posed (one input → many plausible HF), and generative models sample the conditional
  distribution rather than regress to one target. So **any** neural BWE (AudioSR, SAGA-SR,
  VoiceFixer, partly UniverSR) can fabricate breath/gasping in quiet gaps → **gate by ear**, this
  reinforces our existing breath-inseparability finding.
- **Speech tools still ruled out (confirmed):** ArtiFree/DeepFilterNet/UNIVERSE++ use speech
  embeddings + WER (undefined for non-speech) → thin non-speech breath. (Apollo / FlashSR /
  VoiceFixer did **not** survive verification — no non-speech evidence.)

### (f) Stereo / spatialization — UNCOVERED
- mono→pseudostereo and subtle convolution room reverb (pedalboard `Reverb`/`Convolution`) are
  plausible for "sitting in a space," but **no claim survived** — unverified. Low priority,
  prototype-and-ear only. **[risk: reverb smears transients + adds wash]**

### (g) Declip / dehum-harmonics / codec cleanup — UNCOVERED
- No tool survived verification beyond our existing notch. If clipping exists, declip *before*
  everything; otherwise skip. Treat as not-validated.

### (h) Chain order (INFERRED — not source-verified as one chain)
```
declip → dehum (our IIR notch) → gated denoise → de-fizz (HPSS/dyn-EQ, experimental)
  → corrective/tilt EQ → BWE (UniverSR or SAGA-SR) → exciter → multiband dynamics + transient shaper
  → LUFS normalize → true-peak limiter
```
- **Safe to automate:** dehum, LUFS, limiter, gentle tilt EQ.
- **Needs per-clip tuning / ear-gate:** all neural BWE, denoise strength, exciter drive, de-essing.
- Open: exciter before vs after BWE; de-fizz before vs after BWE — both unverified, A/B needed.

### (i) Validation — our render-battery is the right instinct
- **PESQ / STOI / ESTOI are invalid** for non-speech BWE (confirmed). LSD beats SNR but is
  full-reference (we have no clean foley reference).
- **Use our non-reference descriptors** (already in the render-battery): HF spectral flatness
  (fizz), HNR on tonal segments (moan clarity), gap-band 300–3000 Hz breath energy, crest factor
  (transients/punch), LUFS. Add **MOS/MUSHRA by ear** as primary.
- For *batch* regression vs a curated real-foley set: **KAD** (kernel/MMD, distribution-free)
  beats **FAD** (FAD assumes Gaussian embeddings, violated for foley). `kadtk` toolkit. (The
  "-0.93 vs -0.80 correlation" sub-claim was refuted; use KAD for the distribution-free property,
  not that specific number.)

## Refuted (do NOT rely on)
- HPSS `margin>1` isolates fizz into residual R → **0-3 false**. Fizz won't cleanly route out.
- ArtiFree as evidence that low-SNR maximizes hallucination → 1-2.
- KAD Spearman −0.93 vs FAD −0.80 → 1-2 (KAD still preferred for distribution-free property).
- AudioSR/FlashSR validated on ESC-50/FreeSound/ShipsEar → 1-2 (no confirmed non-speech eval).

## Open questions / next empirical steps
1. **De-fizz is unsolved** — find/port an open-source dynamic resonance suppressor, or host a VST
   de-harsher via pedalboard, and A/B on fizzy clips. Highest-value gap.
2. **SAGA-SR vs AudioSR vs UniverSR by ear** on moaning/wet/slap — and at what input bandwidth/SNR
   each starts fabricating breath. Per-clip A/B.
3. **Exciter before vs after BWE**, and optimal chain order — A/B.
4. **Calibrate a composite non-reference score** (flatness+HNR+gap-breath+crest+LUFS) to match our
   by-ear ranking; consider KAD vs a curated real-foley reference for batch regression.

## Recommended first prototypes (highest value / lowest risk)
1. **`FoleyTuneExciter`** node — HPF→tanh soft-clip→attenuate→parallel-mix (pedalboard). Directly
   attacks the dark-tail problem with a cleaner character than BWE fizz. Cheap, big upside.
2. **`FoleyTuneTransientShaper`** — fast−slow envelope (scipy). Punches the slaps; low risk.
3. **LUFS + true-peak limiter** tail on the existing chain (`pyloudnorm`). Safe, automatable.
4. Then tackle **de-fizz** (the hard one) and **SAGA-SR** A/B as research spikes, both ear-gated.
