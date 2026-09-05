# Dataset V1 — Hindi & Hinglish Audio Deepfake Detection

**Version:** v1 · **Status:** FROZEN · **Freeze date:** 2026-08-11 (UTC)
**Canonical development samples:** 20,000 · **External evaluation:** 2,000 (separate)

Reproducibility manifest: [`metadata/dataset_v1_freeze_manifest.json`](metadata/dataset_v1_freeze_manifest.json)
Shortcut report: [`metadata/DATASET_V1_SHORTCUT_REPORT.md`](metadata/DATASET_V1_SHORTCUT_REPORT.md)

---

## A. Purpose

A Hindi and Hinglish (Hindi–English code-switched) real/fake speech dataset assembled for **research and learning** on audio deepfake detection.

It is explicitly a **first baseline**. It is not production-grade and not representative of real-world deepfake attacks. See Section F before drawing conclusions from any result obtained on it.

---

## B. Composition

| Language | Label | Source | Clips |
|---|---|---|---|
| Hindi | real | Common Voice Hindi 26.0 | 2,500 |
| Hindi | real | Kathbath | 2,500 |
| Hindi | fake | IndicSynth — XTTS-v2 | 2,500 |
| Hindi | fake | IndicSynth — FreeVC24 | 2,500 |
| Hinglish | real | MUCS / OpenSLR-104 | 5,000 |
| Hinglish | fake | Indic Parler-TTS (voice "Rani") | 1,756 |
| Hinglish | fake | Azure/Edge Neural TTS (Madhur, Swara) | 3,244 |
| | | **Total** | **20,000** |

Balanced at 5,000 per `language_type × label` cell.

**External evaluation (NOT part of development/training data):**

| Dataset | Clips | Location |
|---|---|---|
| SEA-Spoof Hindi (official `evaluation` split) | 2,000 | `data/evaluation/seaspoof_hi/` |

SEA-Spoof is frozen separately, excluded from every development manifest, and must never enter training.

Total audio: **29.92 hours**, median clip 5.22 s (min 0.21 s, max 30.0 s).

---

## C. Split strategy

Target 70/15/15, split **independently per source**, assigning **whole groups** so no group is ever divided across splits.

| Source | Grouping key | Unique groups |
|---|---|---|
| Common Voice | `speaker_id` | 357 |
| Kathbath | `speaker_id` | 51 |
| IndicSynth | `target_speaker_id` | 101 |
| MUCS | `speaker_id` (also session/recording-disjoint) | 520 |
| Indic Parler | normalized `transcript` | 1,756 |
| Azure Edge TTS | normalized `transcript` | 1,612 |

Speaker IDs are **source-local namespaces** — all 51 Kathbath IDs collide with IndicSynth `target_speaker_id` strings. Grouping therefore uses a composite key `source_dataset + "::" + group_id`. Never group on a bare `speaker_id`.

**Resulting counts**

| Source | train | validation | test |
|---|---:|---:|---:|
| Common Voice | 1,708 | 385 | 407 |
| Kathbath | 1,751 | 375 | 374 |
| IndicSynth | 3,500 | 749 | 751 |
| MUCS | 3,503 | 753 | 744 |
| Azure Edge TTS | 2,272 | 486 | 486 |
| Indic Parler | 1,229 | 264 | 263 |
| **Total** | **13,963** (69.81%) | **3,012** (15.06%) | **3,025** (15.12%) |

| | train | validation | test |
|---|---:|---:|---:|
| Hindi real | 3,459 | 760 | 781 |
| Hindi fake | 3,500 | 749 | 751 |
| Hinglish real | 3,503 | 753 | 744 |
| Hinglish fake | 3,501 | 750 | 749 |

**Zero group leakage**, verified per source and on the pooled composite key: `train∩validation = train∩test = validation∩test = 0` in every case.

For Hinglish fake there are only three synthetic voices, so a speaker-disjoint split is impossible; splits are **transcript-disjoint** instead. This is stated plainly rather than dressed up as speaker disjointness.

---

## D. Audio processing

Original and selected audio under `data/raw/` and `data/selected/` is **preserved unmodified**. Canonical audio is written to a separate tree.

Every one of the 20,000 files went through the **same** pipeline:

```
original encoded file → decode → mono → resample 16 kHz → PCM16 WAV
```

Canonical output (`data/processed/v1_16k/`): **16,000 Hz · mono · WAV/PCM_16**, verified on all 20,000 files.

Explicitly **not** applied: denoising, dereverberation, silence trimming, VAD cutting, speech enhancement, per-source normalization, intentional duration change. Full utterances are kept, including clips under 1 second.

Duration drift from decode/resample: max 0.0428 s, 0 files above the 0.05 s tolerance.

---

## E. Augmentation

Derived audio lives in separate trees and never overwrites canonical files.

**Policy:** one derived file per **training** parent (13,963 files). Validation and test are **not** augmented for training. Every derived sample **inherits its parent's split verbatim**, and carries `parent_sample_id`, `source_dataset`, `language_type`, `label`, `split` and `group_id`.

Conditions are dealt round-robin inside each `source_dataset × label × language_type` stratum with **seed 42**, so condition cannot correlate with label, language, source, generator or voice.

| Condition | Parameters | Implementation |
|---|---|---|
| `canonical_clean` | — | the canonical 16 kHz file itself; no derived copy |
| `mp3_32` / `mp3_64` / `mp3_128` | 32 / 64 / 128 kbps | real libmp3lame encode → decode → 16 kHz PCM16 |
| `opus_12` / `opus_24` / `opus_48` | 12 / 24 / 48 kbps | real libopus encode → decode → 16 kHz PCM16 |
| `telephone` | 300–3400 Hz, 8 kHz intermediate | high/low-pass → true 8 kHz stage → back to 16 kHz |
| `lowpass` | cutoff 4000 / 5000 / 6000 / 7000 Hz | ffmpeg `lowpass` |
| `gain` | −9 / −6 / −3 / +3 dB | ffmpeg `volume` |
| `mild_clipping` | +6 dB into PCM16 saturation, then −6 dB | ~2.1% samples clipped; intelligibility preserved |
| `multi_codec` | `mp3_64→opus_24` or `opus_24→mp3_64` | two real encode/decode round trips |

All derived files are 16 kHz mono PCM16. Duration delta vs parent: max 0.0027 s.

**Robustness evaluation set** (`data/evaluation/robustness_v1/`): 400 **test** parents (100 per `language_type × label`) × 5 conditions (mp3_64, opus_24, telephone, lowpass 4 kHz, multi_codec) = **2,000 files**. Same parents across all conditions so conditions are directly comparable. No parent is shared with the training augmentation view.

**Not implemented:** environmental noise and room impulse responses exist as optional hooks only (`NOISE_DIR`, `RIR_DIR`); nothing was downloaded. **Physical replay** (play through a speaker, re-record with a microphone) is deliberately absent — it cannot be synthesized honestly in software.

---

## F. Known limitations

Read this section before reporting any number from this dataset.

- **`source_dataset` is perfectly correlated with the label.** Each corpus contributes only real or only fake audio, so source identity alone predicts the label with 100% accuracy. This is structural in V1. Source must never be a model input, and generalization should be measured across held-out sources.
- **Historical codec/channel artifacts may survive canonical conversion.** Common Voice and Azure audio was MP3 before collection; that quantization noise persists through decoding and resampling.
- **Resampling history differs by source.** Sources above 16 kHz (Common Voice 32/48 kHz, IndicSynth and Hinglish fake 24 kHz) were low-passed on the way down; natively-16 kHz sources (Kathbath, MUCS) were not. The resulting high-band difference is a residual source-correlated cue.
- **Duration differs by source** (Indic Parler ~1.5 s vs MUCS ~5.1 s vs Azure ~5.4 s). Fixed-duration windows are required so utterance length cannot act as a cue. Short clips were retained, not deleted.
- **Kathbath is 100% female** (all 2,500 clips), which skews gender against label in the Hindi half.
- **Gender metadata is missing** for MUCS and both Hinglish fake sources; vocabulary differed across sources and was normalized into `gender_normalized` (female / male / unknown) without altering the original `gender` column.
- **Hinglish fake has only 2 generators and 3 voices** (Indic Parler "Rani"; Azure Madhur and Swara).
- **IndicSynth provides only 2 fake-generation families in Hindi** (XTTS-v2 text-to-speech and FreeVC24 voice conversion). VITS does not exist in the Hindi config despite the dataset card describing three generators.
- **MUCS is technical/tutorial-style Hinglish** (Linux, LibreOffice, Bash spoken tutorials), not broad conversational Hinglish; its code-switching is driven by technical vocabulary.
- **MUCS speaker and recording/session are nearly confounded** — 520 speakers across 521 recordings, so each speaker is essentially one recording with one channel and one topic.
- **Octopus/Azure source license remains UNVERIFIED** (3,244 clips); `redistribution_allowed = false` on all derived rows.
- **IndicSynth is CC BY-NC 4.0**, which makes the whole derived development set non-commercial.
- **This dataset must NOT be described as production-grade or fully representative of real-world deepfakes.**

> **Canonicalization removes explicit codec and sample-rate differences from the file representation, but it DOES NOT prove that underlying codec/channel signatures have disappeared.** Uniform WAV/PCM16 output means those metadata columns were made constant — not that the acoustic evidence of the original encoding is gone.

---

## G. Licenses

Recorded as found upstream. **No license was inferred.**

| Source | Clips | License | Status |
|---|---:|---|---|
| Common Voice Hindi 26.0 | 2,500 | CC0-1.0 | VERIFIED |
| Kathbath | 2,500 | CC BY 4.0 | VERIFIED |
| IndicSynth | 5,000 | CC BY-NC 4.0 | VERIFIED |
| MUCS / OpenSLR-104 | 5,000 | CC BY-SA 4.0 | VERIFIED |
| nameissakthi/hindi-english-bilingual | 1,756 | CC BY 4.0 | VERIFIED |
| lingamvamshikrishnareddy/octopus-tts-hinglish | 3,244 | none stated upstream | **UNVERIFIED** |
| SEA-Spoof (external eval) | 2,000 | non-commercial academic; gated, author approval | VERIFIED |

The Octopus/Azure repository has no README, no LICENSE file, no license tag and no card metadata; all commits are bulk uploads with no stated terms. Those files are marked `license_status = UNVERIFIED` and `redistribution_allowed = false` — **do not redistribute them**. Separately, that audio was produced with Microsoft Azure/Edge neural voices, whose upstream output terms are an independent question that has not been resolved here.

The most restrictive terms in the development pool are IndicSynth's **CC BY-NC 4.0** (non-commercial) and MUCS's **CC BY-SA 4.0** (share-alike).

---

## Artifacts

| Path | Contents |
|---|---|
| `metadata/dataset_v1_master_v2.csv` | 20,000 canonical rows, splits, group IDs, per-file SHA256 |
| `metadata/dataset_v1_augmented.csv` | 13,963 training augmentation rows |
| `metadata/evaluation/robustness_v1.csv` | 2,000 robustness rows (test parents only) |
| `metadata/dataset_v1_freeze_manifest.json` | reproducibility manifest, seeds, aggregate hashes |
| `metadata/dataset_v1_audit.json` | pre-fix 20k audit |
| `metadata/dataset_v1_preprocessing_audit.json` | canonicalization audit |
| `metadata/dataset_v1_shortcut_audit.json` | shortcut/confound analysis |
| `data/processed/v1_16k/` | 20,000 canonical WAVs (3.45 GB) |
| `data/augmented/v1/` | 13,963 augmented WAVs (2.40 GB) |
| `data/evaluation/robustness_v1/` | 2,000 robustness WAVs (0.35 GB) |

Scripts: `fix_dataset_splits.py`, `preprocess_dataset_v1.py`, `augment_dataset_v1.py`, `audit_shortcuts_v1.py`, `audit_dataset_v1.py`, `freeze_dataset_v1.py`.

Seeds: 42 (split assignment, augmentation deal), 43 (robustness parent selection). Kathbath allocation is fully deterministic and consumes no RNG.
