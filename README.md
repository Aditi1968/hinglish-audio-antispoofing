# Hindi & Hinglish Audio Deepfake Detection — Dataset V1

A reproducible recipe for assembling a 20,000-clip Hindi and Hinglish (Hindi–English
code-switched) real-vs-synthetic speech dataset, plus a **full audit of everything wrong with
it**.

**No audio is distributed here.** This repository contains code, documentation and redacted
metadata. You assemble the dataset yourself from seven upstream sources, under their licences.

| | |
|---|---|
| **Status** | Dataset V1 frozen 2026-08-11. No model training has been done. |
| **Scale** | 20,000 development clips · 29.92 h · plus 2,000 external-eval clips |
| **Languages** | Hindi, Hinglish (Hindi–English code-switched) |
| **Code licence** | MIT ([`LICENSE`](LICENSE)) |
| **Data licence** | not included; upstream terms, **non-commercial in aggregate** ([`DATA_LICENSES.md`](DATA_LICENSES.md)) |
| **Audit** | [`DATASET_REPORT.md`](DATASET_REPORT.md) — read this before using any number |

---

## What this is

Most audio deepfake detection work is English-first. Hindi and Hinglish are spoken by hundreds
of millions of people, are a live target for voice-cloning fraud, and are barely represented in
public spoof-detection benchmarks. This project builds a first Hindi/Hinglish real-vs-fake
speech corpus and asks a narrower question than "can we detect deepfakes":

> **Research question.** When a detector scores well on an assembled Hindi/Hinglish
> real-vs-synthetic corpus, what is it actually keying on — the synthesis artifacts, or the
> collection artifacts?

That framing is deliberate. The honest finding of this project so far is **not** a detection
score; it is that a corpus assembled the obvious way — real speech from human ASR corpora,
fake speech from TTS repositories — comes pre-loaded with shortcuts strong enough that a
three-line heuristic beats most of the work a model would do. Within the Hindi half,
`np.max(np.abs(x)) > 0.9` separates real from fake at **97.3%** accuracy.

So this repo is published as a **negative result with a reproducible recipe**: here is how to
build the dataset, and here is a measured account of the six ways it will mislead you.

**This dataset is not production-grade and is not representative of real-world deepfake
attacks.** It is a research and learning artifact.

---

## Composition

Balanced at 5,000 per `language_type × label`. Every source is 100% one label — a structural
confound, see [Confound 1](#1-source-identity-is-the-label).

| Language | Label | Source | Generator | Clips |
|---|---|---|---|---:|
| Hindi | real | Common Voice Hindi 26.0 | — | 2,500 |
| Hindi | real | Kathbath (IndicSUPERB) | — | 2,500 |
| Hindi | fake | IndicSynth | XTTS-v2 | 2,500 |
| Hindi | fake | IndicSynth | FreeVC24 (voice conversion) | 2,500 |
| Hinglish | real | MUCS / OpenSLR-104 | — | 5,000 |
| Hinglish | fake | nameissakthi bilingual | Indic Parler-TTS ("Rani") | 1,756 |
| Hinglish | fake | octopus-tts-hinglish | Azure/Edge Neural TTS (Madhur, Swara) | 3,244 |
| | | | **Development total** | **20,000** |
| Hindi | both | SEA-Spoof (`evaluation` split) | 14 systems + bonafide | 2,000 |

### Splits

Target 70/15/15, assigned **per source**, whole groups only, seed 42.

| Split | real | fake | total | share |
|---|---:|---:|---:|---:|
| train | 6,962 | 7,001 | 13,963 | 69.81% |
| validation | 1,513 | 1,499 | 3,012 | 15.06% |
| test | 1,525 | 1,500 | 3,025 | 15.12% |

| | test | train | validation |
|---|---:|---:|---:|
| Hindi real | 781 | 3,459 | 760 |
| Hindi fake | 751 | 3,500 | 749 |
| Hinglish real | 744 | 3,503 | 753 |
| Hinglish fake | 749 | 3,501 | 750 |

Grouping keys: Common Voice / Kathbath / MUCS → `speaker_id`; IndicSynth →
`target_speaker_id`; the two Hinglish-fake sources → normalised `transcript`. Group
disjointness across splits was independently re-verified and holds exactly
(`train ∩ validation = train ∩ test = validation ∩ test = 0`) — **but see
[Confound 3](#3-real-speakers-and-their-clones-share-a-namespace) and
[Confound 4](#4-transcript-disjointness-is-vacuous-for-one-source), which are leaks
*underneath* that key.**

### Derived sets

| Set | Clips | Built from |
|---|---:|---|
| `data/augmented/v1/` | 13,963 | one derived clip per **train** parent; 11 codec/channel conditions dealt round-robin |
| `data/evaluation/robustness_v1/` | 2,000 | 400 **test** parents × 5 conditions |

Canonical audio is **16 kHz mono WAV/PCM_16**, verified on all 20,000 files, produced by one
identical pipeline: `decode → mono → resample 16 kHz → PCM16`. No denoising, no silence
trimming, no VAD, no normalisation.

---

## Known confounds

These are the six findings from [`DATASET_REPORT.md`](DATASET_REPORT.md) that would materially
distort a result. They are summarised honestly here because the point of publishing this repo
is the audit, not a leaderboard number. Severity is the report's.

### 1. Source identity **is** the label
Every corpus contributes only real or only fake audio, so `source_dataset` predicts the label
with **accuracy 1.0000**. Structural in V1; cannot be preprocessed away. Source must never
reach the model, and generalisation must be measured across held-out sources.

### 2. 🔴 Loudness and silence separate the classes at ~97%
The dominant problem, and one the project's original metadata-only shortcut audit missed
entirely because it never opened the audio. Measured across all 20,000 canonical files
(single-feature best-threshold accuracy; 0.50 = chance):

| Feature | Pooled | **Hindi** | Hinglish |
|---|---:|---:|---:|
| peak amplitude | 0.702 | **0.973** | 0.684 |
| clipped-sample fraction | 0.735 | **0.972** | 0.924 |
| leading silence | 0.724 | **0.969** | 0.742 |
| RMS level (dBFS) | 0.761 | 0.862 | 0.746 |
| energy ≥ 7.6 kHz | 0.574 | 0.610 | 0.729 |

IndicSynth output is effectively peak-normalised (median peak 0.9987) where human corpora sit
at 0.61–0.71; TTS is tightly trimmed (0.00 s leading silence) where crowdsourced human
recording carries slack (Common Voice 0.44 s). Canonicalisation made the *container* uniform,
not the acoustics.

**Mitigate at load time**: per-utterance peak or LUFS normalisation, identical silence handling
for both classes, and publish a peak-amplitude-only baseline beside any model number.

### 3. 🔴 Real speakers and their clones share a namespace
Kathbath's 51 human speakers are the **same AI4Bharat speakers IndicSynth cloned** — identical
`<15-digit>-<speaker>-<m|f>` ID scheme across 2,500/2,500 and 5,000/5,000 rows, all 51 Kathbath
IDs present in IndicSynth's 101 targets, all 51 female where IndicSynth is 53f/48m
(p ≈ 5×10⁻¹⁵), and 169 shared utterance-ID prefixes. **30 of 51 speakers straddle splits** once
the two sources are pooled. The earlier documentation treated this as a coincidental namespace
collision; it is not. Speaker identity is not actually held out in the Hindi half.

### 4. ⚠ "Transcript-disjointness" is vacuous for one source
Indic Parler's 1,756 clips produced 1,756 distinct `group_id`s — one group per clip, so that
split is effectively random rather than transcript-disjoint. Under stricter normalisation, 114
transcript pairs straddle splits. (The Azure source's grouping is genuine: 3,244 clips → 1,612
groups, 1 collision.)

### 5. 🔴 The external eval set has a 100%-precision codec shortcut
In SEA-Spoof, **PCM_16 ⇒ spoof: 642 fake, 0 real.** All 1,000 bonafide files are PCM_24.
Bit depth alone scores **82.1%** on a set built as a fair 50/50 test, because each upstream TTS
system wrote a fixed depth. Duration adds a second cue (AUC 0.705). **Canonicalise SEA-Spoof to
a fixed bit depth before evaluating on it**, exactly as the development set was.

### 6. ⚠ Dead audio and no-op augmentations
Twelve clips carry confident labels but contain no usable speech — four are pure digital
silence up to 10.7 s, and one of those sits in the **test** split. The silence originates
upstream; the pipeline converted it faithfully, but nothing ever checked that the audio
contained audio. Separately, **178 "augmented" files are byte-identical to their parents**
(13.9% of the `mild_clipping` condition, skewed 161:17 toward the real class) because the
±6 dB round trip never saturated on quiet input.

### Also worth knowing
Duration differs by source (Indic Parler median 1.49 s vs MUCS 6.27 s mean) — use
fixed-duration windows. Kathbath is 100% female. Hinglish-fake has only 2 generators and
3 voices. MUCS is tutorial-domain Hinglish with speaker ≈ recording (520 speakers /
521 recordings). Historical MP3 artifacts survive canonicalisation for Common Voice and Azure
audio. Resampling history differs: sources above 16 kHz were low-passed on the way down,
natively-16 kHz ones were not.

---

## Metadata: public vs local

The full manifests are **not published**. They carry verbatim transcripts belonging to the
upstream corpora, pseudonymous speaker identifiers for real people, the AI4Bharat utterance IDs
that link Kathbath humans to their IndicSynth clones (Confound 3), and gender.

| | Published | Contents |
|---|---|---|
| [`metadata/public/`](metadata/public/) | ✅ 4 CSVs, 37,963 rows | `sample_id`, `label`, `language_type`, `source_dataset`, `generator`, `split`, `duration`, `codec`, `sha256`, hashed `group_id`, relative `path` |
| `metadata/` (everything else) | ❌ gitignored | transcripts, speaker/session/utterance IDs, gender, licence provenance, audit JSON |

`group_id` is published as a **salted SHA-256 hash**, because for two sources the raw group key
*is the verbatim transcript*. The salt lives in `metadata/public/.group_salt` and is gitignored.
Hashing is collision-free here (4,397 raw groups → 4,397 hashes) and preserves exactly what you
need: split disjointness still verifies.

Regenerate with [`scripts/make_public_manifests.py`](scripts/make_public_manifests.py); it
refuses to write a file if a forbidden column or an un-hashed group key survives.

See [`metadata/public/README.md`](metadata/public/README.md) for the column reference.

---

## How to reproduce

Requires Python 3.12, ffmpeg on `PATH`, and ~27 GB free disk.

```bash
git clone <this-repo> && cd deepfake_dataset_v1_starter
python -m venv .venv && source .venv/Scripts/activate   # or bin/activate on Unix
pip install -r requirements.txt
```

### 1. Download the sources

One script per source in [`scripts/download/`](scripts/download/). Run from the repo root.
Read [`DATA_LICENSES.md`](DATA_LICENSES.md) first — three of the seven are gated.

```bash
python scripts/download/download_common_voice.py    # prints manual steps (browser required)
python scripts/download/download_kathbath.py        # gated: accept terms, then `hf auth login`
python scripts/download/download_indicsynth.py      # public
python scripts/download/download_mucs.py            # public, ranged parallel fetch (~6.8 GB)
python scripts/download/download_hinglish_parler.py # public
python scripts/download/download_octopus_tts.py     # gated + NO LICENCE; refuses without a flag
python scripts/download/download_seaspoof.py        # never downloads; prints request procedure
```

### 2. Select and build per-source manifests

```bash
python scripts/prepare_common_voice.py --input data/raw/<cv-dir>/hi \
    --output data/selected/common_voice_hi_v26 \
    --metadata metadata/common_voice_hi_v26.csv --n 2500
python scripts/prepare_kathbath.py
python scripts/index_indicsynth_shards.py && python scripts/prepare_indicsynth_v2.py
python scripts/finalize_indicsynth.py
python scripts/prepare_mucs.py
python scripts/prepare_hinglish_fake.py && python scripts/prepare_hinglish_fake_topup.py
python scripts/prepare_seaspoof.py                  # only after SEA-Spoof approval
```

### 3. Audit, split, canonicalise, augment, freeze

Run in this order — each step consumes the previous step's output:

```bash
python scripts/audit_dataset_v1.py        # 1. finds split + shortcut defects
python scripts/fix_dataset_splits.py      # 2. -> metadata/dataset_v1_master_v2.csv
python scripts/preprocess_dataset_v1.py   # 3. -> data/processed/v1_16k/  (16 kHz mono PCM16)
python scripts/augment_dataset_v1.py      # 4. -> data/augmented/v1/ + robustness_v1/
python scripts/audit_shortcuts_v1.py      # 5. -> metadata/dataset_v1_shortcut_audit.json
python scripts/freeze_dataset_v1.py       # 6. read-only verify + freeze manifest
python scripts/make_public_manifests.py   # 7. -> metadata/public/
```

Seeds: 42 (split assignment, augmentation deal), 43 (robustness parent selection).

### 4. Verify your rebuild

`metadata/public/*.csv` carries a `sha256` per file. Every canonical file should match; if
yours do not, your ffmpeg version resamples differently and your numbers will not be comparable.

> **Memory note.** The original build machine ran near its ceiling. Every heavy script uses
> 2 workers and 256 KB chunks and is resumable. More than 2–3 concurrent ffmpeg processes
> caused `WinError 1455` and `MemoryError` during hashing.

---

## Ethics and scope

**This project is for detection only.**

- **Nothing that generates speech is released here.** No TTS or voice-conversion models, no
  fine-tuned weights, no checkpoints, no synthesised audio. The repository contains collection,
  preprocessing and audit code, and metadata. `.gitignore` blocks weight and audio formats.
- **No audio is redistributed.** Every clip must be obtained from its upstream source under
  that source's terms. Two sources explicitly forbid redistribution
  ([`DATA_LICENSES.md`](DATA_LICENSES.md)).
- **Research and non-commercial use only.** IndicSynth's CC BY-NC 4.0 governs the aggregate,
  so anything trained on the assembled pool is non-commercial.
- **All human speech comes from public research corpora** whose contributors recorded for
  research use: Common Voice (CC0 donations), Kathbath/IndicSUPERB, MUCS/OpenSLR-104. All
  synthetic speech is TTS output. No speech was scraped from private individuals, and no clip
  targets a specific person.
- **Gated sources stay gated.** Kathbath, octopus-tts-hinglish and SEA-Spoof require
  upstream approval. Do not work around the gate and do not mirror gated audio. If you cannot
  obtain approval, report on the subset you can legitimately obtain.
- **One consent question this project does not resolve.** Confound 3 establishes that the
  Kathbath speakers are the people IndicSynth cloned. Those speakers consented to appear in a
  research ASR corpus; whether that extends to voice cloning was decided upstream by AI4Bharat
  and the IndicSynth authors, not here. Anyone building on this should be aware they are
  handling paired human/cloned voice data for identifiable speaker IDs — which is exactly why
  those IDs are stripped from `metadata/public/`.
- **Dual use.** Deepfake *detection* research inevitably documents what makes synthesis
  detectable. The confound analysis here describes collection artifacts, not techniques for
  making synthesis more convincing.

**Do not** describe this dataset as production-grade, as representative of real-world deepfake
attacks, or as evidence that detection is solved for Hindi/Hinglish.

---

## Repository layout

```
├── README.md                    this file
├── DATASET_REPORT.md            ★ independent audit — read before trusting a number
├── DATASET_V1_CARD.md           dataset card (composition, splits, limitations, licences)
├── HANDOFF_DATASET_V1.md        build history, stage by stage
├── DATA_LICENSES.md             ★ per-source licence, citation, redistribution status
├── LICENSE                      MIT (code and docs only)
├── metadata/public/             ★ the 4 publishable manifests (see its README)
├── scripts/
│   ├── download/                one script per upstream source
│   ├── prepare_*.py             per-source selection
│   ├── audit_*.py               defect and shortcut audits
│   ├── fix_dataset_splits.py    group-disjoint re-split
│   ├── preprocess_dataset_v1.py canonicalisation to 16 kHz mono PCM16
│   ├── augment_dataset_v1.py    augmentation + robustness set
│   ├── freeze_dataset_v1.py     read-only verification + freeze manifest
│   └── make_public_manifests.py metadata/ -> metadata/public/ redaction
└── data/                        gitignored — you build this locally
```

## Citing

Cite the upstream corpora — the citations are in [`DATA_LICENSES.md`](DATA_LICENSES.md). This
repository is a recipe and an audit, not a corpus.

## Contributing

The most useful contributions are **more confounds**. If you find a shortcut this audit missed,
open an issue with the measurement. `DATASET_REPORT.md` documents its method so results can be
compared.
