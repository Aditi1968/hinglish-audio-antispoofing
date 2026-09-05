# Project Handoff — Hindi/Hinglish Audio Deepfake Dataset V1

Paste this to a fresh assistant as context. **Dataset V1 is FROZEN.** Nothing in it
should be collected, re-split, re-augmented, preprocessed, renamed or deleted.
No model/training work has been done yet, and none should start without explicit
instruction.

Project root: `<REPO_ROOT>`
Platform: Windows 11, Git Bash, Python 3.12 (`.venv`), ffmpeg full build on PATH.

---

## 1. What this project is

A Hindi + Hinglish (Hindi-English code-switched) **real vs fake speech dataset** for
audio deepfake detection research/learning. 20,000 development samples, balanced
5,000 per `language_type × label`, plus a separate 2,000-sample external evaluation
set that must never touch training.

---

## 2. Final composition (frozen)

| Language | Label | Source | Clips |
|---|---|---|---|
| Hindi | real | Common Voice Hindi 26.0 | 2,500 |
| Hindi | real | Kathbath (ai4bharat) | 2,500 |
| Hindi | fake | IndicSynth — XTTS-v2 | 2,500 |
| Hindi | fake | IndicSynth — FreeVC24 | 2,500 |
| Hinglish | real | MUCS / OpenSLR-104 | 5,000 |
| Hinglish | fake | Indic Parler-TTS (voice "Rani") | 1,756 |
| Hinglish | fake | Azure/Edge Neural TTS (Madhur, Swara) | 3,244 |
| | | **Development total** | **20,000** |
| Hindi | both | SEA-Spoof (external eval, **excluded**) | 2,000 |

Splits: train 13,963 (69.81%) · validation 3,012 (15.06%) · test 3,025 (15.12%).
Audio: 29.92 hours, median clip 5.22 s.

---

## 3. Folder structure

```
deepfake_dataset_v1_starter/
├── DATASET_V1_CARD.md                    # dataset card (purpose, composition, limits, licenses)
├── HANDOFF_DATASET_V1.md                 # this file
├── README.md                             # original starter readme
├── requirements.txt
├── *.log                                 # run logs for every stage
│
├── data/
│   ├── raw/                              # UNTOUCHED source downloads
│   │   ├── 1781715680033-cv-corpus-26.0-2026-06-12-hi/
│   │   └── mucs/
│   │       ├── Hindi-English_train.tar.gz          # 7.33 GB official OpenSLR-104
│   │       ├── extracted/train/                    # 522 files, 10.35 GB (521 wavs + transcripts/)
│   │       └── probe/                              # test-archive transcripts used for schema discovery
│   │
│   ├── selected/                         # UNTOUCHED per-source originals (byte-preserved)
│   │   ├── common_voice_hi_v26/          #  2,500 files  0.08 GB  MP3 @32/48k
│   │   ├── kathbath_hi_v1/               #  2,500 files  0.44 GB  FLAC/PCM_24 @16k
│   │   ├── indicsynth_hi/                #  5,000 files  1.41 GB  WAV/PCM_16 @24k
│   │   ├── indicsynth_hi_v0_lowspk/      #  5,000 files  1.29 GB  ARCHIVED REJECTED v0 (keep)
│   │   ├── mucs_hinglish/                #  5,000 files  1.00 GB  WAV/PCM_16 @16k
│   │   └── hinglish_fake/                #  5,000 files  0.27 GB  WAV @24k + MP3-in-.wav @24k
│   │
│   ├── processed/v1_16k/                 # 20,000 files  3.45 GB  CANONICAL 16k mono PCM16
│   ├── augmented/v1/                     # 13,963 files  2.40 GB  one derived file per TRAIN parent
│   └── evaluation/
│       ├── seaspoof_hi/                  #  2,000 files  0.38 GB  EXTERNAL EVAL — frozen, never train
│       └── robustness_v1/                #  2,000 files  0.35 GB  transformed TEST parents
│
├── metadata/
│   ├── dataset_v1_master_v2.csv          # ★ CANONICAL 20,000-row manifest (frozen)
│   ├── dataset_v1_augmented.csv          # ★ 13,963 augmentation rows (frozen)
│   ├── dataset_v1_freeze_manifest.json   # ★ seeds, counts, aggregate + artifact hashes
│   ├── DATASET_V1_SHORTCUT_REPORT.md     # ★ before/after confound analysis
│   ├── dataset_v1_master.csv             # pre-split-fix master (superseded, kept)
│   ├── dataset_v1_audit.json             # 20k audit that found the defects
│   ├── dataset_v1_preprocessing_audit.json
│   ├── dataset_v1_shortcut_audit.json    # numeric shortcut analysis
│   ├── common_voice_hi_v26.csv           # per-source manifests (provenance)
│   ├── kathbath_hi_v1.csv
│   ├── indicsynth_hi.csv  + _sha256.csv + _dataset_summary.json
│   ├── indicsynth_hi_shard_index.csv     # parquet row-group index (footer-only scan)
│   ├── indicsynth_hi_v0_lowspk.csv       # archived rejected v0 manifest
│   ├── indicsynth_hi_v2_checkpoint.csv / _processed.csv   # resume state
│   ├── mucs_hinglish.csv + _summary.json
│   ├── hinglish_fake.csv + _summary.json + _sourceA_only.csv
│   └── evaluation/
│       ├── robustness_v1.csv             # ★ 2,000 robustness rows (frozen)
│       ├── seaspoof_hi.csv               # external eval manifest
│       ├── seaspoof_hi_v0_with_commonvoice.csv   # superseded version
│       ├── seaspoof_hi_checkpoint.csv
│       └── seaspoof_eval_index.csv       # 57,226-row metadata index (no audio)
│
└── scripts/
    ├── COLLECTION
    │   ├── prepare_common_voice.py, prepare_kathbath.py, inspect_kathbath.py, test_kathbath.py
    │   ├── test_indicsynth.py, index_indicsynth_shards.py, probe_freevc_speakers.py
    │   ├── prepare_indicsynth.py (v0, superseded), prepare_indicsynth_v0_lowspk.py (archive copy)
    │   ├── prepare_indicsynth_v2.py, finalize_indicsynth.py
    │   ├── download_mucs.py, prepare_mucs.py
    │   ├── prepare_hinglish_fake.py, prepare_hinglish_fake_topup.py
    │   └── test_seaspoof.py, prepare_seaspoof.py
    └── DATASET V1 PIPELINE (run in this order)
        ├── audit_dataset_v1.py           # 1. audit → found split + shortcut defects
        ├── fix_dataset_splits.py         # 2. → dataset_v1_master_v2.csv
        ├── preprocess_dataset_v1.py      # 3. → data/processed/v1_16k/
        ├── augment_dataset_v1.py         # 4. → data/augmented/v1/ + robustness_v1/
        ├── audit_shortcuts_v1.py         # 5. → dataset_v1_shortcut_audit.json
        └── freeze_dataset_v1.py          # 6. read-only verify + freeze manifest
```

Total on disk ≈ 21.4 GB (raw MUCS extraction is the largest single item at 10.35 GB).

---

## 4. What was done, stage by stage

### Collection (each source inspected live before coding — never trusting dataset cards)

1. **Common Voice Hindi 26.0** — 2,500 real, speaker-disjoint splits.
2. **Kathbath** — 2,500 real, 51 speakers.
3. **IndicSynth Hindi** — 5,000 fake. Card claims 3 generators; **live inspection proved
   VITS does not exist in the Hindi config** (footer scan of all 107 parquet shards:
   shards 0–54 freevc24, 54–106 xtts_v2, nothing between). Rebalanced to
   2,500 XTTS-v2 + 2,500 FreeVC24 with the user's approval.
   - A first attempt (v0) yielded only **7 XTTS speakers** and **0 XTTS in test** → rejected,
     archived at `data/selected/indicsynth_hi_v0_lowspk/`.
   - v2 samples one parquet row-group per speaker → **100 XTTS + 95 FreeVC speakers**.
4. **MUCS / OpenSLR-104** — 5,000 real Hinglish. Kaldi layout; utterances had to be
   **cut** from 521 long tutorial recordings using `segments` timestamps
   (lossless slices, verified bit-identical). 520 speakers, recording-disjoint by construction.
5. **Hinglish fake** — README claimed 10,094 Hinglish; **transcript script analysis showed
   only 1,756 are genuinely code-switched** (the rest are pure Hindi or pure English mislabeled).
   Topped up to 5,000 with a second generator (Azure/Edge Neural TTS, 2 voices, 138 domains).
6. **SEA-Spoof Hindi** — 2,000 external eval from the official held-out `evaluation` split.
   Bonafide half was re-collected to **exclude Common Voice-derived audio** (training overlap),
   using only `indic_tts` bonafide after verifying it is genuine human speech.

### Dataset V1 pipeline

7. **Audit** (`audit_dataset_v1.py`) found two serious defects:
   - **Kathbath was 100% train** — empty validation and test.
   - **Codec and sample rate perfectly predicted the label** (majority-vote accuracy 1.0000).
8. **Split fix** (`fix_dataset_splits.py`) — only Kathbath re-split (1,751/375/374 over 32/9/10
   speakers). Added `group_id` composite key and `gender_normalized`. Zero leakage everywhere.
9. **Canonicalization** (`preprocess_dataset_v1.py`) — identical ffmpeg pipeline for all 20,000:
   `decode → mono → 16 kHz → PCM16 WAV`. No denoise/trim/VAD/enhance/normalize. Max duration
   drift 0.0428 s.
10. **Augmentation** (`augment_dataset_v1.py`) — 11 conditions (mp3 32/64/128, opus 12/24/48,
    telephone, lowpass, gain, mild_clipping, multi_codec), one derived file per **training**
    parent, dealt round-robin per `source × label × language_type` (seed 42). Plus a 2,000-file
    robustness set from **test** parents only.
11. **Shortcut audit** (`audit_shortcuts_v1.py`) — quantified every metadata column.
12. **Freeze** (`freeze_dataset_v1.py`) — re-hashed all 35,963 files, 0 mismatches.

---

## 5. Key facts a new assistant must know

**Grouping keys (per source, splits are group-disjoint):**

| Source | Key | Groups |
|---|---|---|
| Common Voice | `speaker_id` | 357 |
| Kathbath | `speaker_id` | 51 |
| IndicSynth | `target_speaker_id` | 101 |
| MUCS | `speaker_id` (⇒ session/recording-disjoint) | 520 |
| Indic Parler | normalized `transcript` | 1,756 |
| Azure Edge TTS | normalized `transcript` | 1,612 |

**Speaker IDs are source-local namespaces.** All 51 Kathbath IDs collide with IndicSynth
`target_speaker_id` strings. Always group on `group_id` (`source_dataset::group`), never on a
bare `speaker_id`.

**Seeds:** 42 (split assignment + augmentation deal), 43 (robustness parent selection).
Kathbath allocation is fully deterministic (clip count desc, tie-break speaker id asc).

**Aggregate fingerprints** (re-run `freeze_dataset_v1.py`, read-only, to verify):
```
canonical    d26400b7dfee110708b832050c97c6e63a96424211884f64ba94e14be56ed4c5
augmented    5b599f8374e3412c9d062358a08e160a72975368e250c547d4f8e0663be05e39
robustness   64fe95503f66687a93416bfd97d13a96a02cff05bbc7a91d9d63b070bcbbf9eb
```

---

## 6. Confounds — read before trusting any number

Measured single-feature majority-vote accuracy (0.50 = chance, 1.00 = feature *is* the label):

| Feature | Before canonicalization | After |
|---|---|---|
| codec | 1.0000 (Hindi) / 0.8244 (Hinglish) | 0.5000 |
| sample rate | 1.0000 (both halves) | 0.5000 |
| bit depth | 1.0000 (Hindi) | 0.5000 |
| duration bin | 0.6389 | unchanged |
| **source_dataset** | **1.0000** | **1.0000 — unchanged** |

- **`source_dataset` still predicts the label perfectly** — each corpus is entirely real or
  entirely fake. This is structural in V1 and cannot be preprocessed away. Source must never be
  a model input; generalization must be checked across held-out sources or on SEA-Spoof.
- **Historical MP3 artifacts survive canonicalization.** Common Voice (real Hindi) and Azure
  (fake Hinglish) were MP3 before collection. Uniform PCM16 output made the *metadata* constant,
  not the acoustics.
- **Resampling history differs by source** — >16 kHz sources were low-passed on the way down,
  natively-16 kHz ones weren't. Residual cue exposed by no metadata column.
- **Duration differs by source** (Parler 1.49 s … IndicSynth 5.72 s medians). Fixed-duration
  windows are still required. Short clips were kept, not deleted.
- Kathbath is 100% female; 11,250/20,000 rows have unknown gender.
- Hinglish fake: only 2 generators / 3 voices. IndicSynth Hindi: only 2 generation families.
- MUCS is tutorial-domain Hinglish; speaker ≈ recording (520 speakers / 521 recordings).
- **Not production-grade, not representative of real-world deepfakes.**

---

## 7. Licenses (recorded as found; none inferred)

| Source | Clips | License | Status |
|---|---:|---|---|
| Common Voice Hindi 26.0 | 2,500 | CC0-1.0 | VERIFIED |
| Kathbath | 2,500 | CC BY 4.0 | VERIFIED |
| IndicSynth | 5,000 | **CC BY-NC 4.0** | VERIFIED |
| MUCS / OpenSLR-104 | 5,000 | CC BY-SA 4.0 | VERIFIED |
| nameissakthi/hindi-english-bilingual | 1,756 | CC BY 4.0 | VERIFIED |
| lingamvamshikrishnareddy/octopus-tts-hinglish | 3,244 | none stated | **UNVERIFIED — do not redistribute** |
| SEA-Spoof | 2,000 | non-commercial academic, gated | VERIFIED |

IndicSynth's CC BY-NC 4.0 makes the whole development pool **non-commercial**.

---

## 8. Rules for the next assistant

**Do not:** collect more data · change splits · regenerate augmentations · preprocess audio ·
create windows · delete archives (including `indicsynth_hi_v0_lowspk`) · rename canonical files ·
modify SEA-Spoof · touch `data/raw/` or `data/selected/`.

**Immutable V1 artifacts:**
`metadata/dataset_v1_master_v2.csv`, `metadata/dataset_v1_augmented.csv`,
`metadata/evaluation/robustness_v1.csv`, `metadata/dataset_v1_freeze_manifest.json`,
`DATASET_V1_CARD.md`, `metadata/DATASET_V1_SHORTCUT_REPORT.md`,
`data/processed/v1_16k/`, `data/augmented/v1/`, `data/evaluation/robustness_v1/`.

**Environment notes:** the machine runs near its memory ceiling (~1.5 GB free of 15.5 GB).
Spawning >2–3 concurrent ffmpeg processes causes `WinError 1455` (paging file too small) and
`MemoryError` during hashing. All heavy scripts use 2 workers, 256 KB chunks, and are resumable.

**Next stage (not started):** model work — fixed-duration windowing, XLS-R/WavLM, training,
evaluation. The user intends to implement dataset logic themselves for learning, so ask before
writing code on their behalf.
