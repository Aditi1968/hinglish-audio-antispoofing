# Dataset Reconnaissance Report

**Target:** `<REPO_ROOT>`
**Date:** 2026-09-05 · **Mode:** read-only (nothing modified, moved, renamed or deleted)
**Tooling:** `soundfile` 0.14.0, `ffprobe` (ffmpeg 2025-01-30 full build), pandas 3.0.5, Python 3.12.5 (`.venv`)

This is an independent audit. It does not take `DATASET_V1_CARD.md` or `HANDOFF_DATASET_V1.md`
at their word — every claim below was re-measured from the files on disk. Where my measurements
contradict the existing documentation, that is called out explicitly.

**Headline:** the bookkeeping is genuinely excellent — 20,000/20,000 label coverage in both
directions, zero missing files, zero duplicate content in the canonical set, and the documented
group-disjoint splits hold exactly. The problems are all *acoustic* and *in the evaluation set*,
which is precisely where the existing shortcut audit did not look. There are six findings that
would materially distort a model result, listed in §7.

---

## 1. Directory tree (3 levels)

```
deepfake_dataset_v1_starter/
├── data/
│   ├── raw/                                          # untouched upstream downloads
│   │   ├── 1781715680033-cv-corpus-26.0-2026-06-12-hi/
│   │   │   └── cv-corpus-26.0-2026-06-12/hi/         # 10 .tsv + 1 .md, clips/ = 19,029 mp3
│   │   └── mucs/
│   │       ├── Hindi-English_train.tar.gz            # 6.83 GB source archive
│   │       ├── extracted/train/                      # 521 long .wav + transcripts/
│   │       └── probe/test/transcripts/               # Kaldi schema probe, 6 files
│   ├── selected/                                     # SPLIT BY SOURCE/GENERATOR
│   │   ├── common_voice_hi_v26/                      # Hindi real
│   │   ├── kathbath_hi_v1/                           # Hindi real
│   │   ├── indicsynth_hi/                            # Hindi fake  (xtts_v2 + freevc24)
│   │   ├── indicsynth_hi_v0_lowspk/                  # ARCHIVED REJECT — see §6.3
│   │   ├── mucs_hinglish/                            # Hinglish real
│   │   └── hinglish_fake/                            # Hinglish fake (Parler + Azure)
│   ├── processed/v1_16k/                             # CANONICAL 20,000 · flat, no class dirs
│   ├── augmented/v1/                                 # 13,963 derived (train parents only)
│   └── evaluation/
│       ├── seaspoof_hi/                              # external eval, 1,000 real + 1,000 fake
│       └── robustness_v1/                            # 2,000 transformed test parents
├── metadata/                                         # 14 csv + 7 json + 1 md
│   └── evaluation/                                   # 5 csv
├── scripts/                                          # 23 .py
└── *.log, *.md, requirements.txt                     # 15 run logs, 3 docs
```

**How the split is organised — important:** the tree is split **by source corpus / generator only**.
There is **no `real/` vs `fake/` directory anywhere**, and no language directories. Every label,
split and language assignment lives exclusively in the `metadata/*.csv` manifests. `data/processed/v1_16k/`
is one flat folder of 20,000 WAVs mixing both classes, both languages and all three splits.

Consequence: **you cannot build a loader from the directory layout.** Any `ImageFolder`-style
"folder name = class" convention will silently produce garbage. The manifest is the only
source of truth.

The one place structure *does* encode class is `data/selected/`, where each subfolder is
100% one label. That is the perfect source↔label correlation the card already admits to (§7.1).

---

## 2. File counts and size per leaf folder

| Leaf folder | Files | Size | Contents |
|---|---:|---:|---|
| `data/processed/v1_16k` | 20,000 | 3.21 GB | `.wav` ×20000 |
| `data/augmented/v1` | 13,963 | 2.24 GB | `.wav` ×13963 |
| `data/evaluation/robustness_v1` | 2,000 | 331.95 MB | `.wav` ×2000 |
| `data/evaluation/seaspoof_hi` | 2,000 | 364.77 MB | `.flac` ×2000 |
| `data/selected/common_voice_hi_v26` | 2,500 | 77.18 MB | `.mp3` ×2500 |
| `data/selected/kathbath_hi_v1` | 2,500 | 419.57 MB | `.wav` ×2500 **(actually FLAC — §3.2)** |
| `data/selected/indicsynth_hi` | 5,000 | 1.31 GB | `.wav` ×5000 |
| `data/selected/indicsynth_hi_v0_lowspk` | 5,000 | 1.20 GB | `.wav` ×5000 (archived reject) |
| `data/selected/mucs_hinglish` | 5,000 | 957.38 MB | `.wav` ×5000 |
| `data/selected/hinglish_fake` | 5,000 | 257.17 MB | `.wav` ×5000 **(3,244 are MP3 — §3.2)** |
| `data/raw/…/hi/clips` | 19,029 | 554.97 MB | `.mp3` ×19029 |
| `data/raw/…/hi` | 11 | 19.58 MB | `.tsv` ×10, `.md` ×1 |
| `data/raw/mucs` | 1 | 6.83 GB | `Hindi-English_train.tar.gz` |
| `data/raw/mucs/extracted/train` | 522 | 9.64 GB | `.wav` ×521 + 1 extensionless |
| `data/raw/mucs/extracted/train/transcripts` | 6 | 14.04 MB | Kaldi files + `.scp` |
| `data/raw/mucs/probe/test/transcripts` | 6 | 861.91 KB | Kaldi files + `.scp` |
| `metadata` | 22 | 39.29 MB | 14 `.csv`, 7 `.json`, 1 `.md` |
| `metadata/evaluation` | 5 | 21.13 MB | 5 `.csv` |
| `scripts` | 23 | 242.57 KB | 23 `.py` |
| *(repo root)* | 19 | 131.40 KB | 15 `.log`, 3 `.md`, 1 `.txt` |

**Grand total: 27.41 GB across 62,963 audio files + ~100 support files** (excluding `.venv`).

### Empty or suspiciously small — findings

- **No empty leaf folders.** Every zero-file directory (`data`, `data/selected`, `data/raw`, …)
  is a pure container with subdirectories. Nothing is a stub.
- **No zero-byte files anywhere.** Smallest audio file is 1,125 B (a raw Common Voice mp3);
  smallest canonical WAV is 6,906 B (0.21 s).
- The two `transcripts/` folders (6 files each) and `metadata/evaluation` (5 files) are flagged
  as "small" only by file count — they are correctly populated Kaldi/manifest folders, not stubs.
- `data/raw/mucs/extracted/train` contains **521 `.wav` + 1 extensionless file** (a Kaldi
  `segments`/`wav.scp`-style file). Expected for the Kaldi layout, not a defect.
- **`data/raw/mucs/` holds 16.47 GB — the 6.83 GB tar.gz *and* its 9.64 GB extraction.** Both
  copies are retained. That is 60% of the entire repo footprint duplicated for no ongoing benefit;
  the 5,000 cut utterances in `data/selected/mucs_hinglish/` are what the dataset actually uses.
  Not a correctness problem, but it is the obvious 6.8 GB to reclaim if disk matters.

---

## 3. Audio inventory

### 3.1 Canonical set — `data/processed/v1_16k` (scanned **all 20,000**, not sampled)

Header scan of every file, plus a full decode of all 20,000 for level/silence statistics.
Sampling was unnecessary here: header reads took 150 s and full decode 174 s.

| Property | Value |
|---|---|
| Format / subtype / rate / channels | **`WAV` / `PCM_16` / `16000 Hz` / `1 ch` — all 20,000, zero exceptions** |
| Files failing to load | **0** |
| Zero-length files | **0** |
| Total duration | **29.922 hours** |

Duration by class:

| Class | n | min | median | mean | max | hours |
|---|---:|---:|---:|---:|---:|---:|
| fake | 10,000 | 0.213 s | 5.304 s | 5.044 s | 28.363 s | 14.011 |
| real | 10,000 | 1.000 s | 5.004 s | 5.728 s | 30.000 s | 15.911 |

By source:

| Source | n | min | median | mean | max | hours |
|---|---:|---:|---:|---:|---:|---:|
| CommonVoice_Hindi_26.0 | 2,500 | 1.260 | 5.040 | 5.232 | 10.548 | 3.633 |
| Kathbath | 2,500 | 2.113 | 4.969 | 5.134 | 13.166 | 3.565 |
| IndicSynth | 5,000 | 2.273 | 5.720 | 5.862 | 28.363 | 8.142 |
| MUCS_OpenSLR104 | 5,000 | 1.000 | 5.080 | 6.273 | 30.000 | 8.712 |
| octopus_tts_hinglish (Azure) | 3,244 | 3.360 | 5.376 | 5.465 | 9.264 | 4.925 |
| nameissakthi (Indic Parler) | 1,756 | 0.213 | **1.493** | 1.936 | 12.755 | 0.945 |

58 files are under 1.0 s, 4 under 0.5 s — all from Indic Parler. Retained deliberately per the card.

**The canonicalisation claim checks out.** 16 kHz / mono / PCM_16 uniformity is real and holds on
every single file. The card's totals (29.92 h, median 5.22 s) reproduce.

### 3.2 Source trees — container vs extension mismatches ⚠

300 files sampled per folder, probed with **both** `soundfile` and `ffprobe` (they agreed 100%).

| Folder | Extension | **Actual container/codec** | Rate | Verdict |
|---|---|---|---|---|
| `common_voice_hi_v26` | `.mp3` | MP3 (265×32 kHz, 35×48 kHz) | 32/48 k | honest |
| `kathbath_hi_v1` | `.wav` | **FLAC / PCM_24** (300/300) | 16 k | **extension lies** |
| `indicsynth_hi` | `.wav` | WAV / PCM_16 | 24 k | honest |
| `indicsynth_hi_v0_lowspk` | `.wav` | WAV / PCM_16 | 24 k | honest |
| `mucs_hinglish` | `.wav` | WAV / PCM_16 | 16 k | honest |
| `hinglish_fake` | `.wav` | **MP3 ×3,244 + WAV/PCM_16 ×1,756** | 24 k | **extension lies** |
| `seaspoof_hi` | `.flac` | FLAC, **mixed PCM_24 / PCM_16** | 16 k | see §7.5 |
| `robustness_v1` | `.wav` | WAV / PCM_16 | 16 k | honest |
| `augmented/v1` | `.wav` | WAV / PCM_16 | 16 k | honest |

I verified the `hinglish_fake` split across **all 5,000** files, not just the sample:

- **MP3-in-`.wav`: exactly the 3,244 Azure/Edge files. Real WAV: exactly the 1,756 Indic Parler files.**
  Container identity is a *perfect* generator identifier in `data/selected/`.
- `metadata/dataset_v1_master_v2.csv` **records this correctly** (`MP3/MPEG_LAYER_III` vs `WAV/PCM_16`).
  Good — the master manifest is not fooled by the extension.
- Likewise Kathbath: `master_v2.csv` correctly says `FLAC/PCM_24` for all 2,500 rows.

**But the per-source manifest is wrong:** `metadata/kathbath_hi_v1.csv` declares `codec = wav`
for all 2,500 rows. The files are FLAC/PCM_24. Anyone trusting that per-source CSV rather than
the master will mis-describe the data.

**Practical risk:** these mislabelled extensions are harmless to `soundfile`/`ffmpeg` (both sniff
content), but they will break any code that dispatches on file extension, and they make
`data/selected/` misleading to inspect by hand.

### 3.3 Files that fail to load, are zero-length, or are silent

- **Load failures: 0** across all 20,000 canonical, all 2,000 SEA-Spoof, and every sampled
  source-tree file. No corruption found anywhere.
- **Zero-length: 0.**
- **Silent / near-silent: 12 files, found by sweeping all 20,000 canonical files.** ⚠

| sample_id | label | source | split | dur | peak | RMS dBFS | verdict |
|---|---|---|---|---:|---:|---:|---|
| `hf_hi_en_00598` | fake | Indic Parler | train | 10.73 s | 0.0 | −240 | **pure digital zeros** |
| `hf_hi_en_00997` | fake | Indic Parler | train | 8.44 s | 0.0 | −240 | **pure digital zeros** |
| `hf_hi_en_01650` | fake | Indic Parler | train | 9.60 s | 0.0 | −240 | **pure digital zeros** |
| `hf_hi_en_01103` | fake | Indic Parler | **test** | 8.72 s | 0.0 | −240 | **pure digital zeros** |
| `hf_hi_en_00135` | fake | Indic Parler | train | 0.26 s | 0.0091 | −70.1 | near-silent |
| `hf_hi_en_00393` | fake | Indic Parler | train | 0.30 s | 0.0193 | −64.9 | near-silent |
| `hf_hi_en_01441` | fake | Indic Parler | train | 0.21 s | 0.0099 | −68.2 | near-silent |
| `mucs_hi_en_00226` | real | MUCS | train | 3.00 s | 3.1e-05 | −96.4 | near-silent |
| `mucs_hi_en_00665` | real | MUCS | train | 7.00 s | 3.1e-05 | −96.3 | near-silent |
| `mucs_hi_en_01652` | real | MUCS | train | 1.00 s | 0.0030 | −73.2 | near-silent |
| `mucs_hi_en_03121` | real | MUCS | train | 2.00 s | 0.0030 | −70.6 | near-silent |
| `mucs_hi_en_04688` | real | MUCS | train | 2.00 s | 0.0027 | −80.4 | near-silent |

**Four files are literally nothing but zeros** — up to 10.7 seconds of digital silence — and each
carries a confident `fake` label. `hf_hi_en_01103` sits in the **test** split, so it is an
unscoreable test item: no classifier can be right or wrong about it for any principled reason.

I traced all 12 back to `data/selected/`: **the silence originates upstream, not in the
preprocessing.** The pipeline faithfully converted files that were already dead. That exonerates
`preprocess_dataset_v1.py` and indicts the collection step — nothing checked that the audio
contained audio. The card's claim that "every one of the 20,000 files went through the same
pipeline" is true; it just never asserted the files were non-empty, and nobody checked.

---

## 4. Class balance

Verified from `metadata/dataset_v1_master_v2.csv` (the canonical manifest).

**Overall: 10,000 real / 10,000 fake — exactly balanced.**

| language_type | fake | real |
|---|---:|---:|
| Hindi | 5,000 | 5,000 |
| Hinglish | 5,000 | 5,000 |

Per source (note: **every source is 100% one label** — see §7.1):

| source_dataset | fake | real | total |
|---|---:|---:|---:|
| CommonVoice_Hindi_26.0 | 0 | 2,500 | 2,500 |
| Kathbath | 0 | 2,500 | 2,500 |
| IndicSynth | 5,000 | 0 | 5,000 |
| MUCS_OpenSLR104 | 0 | 5,000 | 5,000 |
| octopus_tts_hinglish (Azure) | 3,244 | 0 | 3,244 |
| nameissakthi (Indic Parler) | 1,756 | 0 | 1,756 |

Per split:

| split | fake | real | total |
|---|---:|---:|---:|
| train | 7,001 | 6,962 | 13,963 |
| validation | 1,499 | 1,513 | 3,012 |
| test | 1,500 | 1,525 | 3,025 |

Per split × language × label — the 4-way cell balance the card claims, confirmed:

| | test | train | validation |
|---|---:|---:|---:|
| Hindi fake | 751 | 3,500 | 749 |
| Hindi real | 781 | 3,459 | 760 |
| Hinglish fake | 749 | 3,501 | 750 |
| Hinglish real | 744 | 3,503 | 753 |

Derived sets: `augmented/v1` 7,001 fake / 6,962 real (all `train`, inherits parents exactly);
`robustness_v1` 1,000/1,000 (all `test`); `seaspoof_hi` 1,000/1,000 (external eval).

**Balance is exact and honest. No complaint here.** Note the class terminology in this dataset is
`real`/`fake`, not `bonafide`/`spoof` — the ASVspoof vocabulary appears only inside SEA-Spoof's
`original_label` column, where `bonafide`→`real` and `spoof`→`fake`.

---

## 5. Metadata / label files

19 CSVs total. Headers and sample rows below (transcripts truncated for width).

### `metadata/dataset_v1_master_v2.csv` — ★ canonical, 20,000 rows, 39 columns

```
sample_id,file_path,language,language_type,label,source_dataset,generator,voice,speaker_id,
target_speaker_id,source_speaker_id,session_id,source_file_id,gender,split,duration,
original_sample_rate,codec,channels,file_bytes,sha256,transcript,license,license_status,
declared_duration,declared_sample_rate,declared_codec,exists,probe_error,origin_metadata_file,
group_id,gender_normalized,processed_file_path,processed_sample_rate,processed_channels,
processed_codec,processed_duration,processed_bytes,processed_sha256
```

| sample_id | label | source_dataset | split | dur | codec | group_id |
|---|---|---|---|---|---|---|
| `cv_hi_00000` | real | CommonVoice_Hindi_26.0 | train | 8.1435 | MP3/MPEG_LAYER_III | `CommonVoice…::6599122…` |
| `kb_hi_00000` | real | Kathbath | train | 4.9923 | FLAC/PCM_24 | `Kathbath::329` |
| `is_hi_xtts_v2_00000` | fake | IndicSynth | train | 5.7600 | WAV/PCM_16 | `IndicSynth::1023` |
| `mucs_hi_en_00000` | real | MUCS_OpenSLR104 | train | 7.0000 | wav/PCM_16 | `MUCS_OpenSLR104::778006` |
| `hf_hi_en_00000` | fake | nameissakthi… | train | 1.1520 | WAV/PCM_16 | `nameissakthi…::‹full transcript›` |

### `metadata/dataset_v1_augmented.csv` — 13,963 rows, 26 columns

```
augmentation_id,parent_sample_id,parent_processed_path,augmented_file_path,language,language_type,
label,source_dataset,split,parent_split,group_id,generator,voice,gender_normalized,
augmentation_condition,augmentation_parameters,sample_rate,channels,codec,duration,parent_duration,
bytes,sha256,license,license_status,redistribution_allowed
```
Sample: `aug_cv_hi_00000__multi_codec | cv_hi_00000 | real | train | multi_codec | {"chain":"opus_24__mp3_64"} | 16000 | 8.136s`

### `metadata/evaluation/robustness_v1.csv` — 2,000 rows, 26 columns (same schema as augmented)
Sample: `rob_cv_hi_00030__mp3 | cv_hi_00030 | real | test | mp3 | {"bitrate_kbps":64,"codec":"mp3"} | 6.876s`

### `metadata/evaluation/seaspoof_hi.csv` — 2,000 rows, 23 columns

```
sample_id,file_path,language,label,original_label,source_dataset,source_split,project_role,
utterance_id,speaker_id,spoof_type,spoof_category,transcript,duration,original_sample_rate,codec,
fake_start,fake_end,has_timestamp_labels,license_or_access_note,row_id,source_model,upstream_dataset
```
Sample: `ss_hi_fake_00000 | fake | spoof | SEA-Spoof | evaluation | offline | 2.688s | hindi_common_edge_tts`

### Per-source manifests

| File | Rows | Cols | First data row (abridged) |
|---|---:|---:|---|
| `common_voice_hi_v26.csv` | 2,500 | 22 | `cv_hi_00000, real, CommonVoice_Hindi_26.0, validated, male_masculine, mp3, CC0-1.0, train` |
| `kathbath_hi_v1.csv` | 2,500 | 21 | `kb_hi_00000, real, Kathbath, spk 329, Female, 4.9923s, codec=wav ⚠, train` |
| `indicsynth_hi.csv` | 5,000 | 22 | `is_hi_xtts_v2_00000, fake, IndicSynth, xtts_v2, tgt_spk 1023, Male, CC BY-NC 4.0, train` |
| `indicsynth_hi_v0_lowspk.csv` | 5,000 | 20 | `is_hi_xtts_v2_00000, fake, …v0_lowspk\…, tgt_spk 888, Female` (archived reject) |
| `mucs_hinglish.csv` | 5,000 | 19 | `mucs_hi_en_00000, real, MUCS_OpenSLR104, spk 778006, 433.0→440.0s, 7.0s, train` |
| `hinglish_fake.csv` | 5,000 | 19 | `hf_hi_en_00000, fake, nameissakthi…, Indic_Parler_TTS, Rani, 1.152s, train` |
| `hinglish_fake_sourceA_only.csv` | 1,756 | 18 | subset of the above (Parler only) |
| `indicsynth_hi_sha256.csv` | 5,000 | 4 | `is_hi_xtts_v2_00000, 325196 bytes, sha256 36ebe4d0…` |
| `indicsynth_hi_shard_index.csv` | 2,140 | 12 | parquet footer index: `Hindi/train-00000-of-00107.parquet, rg 0, freevc24, spk 1102–1177` |
| `indicsynth_hi_v2_checkpoint.csv` | 5,000 | 22 | resume state (paths point at a non-existent `indicsynth_hi_v2` dir — see below) |
| `indicsynth_hi_v2_processed.csv` | 195 | 2 | `shard, row_group` resume ledger |
| `dataset_v1_master.csv` | 20,000 | 30 | superseded pre-split-fix master |
| `evaluation/seaspoof_eval_index.csv` | 57,226 | 16 | upstream metadata index, **no audio** |
| `evaluation/seaspoof_hi_checkpoint.csv` | 2,000 | 23 | resume state |
| `evaluation/seaspoof_hi_v0_with_commonvoice.csv` | 2,000 | 23 | superseded (the version that had CV overlap) |

JSON: `dataset_v1_freeze_manifest.json`, `dataset_v1_audit.json`,
`dataset_v1_preprocessing_audit.json`, `dataset_v1_shortcut_audit.json`,
`indicsynth_hi_dataset_summary.json`, `hinglish_fake_summary.json`, `mucs_hinglish_summary.json`.

### Does every audio file have a label? Does every label point to a real file?

**Yes, in both directions, with zero exceptions.** This is the strongest part of the dataset.

| Manifest | Rows | Refs resolved | Unique | **Missing on disk** |
|---|---:|---:|---:|---:|
| `dataset_v1_master_v2.csv` → `file_path` | 20,000 | 20,000 | 20,000 | **0** |
| `dataset_v1_master_v2.csv` → `processed_file_path` | 20,000 | 20,000 | 20,000 | **0** |
| `dataset_v1_master.csv` | 20,000 | 20,000 | 20,000 | **0** |
| `dataset_v1_augmented.csv` | 13,963 | 13,963 | 13,963 | **0** |
| `robustness_v1.csv` | 2,000 | 2,000 | 2,000 | **0** |
| `seaspoof_hi.csv` | 2,000 | 2,000 | 2,000 | **0** |
| 6 per-source manifests | 22,756 | 22,756 | 22,756 | **0** |

Reverse direction — every file on disk claimed by a manifest row:

| Folder | On disk | Referenced | **Unlabelled** |
|---|---:|---:|---:|
| all 6 `data/selected/*` | 25,000 | 25,000 | **0** |
| `data/processed/v1_16k` | 20,000 | 20,000 | **0** |
| `data/augmented/v1` | 13,963 | 13,963 | **0** |
| `data/evaluation/robustness_v1` | 2,000 | 2,000 | **0** |
| `data/evaluation/seaspoof_hi` | 2,000 | 2,000 | **0** |

No orphans, no dangling references, no duplicate path references. Every path resolved on the
first try. Whoever built this was disciplined about manifests.

**One stale artifact:** `indicsynth_hi_v2_checkpoint.csv` references
`data\selected\indicsynth_hi_v2\…`, a directory that does not exist (the run was finalised into
`indicsynth_hi/`). It is resume state, not a label file, and nothing consumes it — but it will
break anything that naively globs `metadata/*.csv` and resolves `file_path`.

---

## 6. Duplicates

### 6.1 Exact duplicates by content hash

Method: bucketed all 62,963 audio files by exact byte size, then SHA-256'd every file sharing a
size with another (57,937 files in 2,061 buckets). This is exact, not approximate.

**246 duplicate groups, 492 files, 246 redundant copies.**

| Where | Groups | Assessment |
|---|---:|---|
| `augmented/v1` ↔ `processed/v1_16k` | **178** | ⚠ **defect — see below** |
| `selected/indicsynth_hi` ↔ `selected/indicsynth_hi_v0_lowspk` | 68 | benign overlap between the live set and its archived predecessor |
| **within `processed/v1_16k`** | **0** | ✅ all 20,000 canonical files are byte-unique |
| within any other single folder | 0 | ✅ |

Independently cross-checked against the manifest hash columns: `master_v2.sha256` has 20,000/20,000
unique values and `master_v2.processed_sha256` has 20,000/20,000 unique. My from-disk hashing
agrees exactly. **No duplicate content in the training data itself.**

**The 178-file defect — augmentations that did nothing:**

| Condition | No-ops | Total | **No-op rate** |
|---|---:|---:|---:|
| `mild_clipping` | 176 | 1,267 | **13.89%** |
| `telephone` | 2 | 1,268 | 0.16% |
| all 9 others | 0 | — | 0% |

176 files labelled `augmentation_condition = mild_clipping` are **byte-identical to their
unaugmented parent**. The card describes `mild_clipping` as "+6 dB into PCM16 saturation, then
−6 dB … ~2.1% samples clipped". For these 176 the +6/−6 dB round trip was mathematically lossless
because the source was quiet enough never to reach saturation — so nothing clipped and the file
came back unchanged.

This is not random: it hits **quiet** files, and quietness is source-correlated (161 of 176 are
`real` — 132 MUCS, 29 Common Voice). So the augmented set contains 178 rows that claim a
transformation that did not occur, skewed toward the real class. Anyone measuring "robustness
under mild clipping" from these rows is measuring clean audio and would over-report robustness,
disproportionately for real clips.

### 6.2 Near-duplicates by filename pattern

Naming is systematic and clean — digit-runs collapsed to `#` reveal exactly one template per source:

| Folder | Templates |
|---|---|
| `common_voice_hi_v26` | `cv_hi_#` ×2500 |
| `kathbath_hi_v1` | `kb_hi_#` ×2500 |
| `indicsynth_hi` | `is_hi_freevc#_#` ×2500, `is_hi_xtts_v#_#` ×2500 |
| `mucs_hinglish` | `mucs_hi_en_#` ×5000 |
| `hinglish_fake` | `hf_hi_en_#` ×1756 (Parler), `hf_hi_en_b_#` ×3244 (Azure) |
| `seaspoof_hi` | `ss_hi_fake_#` ×1000, `ss_hi_real_#` ×1000 |
| `augmented/v1` | `aug_<parent>__<condition>` |
| `robustness_v1` | `rob_<parent>__<condition>` |

Note `hf_hi_en_` vs `hf_hi_en_b_` — **the filename prefix alone identifies the generator**
(and therefore, within Hinglish-fake, the container format). Same for `ss_hi_fake_`/`ss_hi_real_`,
where **the SEA-Spoof filename contains the ground-truth label**. Neither is a leak into a model
unless filenames are fed as features, but both are trivially exploitable by accident (e.g. sorting
a directory listing and assuming order is random, or any debug feature derived from the path).

Same stem appearing in more than one tree (expected — a clip carried through the pipeline):
5,000 `indicsynth`, 5,000 `mucs`, 5,000 `hinglish_fake`, 2,500 `common_voice`, 2,500 `kathbath`,
each also present in `processed/v1_16k`.

### 6.3 ⚠ Sample-ID collision between the live and archived IndicSynth sets

`data/selected/indicsynth_hi/` and `data/selected/indicsynth_hi_v0_lowspk/` contain
**5,000 identical filenames** (`is_hi_xtts_v2_00000.wav` exists in both) — but of 400 sampled
name-pairs, **0 were byte-identical**. The archived v0 reject holds *different audio under the
same sample_id*, and all 5,000 v0 `sample_id`s also appear in `master_v2.csv` pointing at the
*other* file.

This is a live footgun. A recursive glob over `data/selected/**/*.wav`, or any dict keyed by
`sample_id` that ingests both manifests, silently mixes 5,000 rejected v0 clips into the dataset
with plausible-looking IDs and correct-looking `fake` labels. Only the full path distinguishes them.
The archive is worth keeping (the card asks for it), but it should not share an ID namespace with
the live set.

---

## 7. Leakage red flags

### 7.1 Source identity is the label — confirmed, and already documented

Every corpus is 100% one class (§4). `source_dataset` predicts the label with **accuracy 1.0000**.
The card states this plainly and I confirm it. Source must never reach the model, and
generalisation is only meaningful across held-out sources or on SEA-Spoof.

### 7.2 ⚠ Do the classes differ systematically in sample rate, duration, channels, or codec?

**In the canonical set: no — and that part is genuinely fixed.** All 20,000 files are
16 kHz / mono / PCM_16 with zero exceptions, so codec, rate, bit depth and channel count are
constant and carry exactly zero information. The card's claim that these confounds were driven
from 1.0000 to 0.5000 is **true as stated**.

**Duration: a mild residual cue.** Single-threshold accuracy 0.576 pooled (AUC 0.521), rising to
0.662 within Hinglish. The tails differ more than the medians:

| decile | fake | real |
|---|---:|---:|
| min | 0.21 s | 1.00 s |
| 10% | 1.56 s | 3.00 s |
| 50% | 5.30 s | 5.00 s |
| 90% | 7.09 s | 9.00 s |
| max | 28.36 s | 30.00 s |

Driven by Indic Parler (median 1.49 s) at the short end and MUCS (max 30 s, with a hard 1.0 s floor
from segment cutting) at the long end. Fixed-duration windowing is mandatory, as the card says.

**But the classes differ enormously in ways no metadata column records — see 7.3.**

### 7.3 🔴 **The dominant finding: loudness and silence structure separate the classes almost perfectly**

The existing `audit_shortcuts_v1.py` scored *metadata columns* (codec, sample rate, bit depth,
duration bin, source). It never opened the audio. I decoded all 20,000 canonical files and scored
**acoustic** features with the same single-feature majority-vote metric the card uses
(0.50 = chance, 1.00 = the feature *is* the label):

| Feature | Pooled acc | **Hindi acc** | Hinglish acc | fake median | real median |
|---|---:|---:|---:|---:|---:|
| **peak amplitude** | 0.702 | **0.9730** | 0.684 | 0.9689 | 0.6273 |
| **clipped-sample fraction** | 0.735 | **0.9722** | 0.924 | 0 | 0 |
| **leading silence** | 0.724 | **0.9694** | 0.742 | 0.02 s | 0.04 s |
| **RMS level (dBFS)** | 0.761 | 0.8624 | 0.746 | −17.2 dB | −22.5 dB |
| voiced frame fraction | 0.680 | 0.7545 | 0.663 | 0.865 | 0.707 |
| energy ≥7.6 kHz | 0.574 | 0.610 | **0.7294** | 3.4e-05 | 1.9e-05 |
| trailing silence | 0.608 | 0.652 | 0.653 | 0.56 s | 0.32 s |
| duration | 0.576 | 0.616 | 0.662 | 5.30 s | 5.00 s |

**Within the Hindi half, three separate one-line features each hit ~97% accuracy.** A `numpy`
one-liner — `np.max(np.abs(x)) > 0.9` — classifies Hindi real vs fake at **97.3%**. No model needed.

Why, per source:

| Source | label | RMS dBFS | **peak** | lead sil | trail sil | voiced frac |
|---|---|---:|---:|---:|---:|---:|
| IndicSynth | fake | −16.53 | **0.9987** | **0.00 s** | 0.28 s | 0.917 |
| nameissakthi (Parler) | fake | −16.81 | 0.8008 | 0.02 s | 0.02 s | 0.924 |
| octopus (Azure) | fake | −20.78 | 0.6290 | 0.24 s | 0.96 s | 0.681 |
| CommonVoice | real | −21.61 | 0.7106 | **0.44 s** | 0.58 s | 0.636 |
| MUCS | real | −22.66 | 0.6064 | 0.06 s | 0.32 s | 0.667 |
| Kathbath | real | −23.32 | 0.6199 | 0.04 s | 0.20 s | 0.819 |

**IndicSynth output is effectively peak-normalised** (median peak 0.9987 — hard against full scale),
while every human corpus sits around 0.61–0.71. The fake classes are ~6 dB louder overall.
TTS output is also tightly trimmed (IndicSynth 0.00 s leading silence) where crowdsourced human
recording carries slack (Common Voice 0.44 s).

**Assessment: this is the most serious problem in the dataset, and it is currently undocumented.**
The card's confound table reports codec/rate/bit-depth confounds "fixed to 0.5000" and gives the
impression the shortcut problem is under control apart from `source_dataset`. It is not.
Canonicalisation normalised the *container*, and the card is careful to say so ("uniform WAV/PCM16
output means those metadata columns were made constant — not that the acoustic evidence is gone").
That caveat is correct but understated: the surviving acoustic cues are not subtle residue, they
are near-perfect classifiers. Any headline accuracy on this dataset should be assumed to be
measuring loudness and silence trimming until proven otherwise.

*Mitigations worth considering (not applied — dataset is frozen): per-utterance peak or LUFS
normalisation, and either VAD-trimming both classes identically or randomising leading/trailing
padding. Report a peak-amplitude-only baseline alongside any model number so the reader can see
how much of the score is free.*

### 7.4 🔴 Is the same speaker/source utterance present in more than one split?

**By the dataset's own grouping key: no — verified, clean, exactly as claimed.**

| Key | Unique | train∩val | train∩test | val∩test |
|---|---:|---:|---:|---:|
| `group_id` (composite, pooled) | 4,397 | **0** | **0** | **0** |
| per-source `group_id` (all 6) | — | **0** | **0** | **0** |
| `source_file_id` (pooled + per-source) | 10,521 | **0** | **0** | **0** |

Group counts reproduce the card exactly (CV 357, Kathbath 51, IndicSynth 101, MUCS 520,
Parler 1,756, Azure 1,612). The split machinery works.

**But three real leaks sit underneath it.**

**(a) 🔴 Kathbath's real speakers are the same people IndicSynth cloned — the card says otherwise.**

`DATASET_V1_CARD.md` asserts: *"Speaker IDs are source-local namespaces — all 51 Kathbath IDs
collide with IndicSynth `target_speaker_id` strings"*, and instructs grouping on the composite key
to neutralise a supposedly coincidental collision. **The evidence says the collision is not
coincidental.**

- Kathbath `source_file_id`: `844424930703439-329-f.m4a`
- IndicSynth `target_reference_audio`: `844424933506642-1023-m.wav`
- **Identical `<15-digit-id>-<speaker>-<m|f>` scheme, matching 2,500/2,500 and 5,000/5,000 rows.**
- All **51/51** Kathbath speaker IDs appear in IndicSynth's 101 target speakers.
- IndicSynth's targets are 53 female / 48 male (P(female) = 0.525). Kathbath is 100% female.
  All 51 shared IDs are female on the IndicSynth side too. **P(that by chance) ≈ 5.2 × 10⁻¹⁵.**
- **169 full 15-digit utterance-ID prefixes are shared between the two corpora** — the same
  upstream recordings feed both.

Both are AI4Bharat corpora; IndicSynth's voice-conversion targets are drawn from the same
speaker pool as Kathbath. These are the same physical voices.

Consequence: **30 of the 51 speakers have their real Kathbath clips in one split and their cloned
fake IndicSynth clips in a different split** (e.g. speaker `1102` → Kathbath train, IndicSynth test;
speaker `321` → Kathbath test, IndicSynth train). Pool the two sources and 30 speakers straddle
splits. The "zero group leakage" claim holds only under the assumption that these namespaces are
unrelated — and that assumption is false. The composite key does not fix this; it *encodes* the
mistake.

This does not necessarily inflate accuracy (a speaker seen as real in train and fake in test may
hurt as easily as help), but it means **speaker identity is not actually held out across splits in
the Hindi half**, and any per-speaker generalisation claim is unsupported.

**(b) ⚠ Transcript leakage across splits, including where transcript *is* the grouping key.**

Under NFKC + punctuation-stripped normalisation:

| Source | unique | train∩val | train∩test | val∩test |
|---|---:|---:|---:|---:|
| **nameissakthi (Indic Parler)** | 1,466 | **57** | **46** | **11** |
| MUCS_OpenSLR104 | 4,681 | 40 | 47 | 12 |
| IndicSynth | 2,457 | 8 | 7 | 3 |
| CommonVoice | 2,490 | 3 | 3 | 0 |
| octopus (Azure) | 1,611 | 0 | 1 | 0 |
| Kathbath | 2,500 | 0 | 0 | 0 |

The Indic Parler row is the problem. The card states: *"For Hinglish fake there are only three
synthetic voices, so a speaker-disjoint split is impossible; splits are transcript-disjoint
instead."* Checking the actual `group_id` values:

- **Indic Parler: 1,756 clips → 1,756 distinct `group_id`s. One group per clip. The grouping is a
  no-op and the split is effectively random.**
- Azure/Octopus: 3,244 clips → 1,612 groups. Genuinely grouped, and it shows (1 collision total).

Their normalisation kept raw transcript text, so near-identical sentences differing only in
punctuation or Unicode form landed in different groups. Under stricter normalisation 114 Parler
transcript pairs straddle splits. **The "transcript-disjoint" guarantee is real for Azure and
vacuous for Indic Parler.**

**(c) ⚠ 46 transcripts appear under both labels** — the same Hindi sentence spoken by a Kathbath
human (real) and synthesised by IndicSynth XTTS-v2 (fake), often in different splits. One such
pair is `kb_hi_01499` (real, train) and `is_hi_xtts_v2_00566` (fake, test), which share a
byte-identical normalised transcript. (The sentences themselves are not reproduced here — see
`metadata/dataset_v1_master_v2.csv` locally.) Combined with (a), this is consistent with IndicSynth having
synthesised Kathbath's own utterances with Kathbath's own speakers. For content this is *desirable*
(text can't be a cue); paired with the speaker finding it confirms the two corpora are not
independent.

Transcript multiplicity is also worth knowing: **all 3,244 Azure clips reuse sentences** (1,611
unique texts, up to 10 clips each), and MUCS reuses one sentence up to 27 times.

### 7.5 🔴 The external evaluation set has a near-perfect codec shortcut

This is the set that is supposed to give the honest generalisation number, and it is compromised.
Probing the actual FLAC subtype of **all 2,000** SEA-Spoof files:

| subtype | fake | real | total |
|---|---:|---:|---:|
| **PCM_16** | **642** | **0** | 642 |
| PCM_24 | 358 | 1,000 | 1,358 |

**Bit depth alone identifies 642 of 1,000 spoofs with zero false positives.** The rule
"PCM_16 ⇒ fake" has 100% precision and 64.2% recall; thresholding on bit depth alone scores
**82.1% accuracy** on a set built to be a fair 50/50 test. All 1,000 bonafide files are PCM_24.

The cause is upstream and per-generator — each TTS system wrote a fixed depth:

| PCM_16 (all-spoof) | PCM_24 |
|---|---|
| elevenlabs, heygen, minimax, hindi_common_fastspeech, hindi_common_indic-tts, hindi_common_xttsv2, hindi_indic_fastspeech, hindi_indic_indic-tts, hindi_indic_tts_xtts-v2 | **bonafide (1,000)**, chatgpt_tts_flac, hindi_common_edge_tts, hindi_common_vits_mms, hindi_indic_edge_tts, hindi_indic_tts_vits-mms |

Duration is a second cue in the same set: real median 7.02 s vs fake 4.48 s, AUC 0.705, threshold
accuracy 0.658 — and one bonafide file runs **127.7 s**, wildly outside the development set's 30 s
ceiling.

**Assessment: SEA-Spoof audio must be canonicalised to a fixed bit depth before use**, exactly as
the development set was. The card treats SEA-Spoof as clean because it is external and held out;
externality does not confer freedom from confounds, and it was never put through the same
canonicalisation as the 20k. As it stands, a "generalisation" score on this set is partly a bit-depth
detector. (Declared vs actual durations do match — 0 mismatches over 0.05 s — so the manifest itself
is accurate.)

### 7.6 ⚠ SEA-Spoof shares 280 transcripts with development train/validation

280 normalised transcripts in the external eval set also appear in dev `train`/`validation`
(310 across the whole 20k). The audio is different and the bonafide half was deliberately
re-collected to exclude Common Voice–derived clips — that fix worked at the audio level. But the
*text* overlap means SEA-Spoof is not fully content-independent of training. For a pure acoustic
detector this is minor; it matters if anything text-conditioned or self-supervised-on-content is used.

### 7.7 Does leading/trailing silence differ between classes?

**Yes — substantially, and it is one of the strongest shortcuts (see 7.3).**

| | fake | real |
|---|---:|---:|
| leading silence (median) | 0.02 s | 0.04 s |
| trailing silence (median) | 0.56 s | 0.32 s |
| voiced frame fraction (median) | 0.865 | 0.707 |

Medians understate it; the *distributions* separate cleanly enough that leading silence alone gives
**96.9% accuracy within the Hindi half**. Per source the pattern is clear: TTS output is tightly
trimmed at onset (IndicSynth 0.00 s, Parler 0.02 s) while human crowdsourced recording carries
slack (Common Voice 0.44 s), and Azure carries a long tail (0.96 s trailing). The card correctly
states that no silence trimming or VAD was applied — but that is exactly why the *upstream*
difference in trimming survives intact.

---

## 8. Summary

**What is genuinely solid**

- 20,000/20,000 bidirectional label coverage; 0 missing files, 0 orphans, across 19 manifests.
- Perfect class balance: 10,000/10,000, and 5,000 in each of the 4 language×label cells.
- Canonicalisation is real: 16 kHz/mono/PCM_16 on all 20,000 files, verified individually.
- 0 duplicate content within the canonical set (20,000 unique SHA-256, confirmed from disk).
- Group-disjoint splits hold exactly on the declared key, all six sources, all three pairs.
- 0 corrupt or unreadable files in everything opened: all 20,000 canonical, all 2,000 SEA-Spoof,
  all 5,000 `hinglish_fake`, plus 300 sampled from each remaining folder.
- Documentation is unusually honest about its own limitations — the source↔label confound,
  the non-commercial licence, the unverified Azure licence, the missing physical replay.

**What is wrong — ranked by impact on any model result**

1. 🔴 **Acoustic shortcuts of ~97% accuracy in the Hindi half** (peak amplitude, clipping
   fraction, leading silence) and 92% in Hinglish. Undetected by the existing audit because it
   only scored metadata columns. Fakes are peak-normalised and tightly trimmed; humans are not.
2. 🔴 **The external eval set has a 100%-precision bit-depth shortcut** — PCM_16 ⇒ spoof, 642 of
   1,000, and 82.1% accuracy from bit depth alone. SEA-Spoof was never canonicalised.
3. 🔴 **Kathbath's real speakers are IndicSynth's cloning targets** (p ≈ 5×10⁻¹⁵, plus 169 shared
   utterance-ID prefixes). 30 of 51 speakers straddle splits once the sources are pooled. The card
   documents this as a coincidental namespace collision; it is not.
4. ⚠ **"Transcript-disjoint" is vacuous for Indic Parler** — 1,756 clips became 1,756 groups, so
   that split is random, with 114 cross-split transcript collisions under stricter normalisation.
5. ⚠ **12 dead audio files** carry confident labels — 4 are pure digital silence up to 10.7 s, and
   one of those sits in the **test** split. Silence originates upstream; the pipeline is not at fault,
   but nothing ever checked that the audio contained audio.
6. ⚠ **178 augmented files are byte-identical to their parents** (13.9% of `mild_clipping`),
   skewed 161:17 toward the real class, claiming a transformation that did not occur.

**Smaller things worth fixing**

- 5,000 `sample_id` collisions between live and archived IndicSynth, holding *different* audio.
- `kathbath_hi_v1.csv` declares `codec = wav`; the files are FLAC/PCM_24 (the master CSV is right).
- `.wav` extensions on FLAC (2,500) and MP3 (3,244) files.
- `indicsynth_hi_v2_checkpoint.csv` points at a directory that no longer exists.
- Filenames encode the label in SEA-Spoof (`ss_hi_fake_`/`ss_hi_real_`) and the generator in
  Hinglish-fake (`hf_hi_en_` vs `hf_hi_en_b_`).
- 16.47 GB in `data/raw/mucs/` is the source tarball plus its own extraction, both retained.

**Bottom line.** The dataset is well-built as an artifact — the manifests, hashing, splits and
freeze discipline are better than most research datasets. But it is not yet safe to train on and
report numbers from. Findings 1 and 2 mean a strong score would most likely be a measurement of
loudness normalisation and bit depth, not of deepfake detection, and finding 2 means the external
eval set cannot currently correct that impression. The dataset is marked FROZEN, so the fixes
belong in the loader and the reporting rather than in the files: normalise level and bit depth at
load time for both the development and evaluation sets, randomise or trim silence identically
across classes, drop or fix the 12 dead files, and publish a peak-amplitude-only baseline next to
any model result so readers can see how much of the accuracy comes for free.

---

*Methodology: all 20,000 canonical files were header-scanned and fully decoded (no sampling was
needed at that size). Source trees were probed at 300 files each with `soundfile` and `ffprobe`
cross-checked; `hinglish_fake` and `seaspoof_hi` were probed exhaustively where the finding
depended on it. Duplicate detection was exact (size-bucketed SHA-256 over all 62,963 files, then
cross-checked against the manifests' own hash columns). Nothing in the dataset was modified.*
