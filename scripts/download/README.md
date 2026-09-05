# `scripts/download/` — obtaining the source audio

No audio ships with this repository. These scripts fetch each source from its **official
upstream location**, under that source's own licence. Read
[`../../DATA_LICENSES.md`](../../DATA_LICENSES.md) before running any of them.

Run every script from the **repo root**, not from this directory:

```bash
python scripts/download/download_indicsynth.py
```

## The seven sources

| Script | Source | Automated? | Why not |
|---|---|---|---|
| `download_common_voice.py` | Common Voice Hindi 26.0 | ❌ instructions only | browser session required; Mozilla has no anonymous API |
| `download_kathbath.py` | Kathbath (IndicSUPERB) | ⚠ after `hf auth login` | gated: must accept terms on the HF page first |
| `download_indicsynth.py` | IndicSynth | ✅ yes | public |
| `download_mucs.py` | MUCS / OpenSLR-104 | ✅ yes | public HTTP, ranged parallel fetch |
| `download_hinglish_parler.py` | nameissakthi/hindi-english-bilingual | ✅ yes | public |
| `download_octopus_tts.py` | lingamvamshikrishnareddy/octopus-tts-hinglish | ⚠ gated, **no licence** | refuses unless `--i-understand-no-license` |
| `download_seaspoof.py` | SEA-Spoof (external eval) | ❌ **never** | manual author approval; redistribution prohibited |

## Order

Downloading is step 1 of 2. Once the raw sources are on disk, run the preparation and
preprocessing pipeline described in the root [`README.md`](../../README.md) →
*How to reproduce*. The `prepare_*.py` scripts in `scripts/` turn each raw download into the
per-source selection under `data/selected/`, then `preprocess_dataset_v1.py` produces the
canonical `data/processed/v1_16k/`.

## Disk budget

The full raw tree is ~21 GB, dominated by MUCS (6.8 GB archive + 9.6 GB extraction — you can
delete the archive after extracting). The canonical + derived output adds ~6 GB.

## A note on gated sources

Three of the seven are gated (Kathbath, octopus-tts-hinglish, SEA-Spoof). Gating is the
upstream authors' access decision. **Do not work around it, and do not mirror gated audio.**
If you cannot get approval, the honest options are to report results on the subset you can
legitimately obtain, or to substitute a source you can.

> There is no `HAV-DF` script here: that corpus is not part of Dataset V1. If you add it later,
> follow the same pattern — point at the authors' request page rather than mirroring.
