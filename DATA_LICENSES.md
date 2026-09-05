# Data Licences and Provenance

**No audio is redistributed by this repository.** This repo contains code, documentation and
redacted metadata only. Every audio file must be obtained by you, from the upstream source,
under that source's own terms. `scripts/download/` automates this where the upstream licence
permits automation, and points you at the authors' request page where it does not.

Licences below were **checked against the upstream page on 2026-09-05**, not copied from the
project's own earlier notes. Where upstream is ambiguous or silent, this file says so rather
than picking the convenient reading.

---

## Summary table

| # | Source | Role in V1 | Clips | Upstream | Licence (as found upstream) | Status | Redistribution |
|---|---|---|---:|---|---|---|---|
| 1 | Common Voice Hindi 26.0 | Hindi real | 2,500 | [commonvoice.mozilla.org/en/datasets](https://commonvoice.mozilla.org/en/datasets) | **CC0-1.0** | VERIFIED | Permitted by licence — but not done here |
| 2 | Kathbath (IndicSUPERB) | Hindi real | 2,500 | [hf.co/datasets/ai4bharat/Kathbath](https://huggingface.co/datasets/ai4bharat/Kathbath) | **Ambiguous: tag `cc-by-4.0`, card text says CC0** | ⚠ CONFLICTING | Not redistributed |
| 3 | IndicSynth | Hindi fake | 5,000 | [hf.co/datasets/vdivyasharma/IndicSynth](https://huggingface.co/datasets/vdivyasharma/IndicSynth) | **CC BY-NC 4.0** | VERIFIED | Non-commercial + attribution |
| 4 | MUCS / OpenSLR-104 | Hinglish real | 5,000 | [openslr.org/104](https://www.openslr.org/104/) | **CC BY-SA 4.0** | VERIFIED | Share-alike applies |
| 5 | nameissakthi/hindi-english-bilingual | Hinglish fake (Indic Parler-TTS) | 1,756 | [hf.co/datasets/nameissakthi/hindi-english-bilingual](https://huggingface.co/datasets/nameissakthi/hindi-english-bilingual) | **CC BY 4.0** | VERIFIED | Attribution |
| 6 | lingamvamshikrishnareddy/octopus-tts-hinglish | Hinglish fake (Azure/Edge TTS) | 3,244 | [hf.co/datasets/lingamvamshikrishnareddy/octopus-tts-hinglish](https://huggingface.co/datasets/lingamvamshikrishnareddy/octopus-tts-hinglish) | **None stated anywhere** | ❌ **UNVERIFIED** | **Do not redistribute** |
| 7 | SEA-Spoof (Hindi eval) | External eval | 2,000 | [hf.co/datasets/Jack-ppkdczgx/SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof) | **"other"** — non-commercial academic, gated | VERIFIED | **Prohibited without author approval** |

**Aggregate terms for the 20,000-clip development pool: NON-COMMERCIAL.** IndicSynth's
CC BY-NC 4.0 is the most restrictive licence in the pool and it governs any derived
combination. MUCS's CC BY-SA 4.0 adds a share-alike obligation on derivatives of that portion.
One source (#6) has no stated licence at all, so its terms are unknown rather than permissive.

---

## Per-source detail

### 1. Common Voice Hindi 26.0 — CC0-1.0 ✅

- **Licence:** CC0 1.0 Universal (public domain dedication). Mozilla releases each Common
  Voice corpus under CC0; sentence text is CC0 via the Sentence Collector or Wikipedia.
- **Access:** free download, no approval; Mozilla asks you to accept terms in the browser.
  Not scriptable without a session token, so `scripts/download/` gives manual instructions.
- **Caveat noted upstream:** files matching `europarl-VERSION-LANG.txt` derive from the
  Europarl Corpus and have separate provenance. Not used by this project (Hindi only).
- **Citation:**
  ```bibtex
  @inproceedings{commonvoice:2020,
    author    = {Ardila, R. and Branson, M. and Davis, K. and Kohler, M. and Meyer, J. and
                 Henretty, M. and Morais, R. and Saunders, L. and Tyers, F. M. and Weber, G.},
    title     = {Common Voice: A Massively-Multilingual Speech Corpus},
    booktitle = {Proceedings of the 12th Conference on Language Resources and Evaluation (LREC 2020)},
    pages     = {4218--4222},
    year      = {2020}
  }
  ```
  Mozilla's own BibTeX prints pages 4211–4215; the ACL Anthology record
  ([2020.lrec-1.520](https://aclanthology.org/2020.lrec-1.520/)) gives 4218–4222. Use the
  Anthology numbers if your venue checks them. Cite the corpus version too (26.0, Hindi).

### 2. Kathbath / IndicSUPERB — ⚠ upstream states two different licences

**This is a genuine, unresolved upstream conflict and this project does not resolve it.**

- The HF dataset card's **licence tag reads `cc-by-4.0`**.
- The card's **licensing prose reads**: *"We license the actual packaging of all this data
  under the Creative Commons CC0 license ('no rights reserved')."*
- **Access:** gated — HF requires you to agree to share contact information before download.

Both readings appear on the same page. CC0 and CC BY 4.0 differ materially: CC BY requires
attribution, CC0 does not. This repo's earlier documentation recorded "CC BY 4.0 VERIFIED"
(matching the tag) and the per-source manifest `metadata/kathbath_hi_v1.csv` recorded
`unknown` — see "Known discrepancies" below.

**Recommended handling:** treat as **CC BY 4.0** (the stricter of the two) and attribute.
Attribution satisfies both licences; assuming CC0 does not.

- **Citation:**
  ```bibtex
  @misc{javed2022indicsuperb,
    doi       = {10.48550/ARXIV.2208.11761},
    url       = {https://arxiv.org/abs/2208.11761},
    author    = {Javed, Tahir and Bhogale, Kaushal Santosh and Raman, Abhigyan and
                 Kunchukuttan, Anoop and Kumar, Pratyush and Khapra, Mitesh M.},
    title     = {IndicSUPERB: A Speech Processing Universal Performance Benchmark for Indian languages},
    publisher = {arXiv},
    year      = {2022}
  }
  ```

### 3. IndicSynth — CC BY-NC 4.0 ✅ (this is what makes the whole set non-commercial)

- **Licence:** `cc-by-nc-4.0`. Attribution required; **commercial use prohibited**.
- **Access:** public, not gated.
- **Upstream card claims three Hindi generators** (xtts_v2, vits, freevc24). This project's
  own footer scan of all 107 Hindi parquet shards found **no VITS in the Hindi config** —
  shards 0–54 are freevc24, 54–106 are xtts_v2, nothing between. V1 uses 2,500 XTTS-v2 +
  2,500 FreeVC24. See `HANDOFF_DATASET_V1.md` §4.3.
- **Citation:**
  ```bibtex
  @inproceedings{sharma2025indicsynth,
    title     = {IndicSynth: A Large-Scale Multilingual Synthetic Speech Dataset for
                 Low-Resource Indian Languages},
    author    = {Sharma, Divya and others},
    booktitle = {Proceedings of ACL 2025},
    address   = {Vienna, Austria},
    year      = {2025}
  }
  ```
  (ACL 2025 Outstanding Paper. Verify the full author list against the published version.)

### 4. MUCS / OpenSLR-104 — CC BY-SA 4.0 ✅

- **Licence:** verbatim from the OpenSLR SLR104 page: **"CC BY-SA 4.0"**. Confirmed
  2026-09-05. Attribution **and share-alike** — derivatives of this portion must be
  released under a compatible licence.
- **Access:** direct HTTP download, no account. openslr.org throttles to ~0.35 MB/s per
  connection, so `scripts/download/download_mucs.py` uses ranged parallel requests.
- **Files:** `Hindi-English_train.tar.gz` (6.83 GB) is the one V1 uses.
- **Domain caveat:** this is technical/tutorial Hinglish (Linux, LibreOffice, Bash spoken
  tutorials), not broad conversational Hinglish.
- **Citation:** Multilingual and code-switching ASR challenges for low resource Indian
  languages (MUCS 2021), <https://navana-tech.github.io/MUCS2021/>.

### 5. nameissakthi/hindi-english-bilingual — CC BY 4.0 ✅

- **Licence:** tag `cc-by-4.0`. Attribution required.
- **Audio origin:** synthesised with the **Rani** voice of `ai4bharat/indic-parler-tts`.
- **Upstream size vs usable size:** the card advertises 23,277 utterances across Hindi,
  English and Hinglish splits (~4.7 GB, 24 kHz WAV) and claims 10,094 Hinglish. This
  project's transcript script analysis found **only 1,756 are genuinely code-switched**;
  the rest are pure Hindi or pure English mislabelled as Hinglish. V1 uses the 1,756.
- **Note:** because this is TTS output from Indic Parler-TTS, the upstream *model's* terms
  are a separate question from the *dataset's* CC BY 4.0 tag.

### 6. lingamvamshikrishnareddy/octopus-tts-hinglish — ❌ NO LICENCE STATED

**Re-checked 2026-09-05. The answer is still "unverified", and access has tightened.**

- **No LICENSE file, no README, no licence tag, no dataset card** ("No dataset card yet").
  All commits are bulk uploads with no stated terms.
- **Newly observed:** the repo is now **gated** — *"You need to agree to share your contact
  information to access this dataset"* — which was not recorded in the earlier project notes.
- **Status: UNVERIFIED. Do not redistribute this audio.** All derived rows in the local
  manifests carry `license_status = UNVERIFIED` and `redistribution_allowed = false`.
- **Separate unresolved question:** this audio was produced with **Microsoft Azure / Edge
  neural voices** (`hi-IN-Madhur`, `hi-IN-Swara`). Azure's terms for synthesised speech
  output are an independent matter from the HF repo's silence, and have **not** been
  resolved by this project. If you need certainty here, resolve both before using these
  3,244 clips for anything beyond local research.

### 7. SEA-Spoof — gated, non-commercial academic, no redistribution ⚠

- **Licence field:** `other`.
- **Access terms (verbatim):** *"Access is restricted to approved non-commercial academic
  research users. The authors will review each request manually."* Requesters must email the
  authors with name, affiliation, intended research use, and confirmation that the use is
  academic and non-commercial.
- **Restrictions:** non-commercial academic only; **cannot be redistributed without author
  approval**; prohibited uses include speaker impersonation, voice cloning, surveillance and
  harmful audio generation.
- **This repo's local note** (`license_or_access_note` in `metadata/evaluation/seaspoof_hi.csv`)
  reads: *"SEA-Spoof; gated repo (manual approval required); license: other - see repo LICENSE;
  research use"* — consistent with upstream.
- **Citation:**
  ```bibtex
  @article{wu2025sea,
    title   = {SEA-Spoof: Bridging The Gap in Multilingual Audio Deepfake Detection for South-East Asian},
    author  = {Wu, Jinyang and Hou, Nana and Pan, Zihan and Zhang, Qiquan and
               Bhupendra, Sailor Hardik and Mondal, Soumik},
    journal = {arXiv preprint arXiv:2509.19865},
    year    = {2025}
  }
  ```
- **Do not** attempt to script this download. `scripts/download/download_seaspoof.py` prints
  the request procedure and exits.

---

## Known discrepancies found during this audit

Recorded rather than silently corrected, because the local manifests are frozen.

| Where | Says | Upstream / actual | Note |
|---|---|---|---|
| `metadata/kathbath_hi_v1.csv` → `license` | `unknown` (all 2,500 rows) | tag `cc-by-4.0` / prose CC0 | The per-source manifest was never updated; `dataset_v1_master_v2.csv` records `CC BY 4.0 / VERIFIED` for the same rows. **Trust the master.** |
| `DATASET_V1_CARD.md` §G | Kathbath "CC BY 4.0 VERIFIED" | upstream states **both** CC BY 4.0 and CC0 | "VERIFIED" overstates it; upstream is self-contradictory. |
| `DATASET_V1_CARD.md` §G | octopus "none stated upstream" | still true, **and now gated** | Gating is new since collection. |
| `metadata/kathbath_hi_v1.csv` → `codec` | `wav` | files are **FLAC/PCM_24** | See `DATASET_REPORT.md` §3.2. Master manifest is correct. |
| Kathbath vs IndicSynth speaker IDs | documented as a coincidental namespace collision | **almost certainly the same AI4Bharat speakers** (p ≈ 5×10⁻¹⁵) | `DATASET_REPORT.md` §7.4a. Relevant here because it means clip-level provenance links two differently-licensed corpora. |

---

## What you may do with this repository

- **Code** (`scripts/`): MIT — see [`LICENSE`](LICENSE).
- **Redacted metadata** (`metadata/public/`): released under
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) to stay compatible with the
  MUCS share-alike obligation it partly describes. It contains no transcripts, no speaker
  identifiers and no audio.
- **Audio**: not in this repo, not obtainable from this repo. Get it upstream, under the
  terms above.
- **Any model or result derived from the assembled 20,000-clip pool is non-commercial**, by
  IndicSynth's CC BY-NC 4.0.

If you redistribute anything derived from this work, reproduce this file with it.
