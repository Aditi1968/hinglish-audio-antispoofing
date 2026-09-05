#!/usr/bin/env python3
"""
SEA-Spoof (Hindi evaluation split) -> data/evaluation/seaspoof_hi/

Upstream : https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof
Licence  : "other" -- non-commercial academic research only
Gated    : YES, with MANUAL AUTHOR APPROVAL
Role     : EXTERNAL EVALUATION (2,000 clips) -- must never enter training

This script deliberately downloads NOTHING.

Upstream terms, verbatim:

    "Access is restricted to approved non-commercial academic research users.
     The authors will review each request manually."

Redistribution is prohibited without author approval, and the terms specifically
forbid speaker impersonation, voice cloning, surveillance and harmful audio
generation. Automating around a manual review process would defeat the review.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "data" / "evaluation" / "seaspoof_hi"

NOTICE = """
SEA-Spoof  --  gated, manual approval required
============================================================================

This script will not download anything. Request access yourself:

  1. Open   https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof
  2. Follow the access request procedure on that page. You will be asked to
     email the authors with:
         - your name
         - your affiliation
         - your intended research use
         - confirmation that the use is academic and NON-COMMERCIAL
  3. Wait for manual approval. The authors review each request.
  4. Once approved, authenticate locally (`hf auth login`) and fetch the
     `evaluation` split, then run:

         python scripts/prepare_seaspoof.py

TERMS YOU ARE AGREEING TO
  - non-commercial academic research only
  - NO redistribution without author approval
  - no speaker impersonation, voice cloning, surveillance, or harmful
    audio generation

WHY THIS SET MATTERS HERE
  SEA-Spoof is this project's only external benchmark. It is frozen separately,
  excluded from every development manifest, and must never enter training.

  Before you trust a number from it, read DATASET_REPORT.md section 7.5:
  bit depth alone separates 642 of the 1,000 spoofs with ZERO false positives
  (all 1,000 bonafide files are PCM_24), giving 82.1% accuracy from a single
  header field. Canonicalise it to a fixed bit depth before evaluating.

CITATION
  Wu, J., Hou, N., Pan, Z., Zhang, Q., Bhupendra, S. H., Mondal, S. (2025).
  "SEA-Spoof: Bridging The Gap in Multilingual Audio Deepfake Detection for
  South-East Asian." arXiv:2509.19865
============================================================================
"""


def main() -> int:
    print(NOTICE)
    if DEST.is_dir():
        n = sum(1 for _ in DEST.glob("*.flac"))
        print(f"STATUS: {DEST.relative_to(REPO)} exists with {n:,} .flac files.")
        if n == 2000:
            print("        -> complete (2,000 expected).")
        return 0
    print(f"STATUS: {DEST.relative_to(REPO)} does not exist yet.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
