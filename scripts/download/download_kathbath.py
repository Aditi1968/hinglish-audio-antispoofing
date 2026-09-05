#!/usr/bin/env python3
"""
Kathbath / IndicSUPERB (Hindi) -> data/raw/kathbath/

Upstream : https://huggingface.co/datasets/ai4bharat/Kathbath
Licence  : AMBIGUOUS UPSTREAM -- the HF tag reads `cc-by-4.0`, the card prose reads
           "We license the actual packaging of all this data under the Creative
           Commons CC0 license". Treat as CC BY 4.0 (the stricter reading) and
           attribute; attribution satisfies both. See DATA_LICENSES.md section 2.
Gated    : YES -- you must accept the terms on the dataset page first
Role     : Hindi REAL (2,500 clips, 51 speakers, all female)

IMPORTANT (see DATASET_REPORT.md section 7.4a): Kathbath's speaker IDs are NOT an
independent namespace from IndicSynth's `target_speaker_id`. These are the same
AI4Bharat speakers -- the humans here are the people IndicSynth cloned. Keep that in
mind when designing splits; do not assume the two corpora are speaker-independent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "data" / "raw" / "kathbath"
REPO_ID = "ai4bharat/Kathbath"

GATE_HELP = f"""
This dataset is GATED. Before this script can work:

  1. Log in at  https://huggingface.co/datasets/{REPO_ID}
  2. Accept the terms on that page (it asks you to share contact information).
  3. Authenticate locally:   hf auth login
     (older huggingface_hub:  huggingface-cli login)

Then re-run this script. Do not attempt to bypass the gate.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required:  pip install -r requirements.txt")
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"Kathbath / IndicSUPERB  ({REPO_ID})")
    print("  licence: upstream states BOTH cc-by-4.0 (tag) and CC0 (prose)")
    print("           -> attribute anyway; see DATA_LICENSES.md")
    print(f"  dest   : {args.dest.relative_to(REPO)}")
    print()

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(args.dest),
            allow_patterns=["*hindi*", "*Hindi*"],
            max_workers=2,
        )
    except Exception as e:
        name = type(e).__name__
        print(f"\ndownload failed: {name}: {e}")
        if "Gated" in name or "401" in str(e) or "403" in str(e) or "Unauthorized" in str(e):
            print(GATE_HELP)
        return 1

    print("\ndone. Next:  python scripts/prepare_kathbath.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
