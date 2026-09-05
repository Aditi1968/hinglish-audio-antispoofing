#!/usr/bin/env python3
"""
IndicSynth (Hindi) -> data/raw/indicsynth/

Upstream : https://huggingface.co/datasets/vdivyasharma/IndicSynth
Licence  : CC BY-NC 4.0  -- attribution required, NON-COMMERCIAL
Gated    : no
Role     : Hindi FAKE (5,000 clips: 2,500 xtts_v2 + 2,500 freevc24)

This is the licence that makes the whole assembled dataset non-commercial.

Note: the upstream card advertises three Hindi generators (xtts_v2, vits, freevc24).
A footer scan of all 107 Hindi parquet shards found NO VITS in the Hindi config --
shards 0-54 are freevc24, 54-106 are xtts_v2, nothing between. V1 therefore uses
2,500 XTTS-v2 + 2,500 FreeVC24. See HANDOFF_DATASET_V1.md section 4.3.

The Hindi config is large. By default this fetches only the parquet shards V1 draws
from; pass --all to mirror the entire Hindi config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "data" / "raw" / "indicsynth"
REPO_ID = "vdivyasharma/IndicSynth"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true",
                    help="mirror the whole Hindi config, not just the shards V1 uses")
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required:  pip install -r requirements.txt")
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    patterns = ["Hindi/*"] if args.all else ["Hindi/train-000*.parquet"]

    print(f"IndicSynth  ({REPO_ID})")
    print("  licence: CC BY-NC 4.0 -- non-commercial, attribution required")
    print(f"  dest   : {args.dest.relative_to(REPO)}")
    print(f"  files  : {patterns[0]}")
    print()

    try:
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(args.dest),
            allow_patterns=patterns,
            max_workers=2,          # this machine OOMs above ~3 concurrent workers
        )
    except Exception as e:
        print(f"\ndownload failed: {type(e).__name__}: {e}")
        print("if this is an auth error, run:  hf auth login")
        return 1

    print("\ndone. Next:  python scripts/index_indicsynth_shards.py")
    print("       then:  python scripts/prepare_indicsynth_v2.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
