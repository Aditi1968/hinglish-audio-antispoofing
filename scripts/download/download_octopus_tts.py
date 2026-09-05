#!/usr/bin/env python3
"""
lingamvamshikrishnareddy/octopus-tts-hinglish -> data/raw/octopus_tts/

Upstream : https://huggingface.co/datasets/lingamvamshikrishnareddy/octopus-tts-hinglish
Licence  : *** NONE STATED ANYWHERE ***
Gated    : YES (contact-information gate, added since original collection)
Role     : Hinglish FAKE, source B (3,244 clips)
Voices   : Azure / Edge neural TTS -- hi-IN-Madhur, hi-IN-Swara

============================================================================
READ THIS BEFORE RUNNING
============================================================================

This repo has NO README, NO LICENSE file, NO licence tag and NO dataset card
("No dataset card yet"). Every commit is a bulk upload with no stated terms.
Re-checked 2026-09-05: still true, and the repo is now gated as well.

"No licence stated" means the terms are UNKNOWN, not permissive. By default
copyright is reserved. All rows derived from this source are marked
`license_status = UNVERIFIED` and `redistribution_allowed = false`.

There is a SECOND, independent question: the audio was produced with Microsoft
Azure / Edge neural voices, whose terms for synthesised speech output are a
separate matter from this repo's silence. That has NOT been resolved here.

Consequences:
  * do not redistribute this audio;
  * do not publish a model trained on it without resolving both questions;
  * consider whether your work can stand on the other 1,756 Hinglish-fake clips
    (see download_hinglish_parler.py), which ARE clearly licensed CC BY 4.0.

This script therefore refuses to run without an explicit acknowledgement flag.
============================================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "data" / "raw" / "octopus_tts"
REPO_ID = "lingamvamshikrishnareddy/octopus-tts-hinglish"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--i-understand-no-license", action="store_true",
                    help="acknowledge that this source states no licence and that you "
                         "accept the risk of using it for local research only")
    ap.add_argument("--dest", type=Path, default=DEST)
    args = ap.parse_args()

    if not args.i_understand_no_license:
        print(__doc__)
        print("Refusing to download. Re-run with --i-understand-no-license if you have")
        print("read the above and accept the risk, for LOCAL RESEARCH USE ONLY.")
        return 2

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required:  pip install -r requirements.txt")
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"octopus-tts-hinglish  ({REPO_ID})")
    print("  licence: NONE STATED -- UNVERIFIED, do not redistribute")
    print(f"  dest   : {args.dest.relative_to(REPO)}")
    print()

    try:
        snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                          local_dir=str(args.dest), allow_patterns=["hi/*"],
                          max_workers=2)
    except Exception as e:
        name = type(e).__name__
        print(f"\ndownload failed: {name}: {e}")
        if "Gated" in name or "401" in str(e) or "403" in str(e):
            print("\nThis repo is gated. Accept the terms at")
            print(f"  https://huggingface.co/datasets/{REPO_ID}")
            print("then:  hf auth login")
        return 1

    print("\ndone. Next:  python scripts/prepare_hinglish_fake_topup.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
