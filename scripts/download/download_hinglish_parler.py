#!/usr/bin/env python3
"""
nameissakthi/hindi-english-bilingual -> data/raw/hinglish_parler/

Upstream : https://huggingface.co/datasets/nameissakthi/hindi-english-bilingual
Licence  : CC BY 4.0 -- attribution required
Gated    : no
Role     : Hinglish FAKE, source A (1,756 clips)
Voice    : "Rani", from ai4bharat/indic-parler-tts

Size caveat: the upstream card advertises 23,277 utterances and claims 10,094 are
Hinglish. Transcript script analysis found only 1,756 are genuinely code-switched --
the rest are pure Hindi or pure English mislabelled as Hinglish. V1 uses the 1,756,
and tops the Hinglish-fake half up to 5,000 with download_octopus_tts.py.

Because this is TTS output, the upstream *model's* terms (indic-parler-tts) are a
separate question from this dataset repo's CC BY 4.0 tag.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "data" / "raw" / "hinglish_parler"
REPO_ID = "nameissakthi/hindi-english-bilingual"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--metadata-only", action="store_true",
                    help="fetch only metadata.jsonl (prepare_hinglish_fake.py streams "
                         "the audio it actually needs)")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("huggingface_hub is required:  pip install -r requirements.txt")
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"Indic Parler-TTS Hinglish  ({REPO_ID})")
    print("  licence: CC BY 4.0 -- attribution required")
    print(f"  dest   : {args.dest.relative_to(REPO)}")
    print()

    try:
        if args.metadata_only:
            p = hf_hub_download(REPO_ID, "metadata.jsonl", repo_type="dataset",
                                local_dir=str(args.dest))
            print(f"  fetched {Path(p).name}")
        else:
            snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                              local_dir=str(args.dest), max_workers=2)
    except Exception as e:
        print(f"\ndownload failed: {type(e).__name__}: {e}")
        print("if this is an auth error, run:  hf auth login")
        return 1

    print("\ndone. Next:  python scripts/prepare_hinglish_fake.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
