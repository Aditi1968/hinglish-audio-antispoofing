#!/usr/bin/env python3
"""
Common Voice Hindi 26.0 -> data/raw/<cv-corpus-dir>/

Upstream : https://commonvoice.mozilla.org/en/datasets
Licence  : CC0-1.0 (public domain dedication)
Gated    : no licence gate, but the download requires a browser session
Role     : Hindi REAL (2,500 clips)

Mozilla serves Common Voice archives through signed, expiring URLs issued after you
accept the terms in a browser. There is no anonymous API, and scraping the download
endpoint would route around a consent step that exists on purpose. So this script
prints instructions and verifies the result rather than fetching anything itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"

INSTRUCTIONS = """
Common Voice Hindi 26.0  --  manual download
============================================================================

  1. Open   https://commonvoice.mozilla.org/en/datasets
  2. Select language "Hindi" and version "Common Voice Corpus 26.0".
  3. Accept the terms and download the archive
     (roughly 1 GB for Hindi; the file is named cv-corpus-26.0-<date>-hi.tar.gz).
  4. Extract it under  data/raw/  so that you end up with:

         data/raw/<something>/cv-corpus-26.0-<date>/hi/
             clips/           (~19,000 .mp3)
             validated.tsv
             ... other .tsv files

  5. Re-run this script to verify the layout, then:

         python scripts/prepare_common_voice.py \\
             --input  data/raw/<something>/cv-corpus-26.0-<date>/hi \\
             --output data/selected/common_voice_hi_v26 \\
             --metadata metadata/common_voice_hi_v26.csv \\
             --n 2500

Licence: CC0-1.0. Attribution is not legally required, but cite the corpus anyway --
see DATA_LICENSES.md section 1 for the LREC 2020 reference.
============================================================================
"""


def find_corpus() -> list[Path]:
    if not RAW.is_dir():
        return []
    return sorted(p for p in RAW.rglob("*/hi/clips") if p.is_dir())


def main() -> int:
    print(INSTRUCTIONS)
    found = find_corpus()
    if not found:
        print("STATUS: no Common Voice Hindi corpus found under data/raw/ yet.")
        return 1

    for clips in found:
        hi = clips.parent
        n = sum(1 for _ in clips.glob("*.mp3"))
        validated = hi / "validated.tsv"
        print(f"STATUS: found {hi.relative_to(REPO)}")
        print(f"        clips/        {n:,} mp3")
        print(f"        validated.tsv {'present' if validated.is_file() else 'MISSING'}")
        if n and validated.is_file():
            print("        -> looks correct; you can run prepare_common_voice.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
