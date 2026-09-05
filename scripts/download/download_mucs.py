#!/usr/bin/env python3
"""
MUCS / OpenSLR-104 Hindi-English -> data/raw/mucs/

Upstream : https://www.openslr.org/104/
Licence  : CC BY-SA 4.0  -- attribution AND share-alike
Gated    : no
Role     : Hinglish REAL (5,000 utterances cut from 521 long recordings)

openslr.org throttles each connection to ~0.35 MB/s, so a single stream would take
~6 hours for the 6.8 GB Hindi-English train archive. The server supports HTTP Range,
so the downloader fetches 8 byte ranges concurrently and concatenates them. Each part
resumes independently -- an interrupted run costs nothing.

Only the Hindi-English archives are listed. Bengali-English is deliberately absent:
this project must not mix it in.

Domain caveat: MUCS is technical/tutorial Hinglish (Linux, LibreOffice, Bash spoken
tutorials), not broad conversational Hinglish. Its code-switching is driven by
technical vocabulary. Speaker and recording are nearly confounded -- 520 speakers
across 521 recordings.

This is a thin wrapper around the original scripts/download_mucs.py, kept so that
every source has one entry point under scripts/download/.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORIGINAL = REPO / "scripts" / "download_mucs.py"


def main() -> int:
    if not ORIGINAL.is_file():
        print(f"missing {ORIGINAL.relative_to(REPO)}")
        return 1

    print("MUCS / OpenSLR-104 Hindi-English")
    print("  licence: CC BY-SA 4.0 -- attribution AND share-alike")
    print("  dest   : data/raw/mucs/")
    print("  note   : ~6.8 GB archive; extraction adds ~9.6 GB.")
    print("           You can delete the .tar.gz once extracted.")
    print()

    # run from the repo root: the original uses repo-relative paths
    sys.path.insert(0, str(REPO / "scripts"))
    runpy.run_path(str(ORIGINAL), run_name="__main__")

    print("\ndone. Extract data/raw/mucs/Hindi-English_train.tar.gz into")
    print("      data/raw/mucs/extracted/, then:  python scripts/prepare_mucs.py")
    return 0


if __name__ == "__main__":
    import os
    os.chdir(REPO)
    sys.exit(main())
