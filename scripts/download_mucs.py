"""
Resumable parallel downloader for the official MUCS / OpenSLR-104 archive.

openslr.org throttles each connection to ~0.35 MB/s, which would make the
7.3 GB Hindi-English train archive a ~6 hour single-stream download. The
server supports HTTP Range (206), so this fetches N byte ranges concurrently
and concatenates them.

Each part is its own file and resumes from wherever it stopped, so an
interrupted run costs nothing. Re-running is always safe.

ONLY the Hindi-English archives are listed here. Bengali-English is
deliberately absent - this project must not mix it in.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sys
import time

import httpx


BASE = "https://www.openslr.org/resources/104"

ARCHIVES = {
    "train": "Hindi-English_train.tar.gz",
    "test": "Hindi-English_test.tar.gz",
}

DEST_DIR = Path("data/raw/mucs")

PARTS = 8
CHUNK = 1024 * 1024
TIMEOUT = 120


def total_size(url):

    with httpx.Client(follow_redirects=True, timeout=TIMEOUT) as client:
        response = client.head(url)
        response.raise_for_status()

        if response.headers.get("accept-ranges") != "bytes":
            raise RuntimeError("server does not advertise range support")

        return int(response.headers["content-length"])


def download_part(args):

    url, index, start, end, part_path = args

    # Resume: skip whatever this part already holds.
    have = part_path.stat().st_size if part_path.exists() else 0
    want = end - start + 1

    if have >= want:
        return index, have, True

    attempt = 0

    while have < want and attempt < 10:

        attempt += 1
        headers = {"Range": f"bytes={start + have}-{end}"}

        try:
            with httpx.Client(follow_redirects=True, timeout=TIMEOUT) as client:
                with client.stream("GET", url, headers=headers) as response:

                    if response.status_code not in (200, 206):
                        time.sleep(2 * attempt)
                        continue

                    with open(part_path, "ab") as handle:
                        for block in response.iter_bytes(CHUNK):
                            handle.write(block)
                            have += len(block)

        except Exception:
            # Network blips are expected over a multi-GB transfer.
            have = part_path.stat().st_size if part_path.exists() else 0
            time.sleep(2 * attempt)

    return index, have, have >= want


def download(name):

    filename = ARCHIVES[name]
    url = f"{BASE}/{filename}"

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    destination = DEST_DIR / filename

    size = total_size(url)
    print(f"{filename}: {size / 1e9:.2f} GB")

    if destination.exists() and destination.stat().st_size == size:
        print("    already complete")
        return destination

    span = size // PARTS
    jobs = []

    for i in range(PARTS):
        start = i * span
        end = size - 1 if i == PARTS - 1 else (start + span - 1)
        jobs.append((url, i, start, end, DEST_DIR / f"{filename}.part{i}"))

    done_bytes = sum(
        p.stat().st_size for _, _, _, _, p in jobs if p.exists()
    )
    print(f"    resuming from {done_bytes / 1e9:.2f} GB\n")

    started = time.time()

    with ThreadPoolExecutor(max_workers=PARTS) as pool:
        results = list(pool.map(download_part, jobs))

    elapsed = time.time() - started

    failed = [i for i, _, ok in results if not ok]

    if failed:
        print(f"\n    parts incomplete: {failed} - re-run to resume")
        return None

    print(f"\n    downloaded in {elapsed / 60:.1f} min, joining parts...")

    with open(destination, "wb") as out:
        for _, _, _, _, part_path in jobs:
            with open(part_path, "rb") as handle:
                while True:
                    block = handle.read(CHUNK * 8)
                    if not block:
                        break
                    out.write(block)

    actual = destination.stat().st_size

    if actual != size:
        raise RuntimeError(f"size mismatch: {actual} != {size}")

    for _, _, _, _, part_path in jobs:
        part_path.unlink()

    print(f"    OK {destination} ({actual / 1e9:.2f} GB)")

    return destination


def main():

    which = sys.argv[1] if len(sys.argv) > 1 else "train"

    if which not in ARCHIVES:
        raise SystemExit(f"unknown archive '{which}'; use one of {list(ARCHIVES)}")

    download(which)


if __name__ == "__main__":
    main()
