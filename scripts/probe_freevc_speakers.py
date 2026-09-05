"""
Measure the TRUE distinct target-speaker pool for freevc24 in IndicSynth Hindi.

The v2 collector stalled at exactly 53 speakers x cap 30 = 1,590 clips, so we
need to know whether the pool really is that small or whether the row-group
sampling simply kept re-reading the same speakers.

Reads ONLY the "Target Speaker ID" column (plus generator) - no audio.
"""

from concurrent.futures import ThreadPoolExecutor
from collections import Counter

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from huggingface_hub import HfFileSystem


REPO = "vdivyasharma/IndicSynth"
INDEX_PATH = Path("metadata/indicsynth_hi_shard_index.csv")
OUT_PATH = Path("metadata/indicsynth_hi_freevc_speakers.json")

MAX_WORKERS = 6


def scan(shard):

    fs = HfFileSystem()

    with fs.open(f"datasets/{REPO}/{shard}", "rb") as handle:
        table = pq.read_table(
            handle,
            columns=["Generative Model", "Target Speaker ID"],
        )

    frame = table.to_pandas()
    frame = frame[frame["Generative Model"] == "freevc24"]

    return shard, Counter(frame["Target Speaker ID"].dropna().astype(int))


def main():

    index = pd.read_csv(INDEX_PATH)

    shards = sorted(
        index[index["generator_min"] == "freevc24"]["shard"].unique()
    )

    print(f"freevc24 shards to scan: {len(shards)}")

    overall = Counter()
    per_shard = {}
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:

        for shard, counts in pool.map(scan, shards):

            per_shard[shard] = len(counts)
            overall.update(counts)
            done += 1

            print(
                f"    {done:>3}/{len(shards)} {shard.split('/')[-1]} | "
                f"speakers in shard {len(counts):>3} | "
                f"pool so far {len(overall):>4}"
            )

    print("\n=========================================")
    print("FREEVC24 SPEAKER POOL")
    print("=========================================")

    print("\nDistinct target speakers:", len(overall))
    print("Total freevc24 rows:", sum(overall.values()))

    clips = pd.Series(overall)

    print("\nClips per speaker:")
    print(f"    min    {clips.min()}")
    print(f"    median {clips.median()}")
    print(f"    mean   {round(clips.mean(), 1)}")
    print(f"    max    {clips.max()}")

    print("\nSpeakers per shard:")
    shard_counts = pd.Series(per_shard)
    print(f"    min {shard_counts.min()}  median {shard_counts.median()}  "
          f"max {shard_counts.max()}")

    for cap in [20, 25, 30, 40, 50]:
        reachable = int(clips.clip(upper=cap).sum())
        needed = -(-2500 // cap)
        print(
            f"\n  cap {cap:>3}: max clips {reachable:,} | "
            f"speakers needed for 2,500 = {needed}"
        )

    OUT_PATH.write_text(
        json.dumps(
            {str(k): int(v) for k, v in overall.items()},
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSpeaker histogram saved to:", OUT_PATH)


if __name__ == "__main__":
    main()
