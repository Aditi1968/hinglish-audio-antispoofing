"""
Stage 1 of the corrected IndicSynth Hindi collection.

Builds a row-group level index of the Hindi config WITHOUT downloading audio.

Every Hindi parquet shard stores 20 row groups of ~100 rows, and the parquet
footer carries per-row-group min/max statistics for "Generative Model" and
"Target Speaker ID". Reading only the footer therefore tells us the generator
and target speaker of each row group for ~free.

That index is what lets stage 2 pick row groups for SPEAKER DIVERSITY instead
of gulping a couple of complete shards (the v0 mistake: 7 XTTS speakers).

Output: metadata/indicsynth_hi_shard_index.csv
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from huggingface_hub import HfApi, HfFileSystem


REPO = "vdivyasharma/IndicSynth"
CONFIG = "Hindi"

INDEX_PATH = Path("metadata/indicsynth_hi_shard_index.csv")

MAX_WORKERS = 8


def list_shards():

    files = HfApi().list_repo_files(REPO, repo_type="dataset")

    return sorted(
        f for f in files
        if f.startswith(f"{CONFIG}/") and f.endswith(".parquet")
    )


def index_shard(shard):
    """Footer-only read of one shard -> one record per row group."""

    # HfFileSystem is not documented as thread-safe; give each call its own.
    fs = HfFileSystem()

    path = f"datasets/{REPO}/{shard}"

    records = []

    with fs.open(path, "rb") as handle:

        meta = pq.ParquetFile(handle).metadata

        names = [
            meta.schema.column(i).name
            for i in range(meta.num_columns)
        ]

        gen_col = names.index("Generative Model")
        spk_col = names.index("Target Speaker ID")

        row_offset = 0

        for rg_index in range(meta.num_row_groups):

            rg = meta.row_group(rg_index)

            gen_stats = rg.column(gen_col).statistics
            spk_stats = rg.column(spk_col).statistics

            records.append({
                "shard": shard,
                "row_group": rg_index,
                "row_offset": row_offset,
                "num_rows": rg.num_rows,
                "total_byte_size": rg.total_byte_size,

                "generator_min":
                    None if gen_stats is None else gen_stats.min,
                "generator_max":
                    None if gen_stats is None else gen_stats.max,

                "speaker_min":
                    None if spk_stats is None else spk_stats.min,
                "speaker_max":
                    None if spk_stats is None else spk_stats.max,
            })

            row_offset += rg.num_rows

    return records


def main():

    shards = list_shards()

    print(f"Hindi shards: {len(shards)}")
    print("Reading parquet footers (no audio downloaded)...\n")

    all_records = []
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:

        for records in pool.map(index_shard, shards):

            all_records.extend(records)
            done += 1

            if done % 10 == 0 or done == len(shards):
                print(f"  indexed {done}/{len(shards)} shards")

    df = pd.DataFrame(all_records)

    # A row group is "pure" when it holds exactly one generator and one
    # target speaker. Those are the only ones stage 2 will draw from, so a
    # selected row group maps cleanly to one speaker group.
    df["pure_generator"] = df["generator_min"] == df["generator_max"]
    df["pure_speaker"] = df["speaker_min"] == df["speaker_max"]
    df["pure"] = df["pure_generator"] & df["pure_speaker"]

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INDEX_PATH, index=False)

    print("\n=========================================")
    print("SHARD INDEX COMPLETE")
    print("=========================================")

    print("\nRow groups:", len(df))
    print("Total rows:", int(df["num_rows"].sum()))

    print("\nPure row groups (single generator + single speaker):")
    print(df["pure"].value_counts())

    pure = df[df["pure"]]

    print("\nRow groups per generator (pure only):")
    print(pure["generator_min"].value_counts())

    print("\nDistinct target speakers per generator (pure only):")
    print(pure.groupby("generator_min")["speaker_min"].nunique())

    print("\nMean row-group size (MB):")
    print(round(df["total_byte_size"].mean() / 1e6, 1))

    print("\nIndex saved to:", INDEX_PATH)


if __name__ == "__main__":
    main()
