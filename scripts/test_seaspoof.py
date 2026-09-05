"""
Live inspection of SEA-Spoof before collecting anything.

Nothing here is taken on trust from the dataset card or dataset_infos.json.
Everything reported is read from the live repo.

Two passes:

  1. streaming smoke test - proves streaming works and shows one real row
  2. metadata scan of the official `evaluation` shards, reading every column
     EXCEPT audio, which is cheap and gives exact distributions

Writes nothing except a cached metadata table used later by the collector.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from datasets import load_dataset, Audio
from huggingface_hub import HfApi, HfFileSystem


REPO = "Jack-ppkdczgx/SEA-Spoof"
EVAL_SPLIT = "evaluation"

# Everything except the audio payload.
META_COLUMNS = [
    "row_id",
    "utterance_id",
    "text",
    "language",
    "label",
    "spoof_type",
    "category",
    "split",
    "source_model",
    "source_dataset",
    "speaker_or_voice",
    "sampling_rate",
    "audio_was_resampled",
]

CACHE_PATH = Path("metadata/evaluation/seaspoof_eval_index.csv")


def section(title):
    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)


def streaming_smoke_test():

    section("1. STREAMING SMOKE TEST")

    try:
        ds = load_dataset(REPO, split=EVAL_SPLIT, streaming=True)
    except Exception as e:
        print("Streaming FAILED:", e)
        return None

    # Inspect the audio payload without decoding it.
    ds = ds.cast_column("audio", Audio(decode=False))

    sample = next(iter(ds))

    print("Streaming works: YES")
    print("\nCOLUMNS:")
    for k in sample:
        print("   ", k)

    print("\nONE LIVE ROW (audio truncated):")
    for k, v in sample.items():
        if k == "audio":
            payload = v.get("bytes")
            print(f"    audio.path  = {v.get('path')}")
            print(f"    audio.bytes = {len(payload) if payload else None} bytes")
            if payload:
                print(f"    audio magic = {payload[:4]!r}")
        else:
            text = str(v)
            print(f"    {k} = {text[:80]}")

    return sample


def list_eval_shards():

    files = HfApi().list_repo_files(REPO, repo_type="dataset")

    return sorted(
        f for f in files
        if f.startswith(f"data/{EVAL_SPLIT}/") and f.endswith(".parquet")
    )


def scan_evaluation_metadata(shards):
    """Read all non-audio columns of the evaluation split."""

    section("2. EVALUATION SPLIT METADATA SCAN (no audio downloaded)")

    fs = HfFileSystem()
    frames = []

    for shard in shards:

        path = f"datasets/{REPO}/{shard}"

        with fs.open(path, "rb") as handle:
            pf = pq.ParquetFile(handle)

            available = set(pf.schema_arrow.names)
            columns = [c for c in META_COLUMNS if c in available]

            table = pf.read(columns=columns)

        frame = table.to_pandas()
        frame["shard"] = shard

        # Row-group boundaries let the collector fetch only what it needs.
        with fs.open(path, "rb") as handle:
            meta = pq.ParquetFile(handle).metadata

        rg_ids = []
        for rg_index in range(meta.num_row_groups):
            rg_ids.extend([rg_index] * meta.row_group(rg_index).num_rows)

        frame["row_group"] = rg_ids
        frame["row_in_group"] = frame.groupby("row_group").cumcount()

        frames.append(frame)

        print(f"    scanned {shard}  rows={len(frame):,} "
              f"row_groups={meta.num_row_groups}")

    return pd.concat(frames, ignore_index=True)


def report(df):

    section("3. LIVE SCHEMA AND VALUE REPORT")

    print("Repo:            ", REPO)
    print("Config:           default (single config, no language configs)")
    print("Evaluation rows:  ", f"{len(df):,}")

    print("\nLANGUAGE VALUES:")
    print(df["language"].value_counts().to_string())

    print("\nLABEL VALUES:")
    print(df["label"].value_counts().to_string())

    hi = df[df["language"] == "hi"]

    section("4. HINDI SUBSET OF THE OFFICIAL EVALUATION SPLIT")

    print("Hindi rows:", f"{len(hi):,}")

    print("\nHindi label counts:")
    print(hi["label"].value_counts().to_string())

    print("\nHindi category counts:")
    print(hi["category"].value_counts().to_string())

    print("\nHindi spoof_type counts:")
    print(hi["spoof_type"].value_counts(dropna=False).to_string())

    if "source_model" in hi:
        print("\nHindi source_model counts:")
        print(hi["source_model"].value_counts(dropna=False).head(25).to_string())

    if "source_dataset" in hi:
        print("\nHindi source_dataset counts:")
        print(hi["source_dataset"].value_counts(dropna=False).head(25).to_string())

    print("\nspoof_type x label (Hindi):")
    print(pd.crosstab(hi["spoof_type"], hi["label"]))

    section("5. IDENTIFIER AVAILABILITY (Hindi)")

    for column in ["utterance_id", "row_id", "speaker_or_voice"]:

        if column not in hi:
            print(f"{column}: COLUMN ABSENT")
            continue

        values = hi[column]
        non_null = values.notna() & (values.astype(str).str.strip() != "")

        print(f"\n{column}")
        print(f"    present   {int(non_null.sum()):,}/{len(hi):,}")
        print(f"    unique    {values[non_null].nunique():,}")
        print(f"    examples  {list(values[non_null].unique()[:5])}")

    if "speaker_or_voice" in hi:
        spk = hi["speaker_or_voice"]
        usable = spk.notna() & (spk.astype(str).str.strip() != "")
        print("\nspeaker_or_voice usable as a speaker ID:",
              "YES" if usable.any() else "NO")
        if usable.any():
            print("    unique speakers by label:")
            print(hi[usable].groupby("label")["speaker_or_voice"]
                  .nunique().to_string())

    section("6. AUDIO FORMAT (declared)")

    if "sampling_rate" in hi:
        print("sampling_rate values:")
        print(hi["sampling_rate"].value_counts().to_string())

    if "audio_was_resampled" in hi:
        print("\naudio_was_resampled values:")
        print(hi["audio_was_resampled"].value_counts().to_string())

    section("7. FEASIBILITY FOR 1000 REAL + 1000 FAKE (Hindi, evaluation)")

    real = int((hi["label"] == "bonafide").sum())
    fake = int((hi["label"] == "spoof").sum())

    print(f"bonafide available: {real:,}   need 1,000   "
          f"{'OK' if real >= 1000 else 'NOT ENOUGH'}")
    print(f"spoof available:    {fake:,}   need 1,000   "
          f"{'OK' if fake >= 1000 else 'NOT ENOUGH'}")

    spoof_types = hi[hi["label"] == "spoof"]["spoof_type"].nunique()
    print(f"\ndistinct Hindi spoof types: {spoof_types} "
          "(collector will spread the 1,000 fakes across these)")

    return hi


def main():

    streaming_smoke_test()

    shards = list_eval_shards()

    print(f"\nEvaluation shards: {len(shards)}")
    for s in shards:
        print("   ", s)

    df = scan_evaluation_metadata(shards)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)

    report(df)

    print("\nEvaluation metadata index cached at:", CACHE_PATH)
    print("No audio was downloaded and nothing was preprocessed.")


if __name__ == "__main__":
    main()
