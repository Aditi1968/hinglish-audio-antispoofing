"""
Freeze the approved IndicSynth Hindi subset.

Writes a SHA256 manifest of the 5,000 selected audio files and a compact
dataset_summary.json. Reads audio only - nothing is modified.
"""

import hashlib
import json

from pathlib import Path

import pandas as pd


METADATA_PATH = Path("metadata/indicsynth_hi.csv")
AUDIO_DIR = Path("data/selected/indicsynth_hi")

MANIFEST_PATH = Path("metadata/indicsynth_hi_sha256.csv")
SUMMARY_PATH = Path("metadata/indicsynth_hi_dataset_summary.json")

CHUNK = 1024 * 1024


def sha256_of(path):

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)

    return digest.hexdigest()


def main():

    df = pd.read_csv(METADATA_PATH)

    print(f"Hashing {len(df):,} files...")

    records = []

    for i, row in enumerate(df.itertuples(index=False), start=1):

        path = Path(row.file_path)

        records.append({
            "sample_id": row.sample_id,
            "file_path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        })

        if i % 1000 == 0:
            print(f"    {i:,}/{len(df):,}")

    manifest = pd.DataFrame(records)
    manifest.to_csv(MANIFEST_PATH, index=False)

    duplicate_hashes = int(manifest["sha256"].duplicated().sum())

    print(f"\nManifest written: {MANIFEST_PATH}")
    print(f"Duplicate audio payloads (identical sha256): {duplicate_hashes}")

    # ---- summary ----

    crosstab = pd.crosstab(df["generator"], df["split"])

    speakers = {
        g: set(part["target_speaker_id"].astype(str))
        for g, part in df.groupby("generator")
    }

    split_speakers = {
        split: set(
            df[df["split"] == split]["target_speaker_id"].astype(str)
        )
        for split in ["train", "validation", "test"]
    }

    overlaps = {
        "train_validation": len(
            split_speakers["train"] & split_speakers["validation"]
        ),
        "train_test": len(
            split_speakers["train"] & split_speakers["test"]
        ),
        "validation_test": len(
            split_speakers["validation"] & split_speakers["test"]
        ),
    }

    summary = {
        "dataset": "IndicSynth Hindi (fake) - Dataset V1",
        "source_repo": "vdivyasharma/IndicSynth",
        "config": "Hindi",
        "label": "fake",
        "language": "Hindi",

        "total_samples": int(len(df)),

        "generator_counts": {
            str(k): int(v)
            for k, v in df["generator"].value_counts().items()
        },

        "speaker_counts": {
            "overall": int(df["target_speaker_id"].nunique()),
            "xtts_v2": len(speakers.get("xtts_v2", set())),
            "freevc24": len(speakers.get("freevc24", set())),
            "shared": len(
                speakers.get("xtts_v2", set())
                & speakers.get("freevc24", set())
            ),
        },

        "split_counts": {
            str(k): int(v)
            for k, v in df["split"].value_counts().items()
        },

        "unique_speakers_per_split": {
            split: len(spk)
            for split, spk in split_speakers.items()
        },

        "generator_x_split": {
            str(gen): {str(s): int(n) for s, n in row.items()}
            for gen, row in crosstab.iterrows()
        },

        "speaker_overlap": overlaps,

        "zero_speaker_overlap_confirmed": all(
            v == 0 for v in overlaps.values()
        ),

        "clips_per_speaker": {
            "min": int(df["target_speaker_id"].value_counts().min()),
            "median": float(df["target_speaker_id"].value_counts().median()),
            "mean": round(
                float(df["target_speaker_id"].value_counts().mean()), 2
            ),
            "max": int(df["target_speaker_id"].value_counts().max()),
        },

        "audio_state": (
            "original bytes as published; not resampled, denoised, "
            "normalized, trimmed or augmented"
        ),

        "license": "CC BY-NC 4.0",

        "sha256_manifest": str(MANIFEST_PATH),
        "total_bytes": int(manifest["bytes"].sum()),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Summary written: {SUMMARY_PATH}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
