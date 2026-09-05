#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def find_one(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename!r} under {root}")
    if len(matches) > 1:
        print(f"Warning: found multiple {filename} files. Using: {matches[0]}")
    return matches[0]


def balanced_speaker_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Round-robin sampling across speakers to reduce speaker dominance."""
    rng = np.random.default_rng(seed)

    groups = {}
    for speaker, group in df.groupby("client_id", dropna=False):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        groups[str(speaker)] = list(idx)

    speakers = list(groups.keys())
    rng.shuffle(speakers)

    chosen = []
    while len(chosen) < n:
        made_progress = False
        rng.shuffle(speakers)
        for speaker in speakers:
            if groups[speaker]:
                chosen.append(groups[speaker].pop())
                made_progress = True
                if len(chosen) == n:
                    break
        if not made_progress:
            break

    if len(chosen) < n:
        raise RuntimeError(
            f"Only found {len(chosen)} usable validated clips; requested {n}."
        )

    return df.loc[chosen].sample(frac=1, random_state=seed).reset_index(drop=True)


def group_split(df: pd.DataFrame, seed: int) -> pd.Series:
    """70/15/15 split with speakers kept entirely inside one split."""
    groups = df["client_id"].fillna("UNKNOWN_SPEAKER").astype(str)

    first = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(first.split(df, groups=groups))

    train = df.iloc[train_idx]
    temp = df.iloc[temp_idx].copy()
    temp_groups = temp["client_id"].fillna("UNKNOWN_SPEAKER").astype(str)

    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
    val_local, test_local = next(second.split(temp, groups=temp_groups))

    split = pd.Series(index=df.index, dtype="object")
    split.iloc[train_idx] = "train"
    split.iloc[temp_idx[val_local]] = "validation"
    split.iloc[temp_idx[test_local]] = "test"
    return split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--metadata", required=True, type=Path)
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    validated_tsv = find_one(args.input, "validated.tsv")
    cv_root = validated_tsv.parent
    clips_dir = cv_root / "clips"
    if not clips_dir.exists():
        clips_matches = [p for p in args.input.rglob("clips") if p.is_dir()]
        if not clips_matches:
            raise FileNotFoundError("Could not find Common Voice clips/ directory.")
        clips_dir = clips_matches[0]

    df = pd.read_csv(validated_tsv, sep="\t", low_memory=False)

    required = {"path", "client_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"validated.tsv is missing required columns: {sorted(missing)}")

    df["source_audio"] = df["path"].map(lambda x: clips_dir / str(x))
    df = df[df["source_audio"].map(Path.exists)].copy()

    if len(df) < args.n:
        raise RuntimeError(
            f"Only {len(df)} validated audio files exist on disk; requested {args.n}."
        )

    selected = balanced_speaker_sample(df, args.n, args.seed)
    selected["split"] = group_split(selected, args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in selected.iterrows():
        src = Path(row["source_audio"])
        sample_id = f"cv_hi_{i:05d}"
        dst = args.output / f"{sample_id}{src.suffix.lower()}"
        shutil.copy2(src, dst)

        rows.append({
            "sample_id": sample_id,
            "file_path": str(dst),
            "language": "Hindi",
            "language_type": "Hindi",
            "label": "real",
            "speaker_id": str(row.get("client_id", "")),
            "gender": row.get("gender", ""),
            "source_dataset": "CommonVoice_Hindi_26.0",
            "source_split": "validated",
            "source_file_id": str(row.get("path", "")),
            "generator": "",
            "generator_family": "",
            "original_sample_rate": "",
            "duration": "",
            "fake_start": "",
            "fake_end": "",
            "has_timestamp_labels": False,
            "codec": "mp3",
            "recording_condition": "crowdsourced_read_speech",
            "license": "CC0-1.0",
            "split": row["split"],
            "transcript": row.get("sentence", ""),
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.metadata, index=False)

    print("\nCreated Common Voice Hindi V1 subset")
    print("-----------------------------------")
    print(f"Source validated rows available: {len(df):,}")
    print(f"Selected clips: {len(out):,}")
    print(f"Unique speakers selected: {out['speaker_id'].nunique():,}")
    print("\nSplit counts:")
    print(out["split"].value_counts())
    print("\nMetadata:")
    print(args.metadata)
    print("\nSelected source audio:")
    print(args.output)


if __name__ == "__main__":
    main()
