#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import Audio, load_dataset
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream and select a balanced Hindi Kathbath subset")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--n", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "ai4bharat/Kathbath",
        "hindi",
        split="train",
        streaming=True,
    )
    dataset = dataset.cast_column("audio_filepath", Audio(decode=False))

    selected_rows: list[dict] = []
    speaker_counts: Counter[str] = Counter()
    selected_speakers: set[str] = set()

    for sample in tqdm(dataset, total=None):
        speaker_id = str(sample.get("speaker_id") or "UNKNOWN")
        speaker_counts[speaker_id] += 1

        if len(selected_rows) >= args.n:
            break

        current_count = speaker_counts[speaker_id]
        if speaker_id not in selected_speakers:
            should_select = True
        else:
            target_per_speaker = max(1, math.ceil(args.n / max(1, len(selected_speakers))))
            should_select = current_count <= target_per_speaker

        if should_select:
            audio_payload = sample.get("audio_filepath")
            if not isinstance(audio_payload, dict):
                continue

            audio_bytes = audio_payload.get("bytes")
            if audio_bytes is None:
                continue

            sample_id = f"kb_hi_{len(selected_rows):05d}"
            dst = args.output / f"{sample_id}.wav"
            dst.write_bytes(audio_bytes)

            selected_rows.append(
                {
                    "sample_id": sample_id,
                    "file_path": str(dst),
                    "language": "Hindi",
                    "language_type": "Hindi",
                    "label": "real",
                    "speaker_id": speaker_id,
                    "gender": sample.get("gender", ""),
                    "source_dataset": "Kathbath",
                    "source_split": "train",
                    "source_file_id": str(sample.get("fname", "")),
                    "generator": "",
                    "generator_family": "",
                    "duration": sample.get("duration", ""),
                    "fake_start": "",
                    "fake_end": "",
                    "has_timestamp_labels": False,
                    "codec": "wav",
                    "recording_condition": "read_speech",
                    "license": "unknown",
                    "split": "train",
                    "transcript": sample.get("text", ""),
                }
            )
            selected_speakers.add(speaker_id)

    if not selected_rows:
        raise RuntimeError("No samples were selected from Kathbath Hindi.")

    out = pd.DataFrame(selected_rows)
    out.to_csv(args.metadata, index=False)

    print("\nCreated Kathbath Hindi V1 subset")
    print("-----------------------------------")
    print(f"Selected clips: {len(out):,}")
    print(f"Unique speakers selected: {out['speaker_id'].nunique():,}")
    print("\nMetadata:")
    print(args.metadata)
    print("\nSelected audio:")
    print(args.output)


if __name__ == "__main__":
    main()
