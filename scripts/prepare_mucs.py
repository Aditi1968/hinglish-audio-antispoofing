"""
MUCS / OpenSLR-104 Hindi-English -> 5,000 REAL Hinglish samples.

Structure discovered on the live archive (Kaldi layout, NOT assumed):

    train/*.wav                  521 long spoken-tutorial recordings
    train/transcripts/wav.scp    recording_id  recording.wav
    train/transcripts/segments   utt_id  recording_id  start_sec  end_sec
    train/transcripts/text       utt_id  transcript
    train/transcripts/utt2spk    utt_id  speaker_id
    train/transcripts/spkr_list  523 speaker ids (520 actually have segments)

    52,825 segments | 521 recordings | 520 speakers | 89.55 h
    speakers per recording: max 1    (recording never shared across speakers)
    recordings per speaker:  max 2    (only one speaker has two)

Utterances do NOT exist as files - they must be cut from the long recordings
using the segment timestamps.

Cutting is a lossless slice: identical sample rate, identical subtype, no
resampling, denoising, normalization, trimming or augmentation. The original
full recordings stay untouched under data/raw/mucs/extracted/.

ONLY Hindi-English is used. The Bengali-English archive is never downloaded
or referenced anywhere in this project.
"""

from collections import Counter, defaultdict
from pathlib import Path

import json
import random

import pandas as pd
import soundfile as sf


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

SOURCE_ROOT = Path("data/raw/mucs/extracted/train")
TRANSCRIPTS = SOURCE_ROOT / "transcripts"

OUTPUT_DIR = Path("data/selected/mucs_hinglish")
METADATA_PATH = Path("metadata/mucs_hinglish.csv")
SUMMARY_PATH = Path("metadata/mucs_hinglish_summary.json")

TARGET = 5000

# Documented selection criterion (not preprocessing): skip degenerate
# segments. Only 63 of 52,825 fall outside this window.
MIN_DURATION = 1.0
MAX_DURATION = 30.0

SPLIT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

LICENSE = "CC BY-SA 4.0 (OpenSLR SLR104 / MUCS 2021)"

SEED = 42


# ---------------------------------------------------------
# LOAD KALDI METADATA
# ---------------------------------------------------------

def read_two_column(path):

    mapping = {}

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]

    return mapping


def load_metadata():

    segments = {}

    with open(TRANSCRIPTS / "segments", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 4:
                segments[parts[0]] = (
                    parts[1], float(parts[2]), float(parts[3])
                )

    text = read_two_column(TRANSCRIPTS / "text")
    utt2spk = read_two_column(TRANSCRIPTS / "utt2spk")
    wav_scp = read_two_column(TRANSCRIPTS / "wav.scp")

    print(f"segments   {len(segments):,}")
    print(f"text       {len(text):,}")
    print(f"utt2spk    {len(utt2spk):,}")
    print(f"wav.scp    {len(wav_scp):,}")

    rows = []

    for utt_id, (recording, start, end) in segments.items():

        duration = round(end - start, 3)

        rows.append({
            "utt_id": utt_id,
            "recording_id": recording,
            "speaker_id": utt2spk.get(utt_id),
            "start_time": start,
            "end_time": end,
            "duration": duration,
            "transcript": text.get(utt_id),
            "wav_name": wav_scp.get(recording),
        })

    frame = pd.DataFrame(rows)

    print(f"\nsegments with no speaker:    {int(frame['speaker_id'].isna().sum())}")
    print(f"segments with no transcript: {int(frame['transcript'].isna().sum())}")
    print(f"segments with no wav entry:  {int(frame['wav_name'].isna().sum())}")

    return frame


# ---------------------------------------------------------
# SELECTION
# ---------------------------------------------------------

def select(frame):
    """
    Round-robin across speakers so the 5,000 clips span as many voices as
    possible, rather than taking the first 5,000 segments (which would come
    from a handful of recordings).
    """

    usable = frame[
        frame["speaker_id"].notna()
        & frame["transcript"].notna()
        & frame["wav_name"].notna()
        & frame["duration"].between(MIN_DURATION, MAX_DURATION)
    ].copy()

    dropped = len(frame) - len(usable)

    print(f"\nusable segments: {len(usable):,} "
          f"({dropped:,} dropped: missing fields or "
          f"duration outside {MIN_DURATION}-{MAX_DURATION}s)")

    rng = random.Random(SEED)

    by_speaker = defaultdict(list)

    for record in usable.to_dict("records"):
        by_speaker[record["speaker_id"]].append(record)

    # Shuffle within speaker so we sample across the whole tutorial rather
    # than only its opening minutes.
    for speaker in by_speaker:
        rng.shuffle(by_speaker[speaker])

    speakers = sorted(by_speaker)
    rng.shuffle(speakers)

    print(f"speakers available: {len(speakers)}")

    picked = []
    exhausted = set()

    while len(picked) < TARGET and len(exhausted) < len(speakers):

        for speaker in speakers:

            if len(picked) >= TARGET:
                break

            pool = by_speaker[speaker]

            if not pool:
                exhausted.add(speaker)
                continue

            picked.append(pool.pop())

    per_speaker = Counter(r["speaker_id"] for r in picked)

    print(f"selected {len(picked):,} clips across "
          f"{len(per_speaker)} speakers "
          f"(max {max(per_speaker.values())} per speaker)")

    return picked


# ---------------------------------------------------------
# SPLITTING
# ---------------------------------------------------------

def assign_splits(df):
    """
    Speaker-level assignment. Because no recording is shared between
    speakers, speaker-disjoint splits are automatically recording-disjoint.
    """

    counts = df.groupby("speaker_id").size().to_dict()

    targets = {
        split: TARGET * frac
        for split, frac in SPLIT_FRACTIONS.items()
    }

    current = Counter()
    assignment = {}

    # Biggest speaker groups first, each into whichever split is furthest
    # below its target.
    for speaker in sorted(counts, key=lambda s: -counts[s]):

        split = max(
            SPLIT_FRACTIONS,
            key=lambda s: targets[s] - current[s],
        )

        assignment[speaker] = split
        current[split] += counts[speaker]

    return df["speaker_id"].map(assignment)


# ---------------------------------------------------------
# CUTTING
# ---------------------------------------------------------

def cut_clips(picked):
    """
    Slice each segment out of its source recording.

    Lossless: same sample rate, same subtype, no processing of any kind.
    Recordings are opened once and reused across their segments.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Group by recording so each large wav is read once.
    by_recording = defaultdict(list)

    for index, record in enumerate(picked):
        by_recording[record["recording_id"]].append((index, record))

    rows = [None] * len(picked)
    done = 0

    for recording, items in sorted(by_recording.items()):

        source = SOURCE_ROOT / items[0][1]["wav_name"]

        if not source.exists():
            print(f"    MISSING source recording: {source}")
            continue

        info = sf.info(str(source))
        sample_rate = info.samplerate
        subtype = info.subtype

        with sf.SoundFile(str(source)) as handle:

            for index, record in items:

                start_frame = int(round(record["start_time"] * sample_rate))
                end_frame = int(round(record["end_time"] * sample_rate))
                end_frame = min(end_frame, len(handle))

                if end_frame <= start_frame:
                    print(f"    empty span for {record['utt_id']}")
                    continue

                handle.seek(start_frame)
                audio = handle.read(end_frame - start_frame, dtype="int16")

                sample_id = f"mucs_hi_en_{index:05d}"
                destination = OUTPUT_DIR / f"{sample_id}.wav"

                sf.write(
                    str(destination),
                    audio,
                    sample_rate,
                    subtype=subtype,
                )

                actual = round(len(audio) / sample_rate, 3)

                rows[index] = {
                    "sample_id": sample_id,
                    "file_path": str(destination),
                    "language": "Hindi-English",
                    "language_type": "Hinglish",
                    "label": "real",
                    "source_dataset": "MUCS_OpenSLR104",
                    "speaker_id": record["speaker_id"],
                    "session_id": record["recording_id"],
                    "source_file_id": record["wav_name"],
                    "utterance_id": record["utt_id"],
                    "transcript": record["transcript"],
                    "start_time": record["start_time"],
                    "end_time": record["end_time"],
                    "duration": actual,
                    "original_sample_rate": sample_rate,
                    "codec": f"wav/{subtype}",
                    "channels": info.channels,
                    "split": "",
                    "license": LICENSE,
                }

                done += 1

        if done % 500 < len(items):
            print(f"    cut {done:,}/{len(picked):,}")

    return pd.DataFrame([r for r in rows if r is not None])


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate(df):

    print("\n" + "=" * 55)
    print("MUCS HINGLISH - VALIDATION")
    print("=" * 55)

    failures = []

    def check(condition, message):
        print(f"    {'OK  ' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    print("\nCOUNTS")
    check(len(df) == TARGET, f"total = {len(df)} (expected {TARGET})")
    check(
        bool((df["label"] == "real").all()),
        f"all label = real ({int((df['label'] == 'real').sum())}/{len(df)})"
    )
    check(
        bool((df["language_type"] == "Hinglish").all()),
        f"all language_type = Hinglish "
        f"({int((df['language_type'] == 'Hinglish').sum())}/{len(df)})"
    )

    print("\nFILES")

    paths = df["file_path"].map(Path)
    exists = paths.map(lambda p: p.exists())
    sizes = paths.map(lambda p: p.stat().st_size if p.exists() else 0)

    check(int(exists.sum()) == TARGET, f"audio files exist = {int(exists.sum())}")
    check(int((sizes == 0).sum()) == 0,
          f"zero-byte files = {int((sizes == 0).sum())}")
    check(int(df['sample_id'].duplicated().sum()) == 0,
          f"duplicate sample IDs = {int(df['sample_id'].duplicated().sum())}")
    check(int(df['file_path'].duplicated().sum()) == 0,
          f"duplicate paths = {int(df['file_path'].duplicated().sum())}")

    print("\nGROUPS")
    print(f"    unique speakers        {df['speaker_id'].nunique()}")
    print(f"    unique sessions        {df['session_id'].nunique()}")
    print(f"    unique source records  {df['source_file_id'].nunique()}")

    print("\nSPLIT COUNTS")
    print(df["split"].value_counts().to_string())

    print("\n    speakers per split:")
    for split in ["train", "validation", "test"]:
        part = df[df["split"] == split]
        print(f"        {split:<11} {len(part):>5} clips | "
              f"{part['speaker_id'].nunique():>4} speakers | "
              f"{part['session_id'].nunique():>4} sessions")

    print("\nLEAKAGE")

    for key in ["speaker_id", "session_id", "source_file_id"]:

        groups = {
            split: set(df[df["split"] == split][key])
            for split in ["train", "validation", "test"]
        }

        for a, b in [
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ]:
            overlap = len(groups[a] & groups[b])
            check(overlap == 0, f"{key}: {a} & {b} overlap = {overlap}")

    print("\nDURATION STATS")
    d = df["duration"]
    print(f"    total hours  {d.sum() / 3600:.2f}")
    print(f"    min          {d.min():.2f}")
    print(f"    median       {d.median():.2f}")
    print(f"    mean         {d.mean():.2f}")
    print(f"    max          {d.max():.2f}")

    print("\n    clips per speaker:")
    per_speaker = df["speaker_id"].value_counts()
    print(f"        min {per_speaker.min()} | median {per_speaker.median()} "
          f"| mean {round(per_speaker.mean(), 2)} | max {per_speaker.max()}")

    print("\nAUDIO FORMAT")
    print("    original_sample_rate:")
    print(df["original_sample_rate"].value_counts().to_string())
    print("\n    codec:")
    print(df["codec"].value_counts().to_string())
    print("\n    channels:")
    print(df["channels"].value_counts().to_string())
    print(f"\n    bytes on disk: {sizes.sum() / 1e9:.2f} GB")

    print()

    if failures:
        print("VALIDATION FAILED:")
        for f in failures:
            print("    -", f)
    else:
        print("ALL VALIDATIONS PASSED")

    return not failures


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    if not TRANSCRIPTS.exists():
        raise SystemExit(
            f"Missing {TRANSCRIPTS}. Run scripts/download_mucs.py train and "
            "extract the archive first."
        )

    print("Loading MUCS Hindi-English train metadata...\n")
    frame = load_metadata()

    picked = select(frame)

    print("\nCutting segments from source recordings (lossless slices)...\n")
    df = cut_clips(picked)

    print("\nAssigning speaker-disjoint splits...")
    df["split"] = assign_splits(df)

    df = df.sort_values("sample_id").reset_index(drop=True)
    df.to_csv(METADATA_PATH, index=False)

    passed = validate(df)

    summary = {
        "dataset": "MUCS / OpenSLR-104 Hindi-English (real Hinglish)",
        "source": "https://www.openslr.org/104/ Hindi-English_train.tar.gz",
        "language": "Hindi-English",
        "language_type": "Hinglish",
        "label": "real",
        "total_samples": int(len(df)),
        "unique_speakers": int(df["speaker_id"].nunique()),
        "unique_sessions": int(df["session_id"].nunique()),
        "split_counts": {
            str(k): int(v) for k, v in df["split"].value_counts().items()
        },
        "speakers_per_split": {
            split: int(df[df["split"] == split]["speaker_id"].nunique())
            for split in ["train", "validation", "test"]
        },
        "duration_hours": round(float(df["duration"].sum() / 3600), 2),
        "audio_state": (
            "lossless time slices of the original recordings; "
            "not resampled, denoised, normalized, trimmed or augmented"
        ),
        "license": LICENSE,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nMetadata:", METADATA_PATH)
    print("Summary: ", SUMMARY_PATH)
    print("Audio:   ", OUTPUT_DIR)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
