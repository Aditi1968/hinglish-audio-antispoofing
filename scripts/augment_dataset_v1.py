"""
Phase 3 - controlled codec / channel augmentation.

Purpose: the canonical stage made every file 16 kHz mono PCM16, which removes
the trivial format shortcut but NOT the historical channel artifacts (Common
Voice and Azure were MP3, Kathbath was FLAC/PCM_24). This stage builds a
class-balanced robustness view so codec/channel conditions stop correlating
with the label.

Two independent products:

  1. TRAINING VIEW      one derived file per TRAINING parent (~13,963).
                        Validation/test are NOT augmented for training.
                        -> data/augmented/v1/
                        -> metadata/dataset_v1_augmented.csv

  2. ROBUSTNESS EVAL    a controlled transformed copy of the internal TEST
                        set: the same 100 parents per (language_type, label)
                        rendered under 5 conditions = 2,000 files.
                        -> data/evaluation/robustness_v1/
                        -> metadata/evaluation/robustness_v1.csv

LEAKAGE RULES enforced here:
  - a derived file ALWAYS inherits its parent's split, verbatim
  - the training manifest may only contain parents whose split == train
  - the robustness manifest may only contain parents whose split == test
  - group_id travels with the parent so downstream grouping stays valid

BALANCE RULE: conditions are dealt round-robin inside every
(source_dataset x label x language_type) stratum with seed 42, so no
condition can correlate with label, language, source, generator or voice.

Canonical files are never modified. Originals are never touched.
"""

from collections import Counter, defaultdict
from pathlib import Path

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import time

import pandas as pd
import soundfile as sf


MASTER = Path("metadata/dataset_v1_master_v2.csv")

AUGMENTED_DIR = Path("data/augmented/v1")
AUGMENTED_META = Path("metadata/dataset_v1_augmented.csv")

ROBUSTNESS_DIR = Path("data/evaluation/robustness_v1")
ROBUSTNESS_META = Path("metadata/evaluation/robustness_v1.csv")

TARGET_RATE = 16000
SEED = 42
WORKERS = 2
CHUNK = 256 * 1024

# Optional hooks. NOT downloaded and NOT implemented in V1 - if you later
# point these at real corpora, add the corresponding conditions explicitly.
NOISE_DIR = Path(os.environ.get("NOISE_DIR", "data/aug_sources/noise"))
RIR_DIR = Path(os.environ.get("RIR_DIR", "data/aug_sources/rir"))

# Physical replay (play through a speaker, re-record with a mic) CANNOT be
# synthesised honestly in software. It is deliberately absent.

# Conditions dealt to training parents. canonical_clean is excluded on
# purpose: the canonical 16 kHz file already exists for every parent, so it
# needs no derived copy.
TRAIN_CONDITIONS = [
    "mp3_32",
    "mp3_64",
    "mp3_128",
    "opus_12",
    "opus_24",
    "opus_48",
    "telephone",
    "lowpass",
    "gain",
    "mild_clipping",
    "multi_codec",
]

LOWPASS_CUTOFFS = [4000, 5000, 6000, 7000]
GAIN_DB = [-9, -6, -3, 3]
MULTI_CODEC_CHAINS = ["mp3_64__opus_24", "opus_24__mp3_64"]

# Robustness evaluation uses one representative parameterisation per family.
ROBUSTNESS_CONDITIONS = [
    ("mp3", {"codec": "mp3", "bitrate_kbps": 64}),
    ("opus", {"codec": "opus", "bitrate_kbps": 24}),
    ("telephone", {"band_hz": "300-3400", "intermediate_rate": 8000}),
    ("lowpass", {"cutoff_hz": 4000}),
    ("multi_codec", {"chain": "mp3_64__opus_24"}),
]

ROBUSTNESS_PER_GROUP = 100


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def require_ffmpeg():

    if not shutil.which("ffmpeg"):
        raise SystemExit(
            "ffmpeg not found on PATH.\n"
            "Install it and re-run:\n"
            "  Windows : winget install Gyan.FFmpeg\n"
            "  macOS   : brew install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg\n"
            "No fallback codec will be used: a mixed encoder path would "
            "itself become a source-correlated artifact."
        )

    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True, text=True,
    ).stdout

    missing = [
        name for name in ("libmp3lame", "libopus")
        if name not in encoders
    ]

    if missing:
        raise SystemExit(
            f"ffmpeg is present but lacks required encoders: {missing}.\n"
            "Install a full build (e.g. Gyan.FFmpeg on Windows, or "
            "ffmpeg with --enable-libmp3lame --enable-libopus).\n"
            "This script will NOT substitute a different codec."
        )


def run_ffmpeg(args):
    """Run one ffmpeg command, retrying transient Windows spawn failures."""

    command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
               "-y"] + args

    last_error = None

    for attempt in range(4):

        try:
            result = subprocess.run(command, capture_output=True, text=True)
        except OSError as e:
            last_error = f"spawn failed: {e}"
            time.sleep(2 * (attempt + 1))
            continue

        if result.returncode == 0:
            return None

        last_error = result.stderr.strip()[:200]
        time.sleep(1 + attempt)

    return last_error


def to_canonical_wav(source, destination, filters=None):
    """Decode anything back to 16 kHz mono PCM16 WAV."""

    args = ["-i", str(source), "-map_metadata", "-1"]

    if filters:
        args += ["-af", filters]

    args += ["-ac", "1", "-ar", str(TARGET_RATE), "-c:a", "pcm_s16le",
             "-f", "wav", str(destination)]

    return run_ffmpeg(args)


def sha256_of(path):

    for attempt in range(5):
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(CHUNK), b""):
                    digest.update(block)
            return digest.hexdigest()
        except MemoryError:
            time.sleep(2 * (attempt + 1))

    raise MemoryError(f"could not hash {path}")


# ---------------------------------------------------------
# CONDITION IMPLEMENTATIONS
# ---------------------------------------------------------

def apply_condition(condition, parameters, source, destination, workdir):
    """
    Every condition ends at 16 kHz mono PCM16 WAV.

    Codec conditions do a REAL encode/decode round trip through a temporary
    compressed file, so the artefacts are genuine rather than simulated.
    """

    if condition.startswith("mp3"):

        bitrate = parameters["bitrate_kbps"]
        temp = workdir / "tmp.mp3"

        error = run_ffmpeg([
            "-i", str(source), "-ac", "1", "-ar", str(TARGET_RATE),
            "-c:a", "libmp3lame", "-b:a", f"{bitrate}k",
            "-f", "mp3", str(temp),
        ])
        if error:
            return error

        return to_canonical_wav(temp, destination)

    if condition.startswith("opus"):

        bitrate = parameters["bitrate_kbps"]
        temp = workdir / "tmp.opus"

        # Opus always runs at 48 kHz internally; decoding returns to 16 kHz.
        error = run_ffmpeg([
            "-i", str(source), "-ac", "1",
            "-c:a", "libopus", "-b:a", f"{bitrate}k",
            "-f", "ogg", str(temp),
        ])
        if error:
            return error

        return to_canonical_wav(temp, destination)

    if condition == "telephone":

        # Genuine narrowband: band-limit, pass through an 8 kHz
        # representation, then return to 16 kHz.
        temp = workdir / "tmp_narrow.wav"

        error = run_ffmpeg([
            "-i", str(source),
            "-af", "highpass=f=300,lowpass=f=3400",
            "-ac", "1", "-ar", "8000", "-c:a", "pcm_s16le",
            "-f", "wav", str(temp),
        ])
        if error:
            return error

        return to_canonical_wav(temp, destination)

    if condition == "lowpass":
        return to_canonical_wav(
            source, destination, f"lowpass=f={parameters['cutoff_hz']}"
        )

    if condition == "gain":
        return to_canonical_wav(
            source, destination, f"volume={parameters['gain_db']}dB"
        )

    if condition == "mild_clipping":

        # Boost into PCM16 saturation, then restore the level. The clipping
        # is baked in by the intermediate s16 write; the result stays
        # intelligible.
        boost = parameters["boost_db"]
        temp = workdir / "tmp_clip.wav"

        error = to_canonical_wav(source, temp, f"volume={boost}dB")
        if error:
            return error

        return to_canonical_wav(destination=destination, source=temp,
                                filters=f"volume=-{boost}dB")

    if condition == "multi_codec":

        chain = parameters["chain"]
        first, second = chain.split("__")

        temp_a = workdir / "chain_a.wav"
        temp_b = workdir / "chain_b.wav"

        for step, (name, output) in enumerate(
            [(first, temp_a), (second, temp_b)]
        ):
            family, bitrate = name.split("_")
            step_source = source if step == 0 else temp_a

            error = apply_condition(
                f"{family}_{bitrate}",
                {"bitrate_kbps": int(bitrate)},
                step_source,
                output,
                workdir,
            )
            if error:
                return error

        shutil.move(str(temp_b), str(destination))
        return None

    return f"unknown condition {condition}"


def parameters_for(condition, index):
    """Deterministic parameters - index is the parent's position in its deal."""

    if condition.startswith("mp3"):
        return {"codec": "mp3", "bitrate_kbps": int(condition.split("_")[1])}

    if condition.startswith("opus"):
        return {"codec": "opus", "bitrate_kbps": int(condition.split("_")[1])}

    if condition == "telephone":
        return {"band_hz": "300-3400", "intermediate_rate": 8000}

    if condition == "lowpass":
        return {"cutoff_hz": LOWPASS_CUTOFFS[index % len(LOWPASS_CUTOFFS)]}

    if condition == "gain":
        return {"gain_db": GAIN_DB[index % len(GAIN_DB)]}

    if condition == "mild_clipping":
        return {"boost_db": 6}

    if condition == "multi_codec":
        return {"chain": MULTI_CODEC_CHAINS[index % len(MULTI_CODEC_CHAINS)]}

    return {}


# ---------------------------------------------------------
# ASSIGNMENT
# ---------------------------------------------------------

def assign_train_conditions(train):
    """
    Deal conditions round-robin inside every
    (source_dataset x label x language_type) stratum.

    Because the deal is per stratum, each source sees the same condition
    proportions, so condition cannot correlate with label, language, source,
    generator or voice.
    """

    rng = random.Random(SEED)
    assignment = {}

    strata = train.groupby(["source_dataset", "label", "language_type"])

    for stratum, part in strata:

        ids = sorted(part["sample_id"])
        rng.shuffle(ids)

        for position, sample_id in enumerate(ids):
            condition = TRAIN_CONDITIONS[position % len(TRAIN_CONDITIONS)]
            assignment[sample_id] = (condition, position)

    return assignment


def select_robustness_parents(test):
    """Same parents across every condition, so conditions are comparable."""

    rng = random.Random(SEED + 1)
    chosen = []

    for (language_type, label), part in test.groupby(
        ["language_type", "label"]
    ):
        ids = sorted(part["sample_id"])
        rng.shuffle(ids)

        take = ids[:ROBUSTNESS_PER_GROUP]

        if len(take) < ROBUSTNESS_PER_GROUP:
            print(f"    WARNING: only {len(take)} test parents available for "
                  f"{language_type}/{label}")

        chosen.extend(take)

    return chosen


# ---------------------------------------------------------
# RENDERING
# ---------------------------------------------------------

def render(job):

    (kind, row, condition, parameters, destination) = job

    if destination.exists() and destination.stat().st_size > 0:
        try:
            info = sf.info(str(destination))
            return kind, row, condition, parameters, destination, info, None
        except Exception:
            destination.unlink(missing_ok=True)

    source = Path(row["processed_file_path"])

    if not source.exists():
        return kind, row, condition, parameters, destination, None, (
            f"missing canonical parent {source}"
        )

    with tempfile.TemporaryDirectory() as tmp:

        error = apply_condition(
            condition, parameters, source, destination, Path(tmp)
        )

    if error:
        return kind, row, condition, parameters, destination, None, error

    try:
        info = sf.info(str(destination))
    except Exception as e:
        return kind, row, condition, parameters, destination, None, str(e)

    return kind, row, condition, parameters, destination, info, None


def build_rows(results, id_prefix):

    rows = []
    errors = []

    for kind, row, condition, parameters, destination, info, error in results:

        if error:
            errors.append({
                "parent_sample_id": row["sample_id"],
                "condition": condition,
                "error": error,
            })
            continue

        rows.append({
            f"{id_prefix}_id": destination.stem,
            "parent_sample_id": row["sample_id"],
            "parent_processed_path": row["processed_file_path"],
            "augmented_file_path": str(destination),

            "language": row["language"],
            "language_type": row["language_type"],
            "label": row["label"],
            "source_dataset": row["source_dataset"],
            "split": row["split"],
            "parent_split": row["split"],

            "group_id": row["group_id"],
            "generator": row.get("generator", ""),
            "voice": row.get("voice", ""),
            "gender_normalized": row.get("gender_normalized", "unknown"),

            "augmentation_condition": condition,
            "augmentation_parameters": json.dumps(
                parameters, sort_keys=True
            ),

            "sample_rate": info.samplerate,
            "channels": info.channels,
            "codec": f"{info.format}/{info.subtype}",
            "duration": round(info.duration, 4),
            "parent_duration": row.get("processed_duration"),
            "bytes": destination.stat().st_size,
            "sha256": sha256_of(destination),

            "license": row.get("license", ""),
            "license_status": row.get("license_status", ""),
            "redistribution_allowed": (
                "false"
                if str(row.get("license_status", "")).upper() == "UNVERIFIED"
                else "true"
            ),
        })

    return rows, errors


def run_jobs(jobs, label):

    print(f"\nrendering {len(jobs):,} {label} files...\n")

    results = []
    done = 0

    with ThreadPoolExecutorCompat(WORKERS) as pool:

        for result in pool.map(render, jobs):

            results.append(result)
            done += 1

            if done % 1000 == 0:
                print(f"    {done:,}/{len(jobs):,}")

    return results


class ThreadPoolExecutorCompat:
    """Small wrapper so the import stays local and workers stay bounded."""

    def __init__(self, workers):
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=workers)

    def __enter__(self):
        return self._pool

    def __exit__(self, *exc):
        self._pool.shutdown(wait=True)
        return False


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    args = parser.parse_args()

    require_ffmpeg()

    for hook, name in [(NOISE_DIR, "NOISE_DIR"), (RIR_DIR, "RIR_DIR")]:
        state = "present" if hook.exists() else "absent (hook only)"
        print(f"{name:<10} {hook}  -> {state}")
    print("physical replay: NOT implemented - cannot be synthesised honestly\n")

    df = pd.read_csv(MASTER, dtype=str, keep_default_na=False)

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    print(f"parents: train {len(train):,} | test {len(test):,}")

    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    ROBUSTNESS_META.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- training view ----------------

    if not args.skip_train:

        assignment = assign_train_conditions(train)

        jobs = []

        for row in train.to_dict("records"):

            condition, position = assignment[row["sample_id"]]
            parameters = parameters_for(condition, position)

            destination = (
                AUGMENTED_DIR / f"aug_{row['sample_id']}__{condition}.wav"
            )

            jobs.append(("train", row, condition, parameters, destination))

        results = run_jobs(jobs, "training augmentation")
        rows, errors = build_rows(results, "augmentation")

        augmented = pd.DataFrame(rows)
        augmented.to_csv(AUGMENTED_META, index=False)

        print(f"\ntraining augmentation: {len(augmented):,} files, "
              f"{len(errors)} errors")

        if errors:
            for e in errors[:10]:
                print("    ", e)

    # ---------------- robustness evaluation ----------------

    if not args.skip_robustness:

        parents = set(select_robustness_parents(test))
        subset = test[test["sample_id"].isin(parents)]

        jobs = []

        for row in subset.to_dict("records"):
            for condition, parameters in ROBUSTNESS_CONDITIONS:
                destination = (
                    ROBUSTNESS_DIR / f"rob_{row['sample_id']}__{condition}.wav"
                )
                jobs.append(("test", row, condition, parameters, destination))

        results = run_jobs(jobs, "robustness evaluation")
        rows, errors = build_rows(results, "robustness")

        robustness = pd.DataFrame(rows)
        robustness.to_csv(ROBUSTNESS_META, index=False)

        print(f"\nrobustness set: {len(robustness):,} files, "
              f"{len(errors)} errors")

        if errors:
            for e in errors[:10]:
                print("    ", e)

    print("\nPHASE 3 rendering complete.")
    print(f"    {AUGMENTED_META}")
    print(f"    {ROBUSTNESS_META}")
    print("\nCanonical and original audio were not modified.")


if __name__ == "__main__":
    main()
