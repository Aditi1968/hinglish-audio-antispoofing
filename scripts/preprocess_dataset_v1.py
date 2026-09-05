"""
Phase 2 - canonical audio standardization.

Every one of the 20,000 development files goes through the SAME pipeline:

    original encoded file -> decode -> mono -> resample 16 kHz -> PCM16 WAV

Nothing else happens. No denoising, no dereverb, no silence trimming, no VAD,
no speech enhancement, no per-source normalization, no augmentation, and no
intentional duration change. Full utterances are preserved.

Originals under data/selected/ and data/raw/ are never written to.

IMPORTANT - what this does and does not fix:

    It removes the TRIVIAL shortcut: after this stage every file is
    16 kHz mono PCM16 WAV, so file format and sample rate no longer encode
    the label.

    It does NOT remove historical channel/compression artifacts. Common Voice
    was MP3 and Kathbath was FLAC/PCM_24; MP3 quantization noise and the
    original capture channel survive decoding and resampling. Those remain a
    confound and are handled later by class-balanced augmentation.

Reads  metadata/dataset_v1_master_v2.csv
Writes data/processed/v1_16k/<sample_id>.wav
       metadata/dataset_v1_master_v2.csv   (adds processed_* columns)
       metadata/dataset_v1_preprocessing_audit.json
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hashlib
import json
import shutil
import subprocess
import time

import pandas as pd
import soundfile as sf


MASTER = Path("metadata/dataset_v1_master_v2.csv")
PROCESSED_DIR = Path("data/processed/v1_16k")
AUDIT_PATH = Path("metadata/dataset_v1_preprocessing_audit.json")

TARGET_RATE = 16000
TARGET_CHANNELS = 1

# Decoder/resampler tolerance. MP3 in particular carries encoder delay and
# padding, so a few tens of milliseconds of drift is expected and benign.
DURATION_TOLERANCE = 0.05

WORKERS = 2
CHUNK = 256 * 1024


def require_ffmpeg():

    binary = shutil.which("ffmpeg")

    if not binary:
        raise SystemExit(
            "ffmpeg not found on PATH.\n"
            "Install it and re-run:\n"
            "  Windows : winget install Gyan.FFmpeg\n"
            "  macOS   : brew install ffmpeg\n"
            "  Linux   : sudo apt install ffmpeg\n"
            "This pipeline will NOT fall back to another decoder, because a "
            "mixed decoding path would itself become a source-correlated "
            "artifact."
        )

    return binary


def sha256_of(path):

    # Retried: this box runs close to its memory ceiling and a plain read
    # can raise MemoryError under transient pressure.
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


def convert(job):

    sample_id, source_path = job

    destination = PROCESSED_DIR / f"{sample_id}.wav"

    if not (destination.exists() and destination.stat().st_size > 0):

        command = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-y",
            "-i", str(source_path),
            "-map_metadata", "-1",       # drop source tags
            "-ac", str(TARGET_CHANNELS),  # mono
            "-ar", str(TARGET_RATE),      # 16 kHz
            "-c:a", "pcm_s16le",          # PCM16
            "-f", "wav",
            str(destination),
        ]

        # Windows exhausts its paging file if too many ffmpeg processes are
        # spawned at once; retry rather than losing the sample.
        last_error = None

        for attempt in range(4):
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True
                )
            except OSError as e:
                last_error = f"spawn failed: {e}"
                time.sleep(2 * (attempt + 1))
                continue

            if result.returncode == 0:
                last_error = None
                break

            last_error = f"ffmpeg failed: {result.stderr.strip()[:160]}"
            time.sleep(1 + attempt)

        if last_error:
            return sample_id, None, last_error

    try:
        info = sf.info(str(destination))
    except Exception as e:
        return sample_id, None, f"unreadable output: {e}"

    return sample_id, {
        "processed_file_path": str(destination),
        "processed_sample_rate": info.samplerate,
        "processed_channels": info.channels,
        "processed_codec": f"{info.format}/{info.subtype}",
        "processed_duration": round(info.duration, 4),
        "processed_bytes": destination.stat().st_size,
        "processed_sha256": sha256_of(destination),
    }, None


def main():

    require_ffmpeg()

    df = pd.read_csv(MASTER, dtype=str, keep_default_na=False)
    print(f"loaded {len(df):,} rows")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    jobs = list(zip(df["sample_id"], df["file_path"]))

    print(f"converting {len(jobs):,} files -> {PROCESSED_DIR} "
          f"({TARGET_RATE} Hz mono PCM16)\n")

    results = {}
    errors = []
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:

        for sample_id, record, error in pool.map(convert, jobs):

            done += 1

            if error:
                errors.append({"sample_id": sample_id, "error": error})
            else:
                results[sample_id] = record

            if done % 2000 == 0:
                print(f"    {done:,}/{len(jobs):,}")

    print(f"\nconverted {len(results):,} | errors {len(errors)}")

    for column in [
        "processed_file_path", "processed_sample_rate", "processed_channels",
        "processed_codec", "processed_duration", "processed_bytes",
        "processed_sha256",
    ]:
        df[column] = df["sample_id"].map(
            lambda s: results.get(s, {}).get(column)
        )

    df.to_csv(MASTER, index=False)

    # ---------------- validation ----------------

    processed = df[df["processed_file_path"].notna()].copy()

    processed["processed_duration"] = pd.to_numeric(
        processed["processed_duration"], errors="coerce"
    )
    processed["original_duration"] = pd.to_numeric(
        processed["duration"], errors="coerce"
    )
    processed["duration_delta"] = (
        processed["processed_duration"] - processed["original_duration"]
    ).abs()

    paths = processed["processed_file_path"].map(Path)
    missing = int((~paths.map(lambda p: p.exists())).sum())
    zero_byte = int(
        (pd.to_numeric(processed["processed_bytes"], errors="coerce") == 0).sum()
    )

    rates = processed["processed_sample_rate"].value_counts().to_dict()
    channels = processed["processed_channels"].value_counts().to_dict()
    codecs = processed["processed_codec"].value_counts().to_dict()

    over_tolerance = processed[processed["duration_delta"] > DURATION_TOLERANCE]

    failures = []

    if len(processed) != 20000:
        failures.append(f"processed {len(processed)} of 20000")
    if missing:
        failures.append(f"{missing} processed files missing")
    if zero_byte:
        failures.append(f"{zero_byte} zero-byte processed files")
    if int(processed["processed_file_path"].duplicated().sum()):
        failures.append("duplicate processed paths")
    if set(rates) != {TARGET_RATE}:
        failures.append(f"unexpected sample rates: {rates}")
    if set(channels) != {TARGET_CHANNELS}:
        failures.append(f"unexpected channel counts: {channels}")
    if set(codecs) != {"WAV/PCM_16"}:
        failures.append(f"unexpected codecs: {codecs}")
    if errors:
        failures.append(f"{len(errors)} conversion errors")

    audit = {
        "stage": "Phase 2 - canonical standardization",
        "pipeline": (
            "decode -> mono -> resample 16000 Hz -> PCM16 WAV, "
            "applied identically to every source"
        ),
        "not_applied": [
            "denoising", "dereverberation", "silence trimming", "VAD",
            "speech enhancement", "per-source normalization", "augmentation",
            "intentional duration change",
        ],
        "total_rows": int(len(df)),
        "processed_files": int(len(processed)),
        "conversion_errors": errors[:50],
        "conversion_error_count": len(errors),

        "processed_sample_rate_distribution": {
            str(k): int(v) for k, v in rates.items()
        },
        "processed_channels_distribution": {
            str(k): int(v) for k, v in channels.items()
        },
        "processed_codec_distribution": {
            str(k): int(v) for k, v in codecs.items()
        },

        "original_codec_distribution": {
            str(k): int(v) for k, v in df["codec"].value_counts().items()
        },
        "original_sample_rate_distribution": {
            str(k): int(v)
            for k, v in df["original_sample_rate"].value_counts().items()
        },

        "codec_before_after_by_source": {
            source: {
                "before": {
                    str(k): int(v)
                    for k, v in part["codec"].value_counts().items()
                },
                "after": {
                    str(k): int(v)
                    for k, v in part["processed_codec"].value_counts().items()
                },
                "sample_rate_before": {
                    str(k): int(v)
                    for k, v in part["original_sample_rate"]
                    .value_counts().items()
                },
                "sample_rate_after": {
                    str(k): int(v)
                    for k, v in part["processed_sample_rate"]
                    .value_counts().items()
                },
            }
            for source, part in processed.groupby("source_dataset")
        },

        "duration_preservation": {
            "tolerance_seconds": DURATION_TOLERANCE,
            "max_delta": round(float(processed["duration_delta"].max()), 4),
            "mean_delta": round(float(processed["duration_delta"].mean()), 6),
            "median_delta": round(
                float(processed["duration_delta"].median()), 6
            ),
            "files_over_tolerance": int(len(over_tolerance)),
            "files_over_tolerance_by_source": {
                str(k): int(v)
                for k, v in over_tolerance["source_dataset"]
                .value_counts().items()
            },
            "files_over_tolerance_by_original_codec": {
                str(k): int(v)
                for k, v in over_tolerance["codec"].value_counts().items()
            },
        },

        "duration_by_language_label_source": {
            f"{lt} | {lb} | {src}": {
                "n": int(len(part)),
                "median": round(float(part["processed_duration"].median()), 3),
                "hours": round(
                    float(part["processed_duration"].sum() / 3600), 2
                ),
            }
            for (lt, lb, src), part in processed.groupby(
                ["language_type", "label", "source_dataset"]
            )
        },

        "duration_by_split": {
            str(split): {
                "n": int(len(part)),
                "hours": round(
                    float(part["processed_duration"].sum() / 3600), 2
                ),
                "median": round(float(part["processed_duration"].median()), 3),
            }
            for split, part in processed.groupby("split")
        },

        "storage_bytes": int(
            pd.to_numeric(processed["processed_bytes"], errors="coerce").sum()
        ),

        "integrity": {
            "missing": missing,
            "zero_byte": zero_byte,
            "duplicate_processed_paths": int(
                processed["processed_file_path"].duplicated().sum()
            ),
            "unique_processed_sha256": int(
                processed["processed_sha256"].nunique()
            ),
            "duplicate_processed_sha256": int(
                processed["processed_sha256"].duplicated().sum()
            ),
        },

        "remaining_confounds": [
            "Historical compression artifacts survive: Common Voice was MP3 "
            "and Azure Hinglish fake was MP3, so their decoded waveforms "
            "still carry MP3 quantization noise even though every file is "
            "now PCM16 WAV.",
            "Kathbath was FLAC/PCM_24 (24-bit) and is now 16-bit; the "
            "original capture channel and its bandwidth remain.",
            "Sources with a native rate above 16 kHz (24/32/48 kHz) have "
            "been low-passed by resampling, while natively-16 kHz sources "
            "(Kathbath, MUCS) have not. That asymmetry is itself a residual "
            "channel cue.",
            "Uniform WAV/PCM16 output does NOT mean codec bias is gone.",
        ],

        "failures": failures,
    }

    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n" + "=" * 58)
    print("PHASE 2 VALIDATION")
    print("=" * 58)
    print(f"    processed files            {len(processed):,}")
    print(f"    missing                    {missing}")
    print(f"    zero-byte                  {zero_byte}")
    print(f"    conversion errors          {len(errors)}")
    print(f"    sample rates               {rates}")
    print(f"    channels                   {channels}")
    print(f"    codecs                     {codecs}")
    print(f"    max duration delta         "
          f"{audit['duration_preservation']['max_delta']} s")
    print(f"    files over {DURATION_TOLERANCE}s tolerance   "
          f"{len(over_tolerance):,}")
    if len(over_tolerance):
        print(f"        by original codec      "
              f"{audit['duration_preservation']['files_over_tolerance_by_original_codec']}")
    print(f"    storage                    "
          f"{audit['storage_bytes'] / 1e9:.2f} GB")

    print(f"\nAudit: {AUDIT_PATH}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("    -", f)
        raise SystemExit(1)

    print("\nPHASE 2 OK")


if __name__ == "__main__":
    main()
