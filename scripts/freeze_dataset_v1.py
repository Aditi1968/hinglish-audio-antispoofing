"""
Freeze Dataset V1.

READ-ONLY with respect to audio. This script never writes, converts or
touches a single audio file. It only:

  1. re-runs the full integrity validation
  2. verifies every stored per-file SHA256 still matches the bytes on disk
  3. computes an aggregate fingerprint per audio collection
  4. hashes every metadata/audit artifact
  5. writes metadata/dataset_v1_freeze_manifest.json

Hashes are streamed in fixed-size chunks; nothing is loaded whole into RAM.
"""

from datetime import datetime, timezone
from pathlib import Path

import hashlib
import json
import time

import pandas as pd


MASTER = Path("metadata/dataset_v1_master_v2.csv")
AUGMENTED = Path("metadata/dataset_v1_augmented.csv")
ROBUSTNESS = Path("metadata/evaluation/robustness_v1.csv")

FREEZE_PATH = Path("metadata/dataset_v1_freeze_manifest.json")

CANONICAL_DIR = Path("data/processed/v1_16k")
AUGMENTED_DIR = Path("data/augmented/v1")
ROBUSTNESS_DIR = Path("data/evaluation/robustness_v1")

ARTIFACTS = [
    "DATASET_V1_CARD.md",
    "metadata/DATASET_V1_SHORTCUT_REPORT.md",
    "metadata/dataset_v1_master.csv",
    "metadata/dataset_v1_master_v2.csv",
    "metadata/dataset_v1_augmented.csv",
    "metadata/evaluation/robustness_v1.csv",
    "metadata/dataset_v1_audit.json",
    "metadata/dataset_v1_preprocessing_audit.json",
    "metadata/dataset_v1_shortcut_audit.json",
    "metadata/common_voice_hi_v26.csv",
    "metadata/kathbath_hi_v1.csv",
    "metadata/indicsynth_hi.csv",
    "metadata/mucs_hinglish.csv",
    "metadata/hinglish_fake.csv",
    "metadata/evaluation/seaspoof_hi.csv",
    "scripts/fix_dataset_splits.py",
    "scripts/preprocess_dataset_v1.py",
    "scripts/augment_dataset_v1.py",
    "scripts/audit_shortcuts_v1.py",
    "scripts/audit_dataset_v1.py",
    "scripts/freeze_dataset_v1.py",
]

CHUNK = 256 * 1024

SPLITS = ["train", "validation", "test"]


def sha256_of(path):
    """Streamed, chunked, retried under memory pressure."""

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


def aggregate_fingerprint(pairs):
    """
    One hash standing for a whole collection.

    Built from sorted "<id>:<sha256>" lines, so it is stable regardless of
    filesystem ordering and changes if any file or id changes.
    """

    digest = hashlib.sha256()

    for identifier, file_hash in sorted(pairs):
        digest.update(f"{identifier}:{file_hash}\n".encode("utf-8"))

    return digest.hexdigest()


def verify_collection(df, id_column, path_column, hash_column, label):
    """Re-hash every file and compare against the stored value."""

    print(f"\nverifying {len(df):,} {label} files (streamed SHA256)...")

    mismatches = []
    missing = []
    zero_byte = []
    pairs = []

    for i, row in enumerate(df.itertuples(index=False), start=1):

        path = Path(getattr(row, path_column))
        identifier = getattr(row, id_column)
        stored = getattr(row, hash_column)

        if not path.exists():
            missing.append(identifier)
            continue

        if path.stat().st_size == 0:
            zero_byte.append(identifier)
            continue

        actual = sha256_of(path)
        pairs.append((identifier, actual))

        if stored and actual != stored:
            mismatches.append(identifier)

        if i % 5000 == 0:
            print(f"    {i:,}/{len(df):,}")

    print(f"    missing {len(missing)} | zero-byte {len(zero_byte)} | "
          f"hash mismatches {len(mismatches)}")

    return {
        "files": int(len(df)),
        "verified": len(pairs),
        "missing": len(missing),
        "zero_byte": len(zero_byte),
        "hash_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:10],
        "aggregate_sha256": aggregate_fingerprint(pairs),
    }


def main():

    failures = []

    master = pd.read_csv(MASTER, dtype=str, keep_default_na=False)
    augmented = pd.read_csv(AUGMENTED, dtype=str, keep_default_na=False)
    robustness = pd.read_csv(ROBUSTNESS, dtype=str, keep_default_na=False)

    print("=" * 62)
    print("DATASET V1 FREEZE - READ-ONLY VALIDATION")
    print("=" * 62)

    def check(condition, message):
        print(f"    {'OK  ' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    # ---------------- counts ----------------

    print("\nCOUNTS")
    check(len(master) == 20000, f"canonical metadata rows = {len(master)}")
    check(len(augmented) == 13963, f"training augmentation rows = {len(augmented)}")
    check(len(robustness) == 2000, f"robustness rows = {len(robustness)}")

    # ---------------- format ----------------

    print("\nCANONICAL FORMAT")
    check(set(master["processed_sample_rate"]) == {"16000"},
          f"sample rate = {sorted(set(master['processed_sample_rate']))}")
    check(set(master["processed_channels"]) == {"1"},
          f"channels = {sorted(set(master['processed_channels']))}")
    check(set(master["processed_codec"]) == {"WAV/PCM_16"},
          f"codec = {sorted(set(master['processed_codec']))}")

    # ---------------- identity uniqueness ----------------

    print("\nUNIQUENESS")
    check(master["sample_id"].duplicated().sum() == 0, "duplicate sample_id = 0")
    check(master["processed_file_path"].duplicated().sum() == 0,
          "duplicate canonical paths = 0")
    check(augmented["augmentation_id"].duplicated().sum() == 0,
          "duplicate augmentation_id = 0")
    check(augmented["augmented_file_path"].duplicated().sum() == 0,
          "duplicate augmented paths = 0")
    check(robustness["robustness_id"].duplicated().sum() == 0,
          "duplicate robustness_id = 0")
    check(robustness["augmented_file_path"].duplicated().sum() == 0,
          "duplicate robustness paths = 0")

    # ---------------- leakage ----------------

    print("\nGROUP LEAKAGE (composite source::group)")

    leakage_report = {}

    for source, part in master.groupby("source_dataset"):

        groups = {
            split: set(part[part["split"] == split]["group_id"])
            for split in SPLITS
        }

        overlaps = {
            "train_validation": len(groups["train"] & groups["validation"]),
            "train_test": len(groups["train"] & groups["test"]),
            "validation_test": len(groups["validation"] & groups["test"]),
        }

        leakage_report[source] = overlaps
        check(not any(overlaps.values()), f"{source}: {overlaps}")

    pooled = {
        split: set(master[master["split"] == split]["group_id"])
        for split in SPLITS
    }
    global_overlap = {
        "train_validation": len(pooled["train"] & pooled["validation"]),
        "train_test": len(pooled["train"] & pooled["test"]),
        "validation_test": len(pooled["validation"] & pooled["test"]),
    }
    check(not any(global_overlap.values()),
          f"pooled global overlap {global_overlap}")

    # ---------------- split inheritance ----------------

    print("\nSPLIT INHERITANCE")

    parent_split = master.set_index("sample_id")["split"]

    train_ids = set(master[master["split"] == "train"]["sample_id"])
    test_ids = set(master[master["split"] == "test"]["sample_id"])

    check(set(augmented["parent_split"]) == {"train"},
          "augmented parents are all train")
    check(set(augmented["parent_sample_id"]) <= train_ids,
          "no validation/test parent in training augmentation manifest")
    check(
        all(
            parent_split.get(p) == s
            for p, s in zip(augmented["parent_sample_id"], augmented["split"])
        ),
        "derived split == parent split for every augmented row",
    )
    check(augmented["parent_sample_id"].duplicated().sum() == 0,
          "one derived file per training parent")

    check(set(robustness["parent_split"]) == {"test"},
          "robustness parents are all test")
    check(set(robustness["parent_sample_id"]) <= test_ids,
          "robustness parents come only from the test split")
    check(
        len(set(augmented["parent_sample_id"])
            & set(robustness["parent_sample_id"])) == 0,
        "no parent shared between training augmentation and robustness",
    )

    # ---------------- hash verification ----------------

    canonical_check = verify_collection(
        master, "sample_id", "processed_file_path", "processed_sha256",
        "canonical",
    )
    augmented_check = verify_collection(
        augmented, "augmentation_id", "augmented_file_path", "sha256",
        "augmented",
    )
    robustness_check = verify_collection(
        robustness, "robustness_id", "augmented_file_path", "sha256",
        "robustness",
    )

    for name, result in [
        ("canonical", canonical_check),
        ("augmented", augmented_check),
        ("robustness", robustness_check),
    ]:
        check(result["missing"] == 0, f"{name}: 0 missing")
        check(result["zero_byte"] == 0, f"{name}: 0 zero-byte")
        check(result["hash_mismatches"] == 0, f"{name}: 0 hash mismatches")

    # ---------------- artifact hashes ----------------

    artifact_hashes = {}

    for relative in ARTIFACTS:
        path = Path(relative)
        if path.exists():
            artifact_hashes[relative] = {
                "sha256": sha256_of(path),
                "bytes": path.stat().st_size,
            }

    # ---------------- storage ----------------

    def directory_bytes(directory):
        return int(sum(
            p.stat().st_size for p in Path(directory).rglob("*") if p.is_file()
        ))

    storage = {
        str(CANONICAL_DIR): directory_bytes(CANONICAL_DIR),
        str(AUGMENTED_DIR): directory_bytes(AUGMENTED_DIR),
        str(ROBUSTNESS_DIR): directory_bytes(ROBUSTNESS_DIR),
    }

    # ---------------- manifest ----------------

    master["processed_duration_num"] = pd.to_numeric(
        master["processed_duration"], errors="coerce"
    )

    def counts(frame, column):
        return {
            str(k): int(v)
            for k, v in frame[column].value_counts().items()
            if str(k).strip()
        }

    speaker_groups = {
        source: int(part["group_id"].nunique())
        for source, part in master.groupby("source_dataset")
    }

    augmentation_parameters = {}

    for condition, part in augmented.groupby("augmentation_condition"):
        augmentation_parameters[str(condition)] = {
            "count": int(len(part)),
            "parameters_observed": sorted(
                set(part["augmentation_parameters"])
            ),
        }

    manifest = {
        "dataset_version": "v1",
        "status": "FROZEN",
        "freeze_date_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),

        "random_seeds": {
            "split_assignment": 42,
            "augmentation_condition_deal": 42,
            "robustness_parent_selection": 43,
            "note": (
                "Kathbath split allocation is fully deterministic (sorted by "
                "clip count desc, tie-break by speaker id asc) and does not "
                "consume the RNG."
            ),
        },

        "totals": {
            "canonical_development_samples": int(len(master)),
            "training_augmentation_files": int(len(augmented)),
            "robustness_evaluation_files": int(len(robustness)),
            "external_evaluation_samples_not_included": 2000,
        },

        "split_counts": counts(master, "split"),
        "language_type_counts": counts(master, "language_type"),
        "label_counts": counts(master, "label"),
        "language_label_split": {
            f"{lt} | {lb}": counts(part, "split")
            for (lt, lb), part in master.groupby(["language_type", "label"])
        },
        "source_counts": counts(master, "source_dataset"),
        "source_split_counts": {
            source: counts(part, "split")
            for source, part in master.groupby("source_dataset")
        },
        "generator_counts": counts(master, "generator"),
        "voice_counts": counts(master, "voice"),
        "gender_normalized_counts": counts(master, "gender_normalized"),
        "unique_groups_per_source": speaker_groups,

        "canonical_audio_format": {
            "sample_rate": 16000,
            "channels": 1,
            "codec": "WAV/PCM_16",
            "pipeline": "decode -> mono -> resample 16 kHz -> PCM16 WAV",
            "not_applied": [
                "denoising", "dereverberation", "silence trimming", "VAD",
                "speech enhancement", "per-source normalization",
                "intentional duration change",
            ],
        },

        "duration": {
            "total_hours": round(
                float(master["processed_duration_num"].sum() / 3600), 2
            ),
            "median": round(
                float(master["processed_duration_num"].median()), 3
            ),
            "min": round(float(master["processed_duration_num"].min()), 3),
            "max": round(float(master["processed_duration_num"].max()), 3),
        },

        "augmentation": {
            "policy": (
                "one derived file per TRAINING parent; validation and test "
                "are never augmented for training; every derived file "
                "inherits its parent's split verbatim"
            ),
            "conditions": augmentation_parameters,
            "not_implemented": [
                "environmental noise (NOISE_DIR hook only)",
                "room impulse responses (RIR_DIR hook only)",
                "physical replay - cannot be synthesised honestly",
            ],
        },

        "robustness_set": {
            "parents": int(robustness["parent_sample_id"].nunique()),
            "parent_split": "test",
            "conditions": counts(robustness, "augmentation_condition"),
            "composition": {
                f"{lt} | {lb}": counts(part, "augmentation_condition")
                for (lt, lb), part in robustness.groupby(
                    ["language_type", "label"]
                )
            },
        },

        "audio_collections": {
            "canonical": {
                "directory": str(CANONICAL_DIR),
                "per_file_hash_manifest": str(MASTER),
                "per_file_hash_column": "processed_sha256",
                **canonical_check,
            },
            "augmented_training": {
                "directory": str(AUGMENTED_DIR),
                "per_file_hash_manifest": str(AUGMENTED),
                "per_file_hash_column": "sha256",
                **augmented_check,
            },
            "robustness_evaluation": {
                "directory": str(ROBUSTNESS_DIR),
                "per_file_hash_manifest": str(ROBUSTNESS),
                "per_file_hash_column": "sha256",
                **robustness_check,
            },
        },

        "artifact_sha256": artifact_hashes,

        "scripts_used": [
            "scripts/fix_dataset_splits.py",
            "scripts/preprocess_dataset_v1.py",
            "scripts/augment_dataset_v1.py",
            "scripts/audit_shortcuts_v1.py",
            "scripts/audit_dataset_v1.py",
            "scripts/freeze_dataset_v1.py",
        ],

        "licenses": {
            source: {
                "license": part["license"].iloc[0],
                "status": part["license_status"].iloc[0],
                "clips": int(len(part)),
            }
            for source, part in master.groupby("source_dataset")
        },

        "external_evaluation": {
            "dataset": "SEA-Spoof Hindi",
            "clips": 2000,
            "audio": "data/evaluation/seaspoof_hi",
            "metadata": "metadata/evaluation/seaspoof_hi.csv",
            "note": (
                "Frozen separately. NOT part of the development pool and "
                "never used for training."
            ),
        },

        "storage_bytes": storage,
        "storage_total_gb": round(sum(storage.values()) / 1e9, 2),

        "leakage_by_source": leakage_report,
        "pooled_global_group_overlap": global_overlap,

        "validation_failures": failures,
        "frozen": not failures,
    }

    FREEZE_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print("AGGREGATE FINGERPRINTS")
    print("=" * 62)
    for name, result in [
        ("canonical  ", canonical_check),
        ("augmented  ", augmented_check),
        ("robustness ", robustness_check),
    ]:
        print(f"    {name} {result['aggregate_sha256']}")

    print(f"\n    storage {manifest['storage_total_gb']} GB")
    print(f"\nFreeze manifest: {FREEZE_PATH}")

    if failures:
        print("\nFREEZE BLOCKED - failures:")
        for f in failures:
            print("    -", f)
        raise SystemExit(1)

    print("\nDATASET V1 FROZEN - all validations passed")
    print("No audio was written or modified by this script.")


if __name__ == "__main__":
    main()
