"""
Final Dataset V1 audit - 20,000 development samples.

Builds metadata/dataset_v1_master.csv and metadata/dataset_v1_audit.json.

SEA-Spoof is deliberately EXCLUDED from the master manifest: it is external
evaluation only and must never enter the development pool.

This script reads audio headers and hashes bytes. It does NOT decode,
resample, convert, augment or otherwise modify a single sample.
"""

from collections import defaultdict
from pathlib import Path

import hashlib
import json

import pandas as pd
import soundfile as sf


METADATA_DIR = Path("metadata")

MASTER_PATH = METADATA_DIR / "dataset_v1_master.csv"
AUDIT_PATH = METADATA_DIR / "dataset_v1_audit.json"

CHUNK = 1024 * 1024

# Canonical groups the audit must confirm exactly.
EXPECTED = {
    ("Hindi", "real", "CommonVoice_Hindi_26.0"): 2500,
    ("Hindi", "real", "Kathbath"): 2500,
    ("Hindi", "fake", "IndicSynth"): 5000,
    ("Hinglish", "real", "MUCS_OpenSLR104"): 5000,
    ("Hinglish", "fake", "nameissakthi_hindi_english_bilingual"): 1756,
    ("Hinglish", "fake", "lingamvamshikrishnareddy_octopus_tts_hinglish"): 3244,
}

TOTAL_EXPECTED = 20000

SOURCES = [
    "common_voice_hi_v26",
    "kathbath_hi_v1",
    "indicsynth_hi",
    "mucs_hinglish",
    "hinglish_fake",
]

# License status is recorded, never inferred.
#   VERIFIED   - an explicit license is declared by the upstream repo
#   UNVERIFIED - no license could be found; do not redistribute
LICENSE_STATUS = {
    "CommonVoice_Hindi_26.0": ("CC0-1.0", "VERIFIED"),
    "Kathbath": ("CC BY 4.0", "VERIFIED"),
    "IndicSynth": ("CC BY-NC 4.0", "VERIFIED"),
    "MUCS_OpenSLR104": ("CC BY-SA 4.0", "VERIFIED"),
    "nameissakthi_hindi_english_bilingual": ("CC BY 4.0", "VERIFIED"),
    "lingamvamshikrishnareddy_octopus_tts_hinglish": (
        "none stated upstream", "UNVERIFIED"
    ),
}


def sha256_of(path):

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)

    return digest.hexdigest()


# ---------------------------------------------------------
# BUILD MASTER
# ---------------------------------------------------------

def load_source(name):

    df = pd.read_csv(METADATA_DIR / f"{name}.csv")

    out = pd.DataFrame()

    out["sample_id"] = df["sample_id"]
    out["file_path"] = df["file_path"]
    out["language"] = df["language"]
    out["language_type"] = df["language_type"]
    out["label"] = df["label"]
    out["source_dataset"] = df["source_dataset"]

    out["generator"] = df.get("generator")
    out["voice"] = df.get("voice")

    # Unify the speaker column across very different sources.
    if "target_speaker_id" in df:
        out["speaker_id"] = df["target_speaker_id"].astype("string")
        out["target_speaker_id"] = df["target_speaker_id"].astype("string")
        out["source_speaker_id"] = df.get("source_speaker_id")
    elif "speaker_id" in df:
        out["speaker_id"] = df["speaker_id"].astype("string")
        out["target_speaker_id"] = pd.NA
        out["source_speaker_id"] = pd.NA
    else:
        out["speaker_id"] = pd.NA
        out["target_speaker_id"] = pd.NA
        out["source_speaker_id"] = pd.NA

    out["session_id"] = df.get("session_id")
    out["source_file_id"] = df.get("source_file_id")
    out["gender"] = df.get("gender")
    out["transcript"] = df.get("transcript")
    out["split"] = df["split"]

    out["declared_duration"] = df.get("duration")
    out["declared_sample_rate"] = df.get("original_sample_rate")
    out["declared_codec"] = df.get("codec")

    out["license"] = df.get("license")
    out["origin_metadata_file"] = f"metadata/{name}.csv"

    return out


def probe(master):
    """Read every file's header and hash its bytes. No decoding of samples."""

    print(f"Probing {len(master):,} files (headers + sha256)...")

    records = []

    for i, path_text in enumerate(master["file_path"], start=1):

        path = Path(path_text)

        record = {
            "exists": path.exists(),
            "file_bytes": 0,
            "sha256": None,
            "duration": None,
            "original_sample_rate": None,
            "channels": None,
            "codec": None,
            "probe_error": None,
        }

        if path.exists():

            record["file_bytes"] = path.stat().st_size

            try:
                info = sf.info(str(path))
                record["duration"] = round(info.duration, 4)
                record["original_sample_rate"] = info.samplerate
                record["channels"] = info.channels
                record["codec"] = f"{info.format}/{info.subtype}"
            except Exception as e:
                record["probe_error"] = str(e)[:120]

            try:
                record["sha256"] = sha256_of(path)
            except Exception as e:
                record["probe_error"] = str(e)[:120]

        records.append(record)

        if i % 2500 == 0:
            print(f"    {i:,}/{len(master):,}")

    return pd.concat(
        [master.reset_index(drop=True), pd.DataFrame(records)],
        axis=1,
    )


# ---------------------------------------------------------
# AUDIT
# ---------------------------------------------------------

def distribution(df, by, value=None):
    """Nested dict distribution, JSON-safe."""

    out = defaultdict(dict)

    if value is None:
        grouped = df.groupby(by).size()
    else:
        grouped = df.groupby(by)[value].median()

    for key, n in grouped.items():
        if isinstance(key, tuple):
            head, tail = key[:-1], key[-1]
            node = out
            label = " | ".join(str(x) for x in head)
            node[label][str(tail)] = (
                int(n) if value is None else round(float(n), 3)
            )
        else:
            out[str(key)] = int(n) if value is None else round(float(n), 3)

    return {k: dict(v) if isinstance(v, dict) else v for k, v in out.items()}


def main():

    frames = [load_source(name) for name in SOURCES]
    master = pd.concat(frames, ignore_index=True)

    print(f"master rows before probe: {len(master):,}")

    master = probe(master)

    # License status, recorded not inferred.
    master["license"] = master["source_dataset"].map(
        lambda s: LICENSE_STATUS.get(s, ("unknown", "UNVERIFIED"))[0]
    )
    master["license_status"] = master["source_dataset"].map(
        lambda s: LICENSE_STATUS.get(s, ("unknown", "UNVERIFIED"))[1]
    )

    column_order = [
        "sample_id", "file_path",
        "language", "language_type", "label",
        "source_dataset", "generator", "voice",
        "speaker_id", "target_speaker_id", "source_speaker_id",
        "session_id", "source_file_id", "gender",
        "split",
        "duration", "original_sample_rate", "codec", "channels",
        "file_bytes", "sha256",
        "transcript",
        "license", "license_status",
        "declared_duration", "declared_sample_rate", "declared_codec",
        "exists", "probe_error", "origin_metadata_file",
    ]

    master = master[[c for c in column_order if c in master.columns]]
    master.to_csv(MASTER_PATH, index=False)

    # ---------------- integrity ----------------

    missing = master[~master["exists"]]
    zero_byte = master[master["file_bytes"] == 0]
    probe_errors = master[master["probe_error"].notna()]

    dup_ids = master[master["sample_id"].duplicated(keep=False)]
    dup_paths = master[master["file_path"].duplicated(keep=False)]

    hashes = master["sha256"].dropna()
    dup_hashes = hashes[hashes.duplicated(keep=False)]

    # ---------------- group counts ----------------

    actual_groups = (
        master.groupby(["language_type", "label", "source_dataset"])
        .size().to_dict()
    )

    group_check = {}
    group_mismatches = []

    for key, expected in EXPECTED.items():
        got = int(actual_groups.get(key, 0))
        label = " | ".join(key)
        group_check[label] = {"expected": expected, "actual": got}
        if got != expected:
            group_mismatches.append(f"{label}: expected {expected}, got {got}")

    unexpected = [
        " | ".join(str(x) for x in k)
        for k in actual_groups
        if k not in EXPECTED
    ]

    # ---------------- declared vs actual ----------------

    both_sr = master[
        master["declared_sample_rate"].notna()
        & master["original_sample_rate"].notna()
    ]
    sr_mismatch = int(
        (
            pd.to_numeric(both_sr["declared_sample_rate"], errors="coerce")
            != both_sr["original_sample_rate"]
        ).sum()
    )

    both_dur = master[
        master["declared_duration"].notna() & master["duration"].notna()
    ]
    dur_mismatch = int(
        (
            (
                pd.to_numeric(both_dur["declared_duration"], errors="coerce")
                - both_dur["duration"]
            ).abs() > 0.05
        ).sum()
    )

    # ---------------- speakers ----------------

    speaker_counts = {}

    for source, part in master.groupby("source_dataset"):
        speakers = part["speaker_id"].dropna()
        speakers = speakers[speakers.astype(str).str.strip() != ""]
        speaker_counts[source] = {
            "rows_with_speaker_id": int(len(speakers)),
            "unique_speakers": int(speakers.nunique()),
        }

    # ---------------- leakage within each source ----------------

    leakage = {}

    for source, part in master.groupby("source_dataset"):

        key = "speaker_id"
        speakers = part.dropna(subset=[key])

        if speakers.empty:
            leakage[source] = "no speaker ids - not speaker-disjoint by design"
            continue

        groups = {
            split: set(speakers[speakers["split"] == split][key])
            for split in ["train", "validation", "test"]
        }

        leakage[source] = {
            "train_validation": len(groups["train"] & groups["validation"]),
            "train_test": len(groups["train"] & groups["test"]),
            "validation_test": len(groups["validation"] & groups["test"]),
        }

    # ---------------- shortcut-feature severity ----------------
    #
    # A metadata feature is a PERFECT shortcut if no single value of it is
    # shared between real and fake within a language half: a classifier
    # reading only that feature would score 100% without hearing anything.

    shortcuts = {}

    for feature in ["codec", "original_sample_rate", "channels"]:

        per_language = {}

        for language_type, part in master.groupby("language_type"):

            real_values = set(part[part["label"] == "real"][feature].dropna())
            fake_values = set(part[part["label"] == "fake"][feature].dropna())

            shared = real_values & fake_values

            covered = int(
                part[feature].isin(shared).sum()
            )

            per_language[str(language_type)] = {
                "real_values": sorted(str(v) for v in real_values),
                "fake_values": sorted(str(v) for v in fake_values),
                "shared_values": sorted(str(v) for v in shared),
                "perfectly_separates_label": len(shared) == 0,
                "rows_with_ambiguous_value": covered,
            }

        per_language["perfect_in_any_language"] = any(
            v["perfectly_separates_label"]
            for v in per_language.values()
            if isinstance(v, dict)
        )

        shortcuts[feature] = per_language

    # ---------------- split ratio sanity ----------------

    split_table = pd.crosstab(master["source_dataset"], master["split"])
    split_ratio = split_table.div(split_table.sum(axis=1), axis=0)

    split_anomalies = []

    for source in split_table.index:

        for split in ["train", "validation", "test"]:

            if split not in split_table.columns:
                continue

            if int(split_table.loc[source, split]) == 0:
                split_anomalies.append(
                    f"{source}: '{split}' split is EMPTY - this source is "
                    f"never evaluated"
                )

    duration = master["duration"].dropna()

    audit = {
        "dataset": "Dataset V1 - development pool (SEA-Spoof excluded)",
        "total_samples": int(len(master)),
        "total_expected": TOTAL_EXPECTED,
        "total_matches_expected": bool(len(master) == TOTAL_EXPECTED),

        "class_counts": {
            "by_label": {
                str(k): int(v)
                for k, v in master["label"].value_counts().items()
            },
            "by_language_type": {
                str(k): int(v)
                for k, v in master["language_type"].value_counts().items()
            },
            "language_type_x_label": distribution(
                master, ["language_type", "label"]
            ),
        },

        "canonical_group_check": group_check,
        "canonical_group_mismatches": group_mismatches,
        "unexpected_groups": unexpected,

        "source_distribution": {
            str(k): int(v)
            for k, v in master["source_dataset"].value_counts().items()
        },
        "generator_distribution": {
            str(k): int(v)
            for k, v in master["generator"].value_counts(dropna=False).items()
        },
        "voice_distribution": {
            str(k): int(v)
            for k, v in master["voice"].value_counts(dropna=False).items()
        },
        "split_distribution": {
            str(k): int(v)
            for k, v in master["split"].value_counts().items()
        },
        "split_x_language_type_x_label": distribution(
            master, ["language_type", "label", "split"]
        ),

        "integrity": {
            "missing_files": int(len(missing)),
            "zero_byte_files": int(len(zero_byte)),
            "probe_errors": int(len(probe_errors)),
            "duplicate_sample_ids": int(len(dup_ids)),
            "duplicate_paths": int(len(dup_paths)),
            "duplicate_sha256_payloads": int(len(dup_hashes)),
            "unique_sha256": int(hashes.nunique()),
            "declared_vs_actual_sample_rate_mismatches": sr_mismatch,
            "declared_vs_actual_duration_mismatches": dur_mismatch,
            "total_bytes": int(master["file_bytes"].sum()),
        },

        "duration": {
            "total_hours": round(float(duration.sum() / 3600), 2),
            "min": round(float(duration.min()), 3),
            "median": round(float(duration.median()), 3),
            "mean": round(float(duration.mean()), 3),
            "max": round(float(duration.max()), 3),
            "median_by_language_type_label_source": distribution(
                master, ["language_type", "label", "source_dataset"], "duration"
            ),
            "hours_by_language_type_label": {
                f"{lt} | {lb}": round(float(part["duration"].sum() / 3600), 2)
                for (lt, lb), part in master.groupby(["language_type", "label"])
            },
            "clips_under_2s": int((duration < 2).sum()),
            "clips_under_1s": int((duration < 1).sum()),
        },

        "codec_distribution": {
            "overall": {
                str(k): int(v)
                for k, v in master["codec"].value_counts(dropna=False).items()
            },
            "by_language_type_label_source": distribution(
                master, ["language_type", "label", "source_dataset", "codec"]
            ),
        },

        "sample_rate_distribution": {
            "overall": {
                str(k): int(v)
                for k, v in master["original_sample_rate"]
                .value_counts(dropna=False).items()
            },
            "by_language_type_label": distribution(
                master, ["language_type", "label", "original_sample_rate"]
            ),
        },

        "channels_distribution": {
            str(k): int(v)
            for k, v in master["channels"].value_counts(dropna=False).items()
        },

        "shortcut_feature_analysis": shortcuts,

        "split_ratio_by_source": {
            str(source): {
                str(split): round(float(split_ratio.loc[source, split]), 3)
                for split in split_ratio.columns
            }
            for source in split_ratio.index
        },
        "split_anomalies": split_anomalies,
        "label_balance_per_split": distribution(
            master, ["split", "language_type", "label"]
        ),

        "speaker_counts": speaker_counts,
        "split_leakage_by_source": leakage,

        "license_status": {
            str(k): int(v)
            for k, v in master["license_status"].value_counts().items()
        },
        "license_by_source": {
            source: {
                "license": LICENSE_STATUS.get(source, ("unknown", ""))[0],
                "status": LICENSE_STATUS.get(source, ("", "UNVERIFIED"))[1],
                "clips": int((master["source_dataset"] == source).sum()),
            }
            for source in master["source_dataset"].unique()
        },

        "external_evaluation_excluded": {
            "dataset": "SEA-Spoof Hindi",
            "clips": 2000,
            "location": "data/evaluation/seaspoof_hi",
            "metadata": "metadata/evaluation/seaspoof_hi.csv",
            "note": (
                "Held-out external evaluation only. Deliberately NOT part of "
                "dataset_v1_master.csv and must never enter training."
            ),
        },

        "confounds_and_limitations": [
            "PERFECT CODEC SHORTCUT (Hindi): Hindi real is FLAC/PCM_24 "
            "(Kathbath 2,500) + MP3 (Common Voice 2,500) while Hindi fake is "
            "100% WAV/PCM_16 (IndicSynth). No codec value is shared between "
            "the classes, so codec alone predicts the label with 100% "
            "accuracy on the Hindi half. This is MORE severe than the "
            "Hinglish codec issue.",

            "PERFECT SAMPLE-RATE SHORTCUT (both halves): Hindi real is "
            "16 kHz / 32 kHz / 48 kHz and Hindi fake is 24 kHz; Hinglish "
            "real is 16 kHz and Hinglish fake is 24 kHz. Sample rate alone "
            "predicts the label with 100% accuracy in BOTH language halves. "
            "A single common resample is mandatory before any training.",

            "SPLIT DEFECT: Kathbath (2,500 Hindi real clips) is 100% train "
            "with an EMPTY validation and test split, so that source is "
            "never evaluated. This also unbalances the Hindi half: test "
            "holds 751 fake vs 407 real, and overall splits are "
            "73.6/13.3/13.2 rather than 70/15/15.",

            "CODEC CONFOUND: Hinglish fake is 3,244 MP3 (Azure) + 1,756 PCM "
            "WAV (Parler), while Hinglish real (MUCS) is 100% PCM WAV. Codec "
            "is therefore correlated with the label. Decoding MP3 to WAV/16k "
            "does NOT remove MP3 compression artefacts - they are baked into "
            "the decoded waveform. Must be handled with controlled, "
            "class-balanced codec augmentation so REAL and FAKE both see "
            "MP3/Opus/etc. NOT applied yet.",

            "DURATION CONFOUND: median clip length differs by source "
            "(Parler ~1.5 s vs MUCS ~5.1 s vs Azure ~5.4 s). Train on fixed "
            "duration windows/crops so utterance length cannot be a cue. "
            "Short clips are retained, not deleted.",

            "SAMPLE-RATE CONFOUND: sources arrive at different native rates "
            "(Hinglish fake 24 kHz vs MUCS 16 kHz). A single common "
            "resampling step is required before training.",

            "Hindi fake (IndicSynth) covers only 2 generators "
            "(xtts_v2, freevc24) - VITS does not exist in the Hindi config.",

            "Hinglish fake covers only 2 generators and 3 synthetic voices. "
            "Not valid evidence of unseen-generator or unseen-speaker "
            "generalization.",

            "Hinglish fake splits are transcript-disjoint, NOT "
            "speaker-disjoint: only three synthetic voices exist.",

            "MUCS speakers are effectively one recording each, so speaker "
            "identity is confounded with recording channel.",

            "LICENSE: Source B "
            "(lingamvamshikrishnareddy/octopus-tts-hinglish, 3,244 clips) has "
            "no README, no LICENSE file, no license tag and no cardData. "
            "Marked UNVERIFIED. Do not redistribute these files. Note also "
            "that the audio was produced with Microsoft Azure/Edge neural "
            "voices, whose upstream usage terms are a separate question.",

            "IndicSynth is CC BY-NC 4.0 - non-commercial use only, which "
            "constrains the whole derived dataset.",
        ],
    }

    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    # ---------------- console report ----------------

    print("\n" + "=" * 58)
    print("DATASET V1 AUDIT")
    print("=" * 58)

    print(f"\nTotal samples: {len(master):,} "
          f"(expected {TOTAL_EXPECTED:,}) "
          f"{'OK' if len(master) == TOTAL_EXPECTED else 'MISMATCH'}")

    print("\nCanonical groups:")
    for label, values in group_check.items():
        status = "OK" if values["expected"] == values["actual"] else "FAIL"
        print(f"    {status}  {label:<62} {values['actual']:>5} "
              f"/ {values['expected']}")

    if unexpected:
        print("\n    UNEXPECTED GROUPS:", unexpected)

    print("\nlanguage_type x label:")
    for (lt, lb), part in master.groupby(["language_type", "label"]):
        print(f"    {lt:<10} {lb:<5} {len(part):>6}")

    print("\nIntegrity:")
    for key, value in audit["integrity"].items():
        print(f"    {key:<48} {value:,}")

    print("\nDuration (hours) by language_type | label:")
    for key, value in audit["duration"]["hours_by_language_type_label"].items():
        print(f"    {key:<24} {value:>7.2f} h")

    print("\nMedian duration by group:")
    for group, part in master.groupby(
        ["language_type", "label", "source_dataset"]
    ):
        print(f"    {' | '.join(group):<66} "
              f"{part['duration'].median():>6.2f} s")

    print("\nCodec by group:")
    for group, part in master.groupby(
        ["language_type", "label", "source_dataset"]
    ):
        codecs = part["codec"].value_counts().to_dict()
        print(f"    {' | '.join(group):<66} {codecs}")

    print("\nSample rate overall:")
    for rate, n in master["original_sample_rate"].value_counts().items():
        print(f"    {rate} Hz  {n:,}")

    print("\nSplits:")
    print(master["split"].value_counts().to_string())

    print("\nSHORTCUT FEATURE ANALYSIS (does metadata alone predict label?):")
    for feature, per_language in shortcuts.items():
        for language_type, values in per_language.items():
            if not isinstance(values, dict):
                continue
            verdict = (
                "PERFECT SHORTCUT" if values["perfectly_separates_label"]
                else f"{values['rows_with_ambiguous_value']:,} ambiguous rows"
            )
            print(f"    {feature:<22} {language_type:<10} {verdict}")

    if split_anomalies:
        print("\nSPLIT ANOMALIES:")
        for anomaly in split_anomalies:
            print("    !", anomaly)

    print("\nSpeakers:")
    for source, values in speaker_counts.items():
        print(f"    {source:<50} {values['unique_speakers']:>5} unique "
              f"({values['rows_with_speaker_id']:,} rows)")

    print("\nSplit leakage (speaker level):")
    for source, values in leakage.items():
        print(f"    {source:<50} {values}")

    print("\nLicense status:")
    for source, values in audit["license_by_source"].items():
        print(f"    {values['status']:<11} {source:<50} "
              f"{values['clips']:>5}  {values['license']}")

    print(f"\nMaster manifest: {MASTER_PATH}")
    print(f"Audit report:    {AUDIT_PATH}")
    print("\nNo audio was decoded, resampled, converted or modified.")


if __name__ == "__main__":
    main()
