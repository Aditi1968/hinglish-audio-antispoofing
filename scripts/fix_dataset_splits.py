"""
Phase 1 - persist the approved split fix.

Only Kathbath is re-split. Common Voice, IndicSynth, MUCS and both Hinglish
fake sources keep their existing, already-validated assignments.

Kathbath allocation (approved dry run):
    sort speakers by clip count descending, tie-break by speaker id ascending,
    assign each WHOLE speaker to the split currently furthest below its
    sample target (1750 / 375 / 375).

Also adds two derived columns used by every later stage:

    group_id           composite "source_dataset::group" identity. Speaker ids
                       are SOURCE-LOCAL: all 51 Kathbath ids collide with
                       IndicSynth target_speaker_id strings, so grouping on a
                       bare speaker_id would silently merge unrelated speakers.

    gender_normalized  female / male / unknown. The original `gender` column
                       is left untouched.

Reads  metadata/dataset_v1_master.csv   (never modified)
Writes metadata/dataset_v1_master_v2.csv
"""

from pathlib import Path

import re

import pandas as pd


MASTER_IN = Path("metadata/dataset_v1_master.csv")
MASTER_OUT = Path("metadata/dataset_v1_master_v2.csv")

# Grouping key per source. Hinglish fake has no speaker ids at all - only
# three synthetic voices - so the transcript is the only real grouping unit.
GROUP_KEY = {
    "CommonVoice_Hindi_26.0": "speaker_id",
    "Kathbath": "speaker_id",
    "IndicSynth": "target_speaker_id",
    "MUCS_OpenSLR104": "speaker_id",
    "nameissakthi_hindi_english_bilingual": "transcript",
    "lingamvamshikrishnareddy_octopus_tts_hinglish": "transcript",
}

KATHBATH_TARGETS = {"train": 1750, "validation": 375, "test": 375}

GENDER_MAP = {
    "female": "female",
    "Female": "female",
    "female_feminine": "female",
    "male": "male",
    "Male": "male",
    "male_masculine": "male",
}

SPLITS = ["train", "validation", "test"]


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def build_group_id(row):

    key = GROUP_KEY[row["source_dataset"]]
    value = row[key]

    if key == "transcript":
        value = normalize_text(value)
    else:
        value = str(value).strip()

    return f"{row['source_dataset']}::{value}"


def normalize_gender(value):

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "unknown"}:
        return "unknown"

    return GENDER_MAP.get(text, GENDER_MAP.get(text.lower(), "unknown"))


def reallocate_kathbath(df):
    """Deterministic, group-aware, sample-count-targeted allocation."""

    kathbath = df[df["source_dataset"] == "Kathbath"]
    counts = kathbath["speaker_id"].value_counts()

    current = {split: 0 for split in KATHBATH_TARGETS}
    assignment = {}

    # Largest speaker first; ties broken by speaker id so the result is
    # reproducible regardless of pandas ordering.
    for speaker in sorted(counts.index, key=lambda s: (-counts[s], str(s))):

        split = max(
            KATHBATH_TARGETS,
            key=lambda s: KATHBATH_TARGETS[s] - current[s],
        )

        assignment[speaker] = split
        current[split] += int(counts[speaker])

    print("Kathbath reallocation:")
    for split in SPLITS:
        speakers = sum(1 for v in assignment.values() if v == split)
        print(f"    {split:<11} {speakers:>3} speakers | "
              f"{current[split]:>5} clips")

    return assignment


def verify(df):

    print("\nLeakage verification (composite group_id):")

    failures = []

    for source, part in df.groupby("source_dataset"):

        groups = {
            split: set(part[part["split"] == split]["group_id"])
            for split in SPLITS
        }

        overlaps = {
            "train/validation": len(groups["train"] & groups["validation"]),
            "train/test": len(groups["train"] & groups["test"]),
            "validation/test": len(groups["validation"] & groups["test"]),
        }

        bad = {k: v for k, v in overlaps.items() if v}

        status = "OK  " if not bad else "FAIL"
        print(f"    {status} {source:<48} "
              + " | ".join(f"{k} {v}" for k, v in overlaps.items()))

        if bad:
            failures.append(f"{source}: {bad}")

        # Every source must appear in all three splits.
        for split in SPLITS:
            if int((part["split"] == split).sum()) == 0:
                failures.append(f"{source}: split '{split}' is empty")

    # Global check across the pooled composite keys.
    groups = {
        split: set(df[df["split"] == split]["group_id"])
        for split in SPLITS
    }

    global_overlap = {
        "train/validation": len(groups["train"] & groups["validation"]),
        "train/test": len(groups["train"] & groups["test"]),
        "validation/test": len(groups["validation"] & groups["test"]),
    }

    print(f"\n    GLOBAL pooled composite-key overlap: {global_overlap}")

    if any(global_overlap.values()):
        failures.append(f"global group leakage: {global_overlap}")

    return failures


def main():

    df = pd.read_csv(MASTER_IN, dtype=str, keep_default_na=False)

    print(f"loaded {len(df):,} rows from {MASTER_IN}")

    df["group_id"] = df.apply(build_group_id, axis=1)
    df["gender_normalized"] = df["gender"].map(normalize_gender)

    assignment = reallocate_kathbath(df)

    mask = df["source_dataset"] == "Kathbath"
    df.loc[mask, "split"] = df.loc[mask, "speaker_id"].map(assignment)

    failures = verify(df)

    print("\nSource x split:")
    table = pd.crosstab(df["source_dataset"], df["split"])
    table = table[[c for c in SPLITS if c in table.columns]]
    print(table.to_string())

    print("\nlanguage_type x label x split:")
    print(pd.crosstab(
        [df["language_type"], df["label"]], df["split"]
    )[SPLITS].to_string())

    print("\nTotals:")
    for split in SPLITS:
        n = int((df["split"] == split).sum())
        print(f"    {split:<11} {n:>6}  {n / len(df) * 100:5.2f}%")

    print("\ngender_normalized:")
    print(df["gender_normalized"].value_counts().to_string())

    if len(df) != 20000:
        failures.append(f"expected 20000 rows, got {len(df)}")

    df.to_csv(MASTER_OUT, index=False)
    print(f"\nWrote {MASTER_OUT} ({len(df):,} rows)")
    print(f"{MASTER_IN} left untouched.")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("    -", f)
        raise SystemExit(1)

    print("\nPHASE 1 OK")


if __name__ == "__main__":
    main()
