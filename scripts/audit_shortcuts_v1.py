"""
Phase 4 - shortcut / confound audit.

Answers one question for every metadata feature: how much of the real-vs-fake
decision could a model make WITHOUT listening to the audio?

Two measures per feature:

  majority_accuracy   accuracy of the best possible classifier that sees only
                      that feature (predict each value's majority label).
                      0.50 = useless, 1.00 = the feature IS the label.

  cramers_v           strength of association, 0..1.

Audited across three representations:
  ORIGINAL    as-collected files (pre-standardisation)
  CANONICAL   16 kHz mono PCM16 (post Phase 2)
  AUGMENTED   the Phase 3 training view

Reads  metadata/dataset_v1_master_v2.csv
       metadata/dataset_v1_augmented.csv
       metadata/evaluation/robustness_v1.csv
Writes metadata/dataset_v1_shortcut_audit.json
"""

from pathlib import Path

import json
import math

import pandas as pd


MASTER = Path("metadata/dataset_v1_master_v2.csv")
AUGMENTED = Path("metadata/dataset_v1_augmented.csv")
ROBUSTNESS = Path("metadata/evaluation/robustness_v1.csv")
AUDIT_PATH = Path("metadata/dataset_v1_shortcut_audit.json")

DURATION_BINS = [0, 1, 2, 3, 5, 8, 12, 20, 1e9]
DURATION_LABELS = [
    "<1s", "1-2s", "2-3s", "3-5s", "5-8s", "8-12s", "12-20s", ">20s"
]

# Above this, a single metadata column carries most of the decision.
STRONG = 0.75


def cramers_v(table):

    chi2 = 0.0
    total = table.values.sum()

    if total == 0:
        return 0.0

    row_sums = table.sum(axis=1).values
    col_sums = table.sum(axis=0).values

    for i, row_total in enumerate(row_sums):
        for j, col_total in enumerate(col_sums):
            expected = row_total * col_total / total
            if expected > 0:
                chi2 += (table.values[i, j] - expected) ** 2 / expected

    k = min(table.shape) - 1

    if k <= 0:
        return 0.0

    return float(math.sqrt(chi2 / (total * k)))


def majority_accuracy(df, feature, label_column="label"):
    """Best accuracy obtainable from this feature alone."""

    part = df[[feature, label_column]].dropna()

    if part.empty:
        return None

    table = pd.crosstab(part[feature], part[label_column])

    return float(table.max(axis=1).sum() / table.values.sum())


def analyse(df, feature, name, label_column="label"):

    part = df[[feature, label_column]].dropna()
    part = part[part[feature].astype(str).str.strip() != ""]

    if part.empty or part[feature].nunique() < 1:
        return None

    table = pd.crosstab(part[feature], part[label_column])

    # A feature may legitimately have only one label present (e.g. generator
    # exists solely on fake rows), in which case the column is absent.
    def values_with(label_value):
        if label_value not in table.columns:
            return set()
        column = table[label_value]
        return set(table.index[column.to_numpy() > 0])

    real_values = values_with("real")
    fake_values = values_with("fake")
    shared = real_values & fake_values

    accuracy = float(table.max(axis=1).sum() / table.values.sum())

    return {
        "feature": name,
        "n": int(table.values.sum()),
        "distinct_values": int(part[feature].nunique()),
        "majority_accuracy": round(accuracy, 4),
        "cramers_v": round(cramers_v(table), 4),
        "perfectly_separates_label": len(shared) == 0,
        "values_shared_by_both_labels": len(shared),
        "verdict": (
            "PERFECT SHORTCUT" if len(shared) == 0
            else "STRONG" if accuracy >= STRONG
            else "weak/none"
        ),
        "table": {
            str(index): {str(c): int(v) for c, v in row.items()}
            for index, row in table.iterrows()
        },
    }


def bit_depth(codec):

    text = str(codec)

    if "PCM_16" in text:
        return "16-bit PCM"
    if "PCM_24" in text:
        return "24-bit PCM"
    if "PCM_32" in text or "FLOAT" in text:
        return "32-bit"
    if "MPEG_LAYER_III" in text:
        return "lossy (MP3)"
    if "OPUS" in text or "VORBIS" in text:
        return "lossy (Opus/Vorbis)"

    return "other"


def print_table(title, analysis):

    if not analysis:
        return

    print(f"\n  {title}")
    print(f"      majority-vote accuracy {analysis['majority_accuracy']:.4f} "
          f"| Cramer's V {analysis['cramers_v']:.4f} "
          f"| {analysis['verdict']}")

    header = sorted({
        column
        for row in analysis["table"].values()
        for column in row
    })

    print("      " + f"{'value':<34}" + "".join(f"{c:>8}" for c in header))

    for value, row in sorted(analysis["table"].items()):
        counts = "".join(f"{row.get(c, 0):>8}" for c in header)
        print("      " + f"{value[:33]:<34}" + counts)


def audit_frame(df, label, features):

    results = {}

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    for column, name in features:

        if column not in df.columns:
            continue

        analysis = analyse(df, column, name)

        if analysis:
            results[name] = analysis
            print_table(name, analysis)

    return results


def main():

    df = pd.read_csv(MASTER, dtype=str, keep_default_na=False)

    df["duration_num"] = pd.to_numeric(df["duration"], errors="coerce")
    df["duration_bin"] = pd.cut(
        df["duration_num"], bins=DURATION_BINS, labels=DURATION_LABELS,
        right=False,
    ).astype(str)

    df["original_bit_depth"] = df["codec"].map(bit_depth)
    df["processed_bit_depth"] = df["processed_codec"].map(bit_depth)

    audit = {
        "stage": "Phase 4 - shortcut audit",
        "metric_notes": {
            "majority_accuracy": (
                "accuracy of the best classifier that sees ONLY this feature; "
                "0.50 is chance for a balanced set, 1.00 means the feature "
                "IS the label"
            ),
            "cramers_v": "association strength 0..1",
            "strong_threshold": STRONG,
        },
    }

    original_features = [
        ("source_dataset", "source_dataset"),
        ("codec", "ORIGINAL codec"),
        ("original_sample_rate", "ORIGINAL sample_rate"),
        ("original_bit_depth", "ORIGINAL bit depth"),
        ("duration_bin", "duration bin"),
        ("gender_normalized", "gender_normalized"),
        ("generator", "generator"),
        ("voice", "voice"),
    ]

    canonical_features = [
        ("processed_codec", "CANONICAL codec"),
        ("processed_sample_rate", "CANONICAL sample_rate"),
        ("processed_bit_depth", "CANONICAL bit depth"),
        ("processed_channels", "CANONICAL channels"),
    ]

    audit["all_development_data"] = {
        "original_representation": audit_frame(
            df, "ORIGINAL REPRESENTATION - all 20k development samples",
            original_features,
        ),
        "canonical_representation": audit_frame(
            df, "CANONICAL REPRESENTATION (16 kHz mono PCM16)",
            canonical_features,
        ),
    }

    # ---- per language half ----

    audit["by_language_type"] = {}

    for language_type, part in df.groupby("language_type"):

        audit["by_language_type"][str(language_type)] = {
            "original": audit_frame(
                part, f"{language_type} - ORIGINAL", original_features
            ),
            "canonical": audit_frame(
                part, f"{language_type} - CANONICAL", canonical_features
            ),
        }

    # ---- augmentation ----

    if AUGMENTED.exists():

        aug = pd.read_csv(AUGMENTED, dtype=str, keep_default_na=False)

        aug["duration_num"] = pd.to_numeric(aug["duration"], errors="coerce")
        aug["duration_bin"] = pd.cut(
            aug["duration_num"], bins=DURATION_BINS, labels=DURATION_LABELS,
            right=False,
        ).astype(str)

        aug_features = [
            ("augmentation_condition", "augmentation_condition"),
            ("codec", "AUGMENTED codec"),
            ("sample_rate", "AUGMENTED sample_rate"),
            ("source_dataset", "source_dataset (augmented view)"),
            ("duration_bin", "duration bin (augmented view)"),
        ]

        audit["augmented_training_view"] = {
            "all": audit_frame(
                aug, "AUGMENTED TRAINING VIEW - all", aug_features
            ),
            "by_language_type": {
                str(language_type): audit_frame(
                    part,
                    f"AUGMENTED TRAINING VIEW - {language_type}",
                    [("augmentation_condition", "augmentation_condition")],
                )
                for language_type, part in aug.groupby("language_type")
            },
        }

        # Condition balance across every stratum that could leak.
        audit["augmentation_condition_balance"] = {
            "by_source_dataset": {
                str(source): {
                    str(condition): int(n)
                    for condition, n in
                    part["augmentation_condition"].value_counts().items()
                }
                for source, part in aug.groupby("source_dataset")
            },
            "by_label": {
                str(label): {
                    str(condition): int(n)
                    for condition, n in
                    part["augmentation_condition"].value_counts().items()
                }
                for label, part in aug.groupby("label")
            },
            "by_generator": {
                str(generator): {
                    str(condition): int(n)
                    for condition, n in
                    part["augmentation_condition"].value_counts().items()
                }
                for generator, part in aug.groupby("generator")
                if str(generator).strip()
            },
            "by_voice": {
                str(voice): {
                    str(condition): int(n)
                    for condition, n in
                    part["augmentation_condition"].value_counts().items()
                }
                for voice, part in aug.groupby("voice")
                if str(voice).strip()
            },
        }

        print("\n" + "=" * 70)
        print("AUGMENTATION CONDITION x LABEL (the key balance check)")
        print("=" * 70)

        for scope, part in [("ALL", aug)] + [
            (str(lt), p) for lt, p in aug.groupby("language_type")
        ]:
            table = pd.crosstab(
                part["augmentation_condition"], part["label"]
            )
            table["real_pct"] = (
                table.get("real", 0)
                / table.sum(axis=1) * 100
            ).round(1)
            print(f"\n  {scope}")
            print("      " + table.to_string().replace("\n", "\n      "))

    # ---- robustness set ----

    if ROBUSTNESS.exists():

        rob = pd.read_csv(ROBUSTNESS, dtype=str, keep_default_na=False)

        audit["robustness_set"] = {
            "total": int(len(rob)),
            "parents_all_from_test": bool(
                (rob["parent_split"] == "test").all()
            ),
            "condition_x_class": {
                f"{lt} | {lb}": {
                    str(condition): int(n)
                    for condition, n in
                    part["augmentation_condition"].value_counts().items()
                }
                for (lt, lb), part in rob.groupby(["language_type", "label"])
            },
            "unique_parents": int(rob["parent_sample_id"].nunique()),
        }

    # ---- honest summary ----

    def gather(section, keys):
        out = []
        for key in keys:
            item = section.get(key)
            if item:
                out.append({
                    "feature": item["feature"],
                    "majority_accuracy": item["majority_accuracy"],
                    "verdict": item["verdict"],
                })
        return out

    audit["summary"] = {
        "original": gather(
            audit["all_development_data"]["original_representation"],
            ["ORIGINAL codec", "ORIGINAL sample_rate", "ORIGINAL bit depth",
             "source_dataset", "duration bin", "gender_normalized"],
        ),
        "canonical": gather(
            audit["all_development_data"]["canonical_representation"],
            ["CANONICAL codec", "CANONICAL sample_rate",
             "CANONICAL bit depth", "CANONICAL channels"],
        ),
    }

    audit["remaining_confounds"] = [
        "Uniform WAV/PCM16 output does NOT mean codec bias is gone. Common "
        "Voice and Azure audio was MP3-encoded before we ever saw it, and "
        "that quantisation noise survives decoding, resampling and re-saving. "
        "The canonical codec column is now constant only because we made it "
        "constant.",

        "Resampling asymmetry: sources captured above 16 kHz (Common Voice "
        "32/48 kHz, IndicSynth and Hinglish fake 24 kHz) were low-passed on "
        "the way down, while natively-16 kHz sources (Kathbath, MUCS) were "
        "not. The resulting difference in high-band content is a residual, "
        "source-correlated cue that no metadata column exposes.",

        "source_dataset remains a perfect predictor of the label by "
        "construction - each source is entirely real or entirely fake. This "
        "is unavoidable in V1 and is exactly why source must never be a "
        "model input, and why held-out generalisation should be measured "
        "across sources.",

        "Duration still differs by source (Parler ~1.5 s vs MUCS ~5.1 s). "
        "Fixed-duration windows at training time are required so utterance "
        "length cannot be used.",

        "Gender is unavailable for MUCS and both Hinglish fake sources, and "
        "Kathbath is 100% female, so gender interacts with source and label "
        "in the Hindi half.",

        "Augmentation covers codec/bandwidth/level/clipping only. Real "
        "environmental noise, room impulse responses and physical replay are "
        "NOT included; noise_dir/rir_dir are hooks only, and physical replay "
        "cannot be synthesised honestly.",
    ]

    AUDIT_PATH.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("SUMMARY - how much can a single metadata column reveal?")
    print("=" * 70)
    print(f"\n  {'feature':<30}{'majority acc':>14}   verdict")

    for scope in ["original", "canonical"]:
        for item in audit["summary"][scope]:
            print(f"  {item['feature']:<30}"
                  f"{item['majority_accuracy']:>14.4f}   {item['verdict']}")

    print(f"\nAudit written: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
