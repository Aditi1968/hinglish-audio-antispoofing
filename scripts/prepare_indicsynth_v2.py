"""
Corrected IndicSynth Hindi collection (v2).

v0 was rejected: it streamed a couple of complete shards, so 2,500 XTTS clips
came from only 7 target speakers and the test split ended up with 0 XTTS.

v2 selects at ROW-GROUP level using metadata/indicsynth_hi_shard_index.csv,
which was built from parquet footers alone. The two generators are laid out
very differently in the Hindi config, so each needs its own strategy:

  xtts_v2   one target speaker per row group (101 speakers available).
            -> take one row group per speaker, ~25 clips each.

  freevc24  speakers are interleaved inside every row group, BUT shards
            0-25 all repeat the same 53 speakers; new speakers only appear
            from shard ~26 onward. The measured pool is ~101 speakers
            (scripts/probe_freevc_speakers.py). Collection therefore looks
            like it stalls around 1,590 clips before recovering - that is
            expected, not a failure.

Audio is written exactly as received. No resampling / denoising /
normalization / augmentation / silence removal.

Resumable: metadata is checkpointed after every row group, so an interruption
never costs more than the row group in flight.
"""

from collections import Counter, defaultdict
from pathlib import Path

import re

import pandas as pd
import pyarrow.parquet as pq

from huggingface_hub import HfFileSystem


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

REPO = "vdivyasharma/IndicSynth"

QUOTAS = {
    "xtts_v2": 2500,
    "freevc24": 2500,
}

TOTAL_TARGET = sum(QUOTAS.values())

# Max clips selected per (generator, target speaker).
# Raised automatically, and reported, if a generator cannot reach its quota.
INITIAL_SPEAKER_CAP = 30

SPLIT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

INDEX_PATH = Path("metadata/indicsynth_hi_shard_index.csv")

OUTPUT_DIR = Path("data/selected/indicsynth_hi_v2")
METADATA_PATH = Path("metadata/indicsynth_hi_v2.csv")

CHECKPOINT_PATH = Path("metadata/indicsynth_hi_v2_checkpoint.csv")
PROCESSED_PATH = Path("metadata/indicsynth_hi_v2_processed.csv")

SEED = 42


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def normalize_speaker_id(value):
    """1087.0 -> "1087"   /   1087 -> "1087"."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    value = str(value).strip()

    if re.fullmatch(r"\d+\.0+", value):
        value = value.split(".")[0]

    return value


def normalize_generator(value):

    if value is None:
        return None

    value = str(value).strip().lower()
    value = value.replace("-", "_").replace(" ", "_")

    aliases = {
        "xttsv2": "xtts_v2",
        "xtts_v2": "xtts_v2",
        "freevc": "freevc24",
        "freevc_24": "freevc24",
        "freevc24": "freevc24",
    }

    return aliases.get(value, value)


def extract_audio_bytes(record):
    """
    The audio column is a struct {bytes, path}. Note that the parquet footer
    reports its LEAF names ("bytes", "path"), so it looks top-level in the
    shard index while arrow actually nests it under "audio".
    """

    audio = record.get("audio")

    if isinstance(audio, dict):
        return audio.get("bytes")

    # Defensive: some readers flatten the struct.
    return record.get("bytes")


def spread(items, count):
    """Evenly spaced subset of `items`, preserving order."""

    if count >= len(items):
        return list(items)

    step = len(items) / count

    return [
        items[int(i * step)]
        for i in range(count)
    ]


# ---------------------------------------------------------
# ROW-GROUP PLANNING
# ---------------------------------------------------------

def plan_row_groups(index):
    """Decide which row groups to download, per generator."""

    plan = {}

    # ---- xtts_v2: one pure row group per distinct speaker ----

    xtts = index[
        (index["generator_min"] == "xtts_v2")
        & (index["generator_max"] == "xtts_v2")
        & index["pure_speaker"]
    ]

    # Prefer the largest row group for each speaker.
    best_per_speaker = (
        xtts.sort_values("num_rows", ascending=False)
        .groupby("speaker_min", as_index=False)
        .first()
        .sort_values(["shard", "row_group"])
    )

    plan["xtts_v2"] = best_per_speaker[
        ["shard", "row_group"]
    ].to_dict("records")

    # ---- freevc24: row groups spread across the shard range ----

    freevc = index[
        (index["generator_min"] == "freevc24")
        & (index["generator_max"] == "freevc24")
    ].sort_values(["shard", "row_group"])

    freevc_rgs = freevc[["shard", "row_group"]].to_dict("records")

    # Later shards hold the speakers that shards 0-25 do not, so the spread
    # must reach across the whole range rather than front-loading.
    plan["freevc24"] = spread(freevc_rgs, 160)

    return plan


# ---------------------------------------------------------
# CHECKPOINTING
# ---------------------------------------------------------

def load_checkpoint():
    """Return (rows, processed_row_groups) from a previous interrupted run."""

    rows = []
    processed = set()

    if CHECKPOINT_PATH.exists():

        done = pd.read_csv(CHECKPOINT_PATH, dtype=str, keep_default_na=False)

        keep = done["file_path"].map(
            lambda p: Path(p).exists() and Path(p).stat().st_size > 0
        )

        dropped = int((~keep).sum())

        if dropped:
            print(f"    checkpoint: dropping {dropped} rows with no audio")

        rows = done[keep].to_dict("records")

    if PROCESSED_PATH.exists():

        seen = pd.read_csv(PROCESSED_PATH, dtype=str, keep_default_na=False)

        processed = {
            (r["shard"], int(r["row_group"]))
            for _, r in seen.iterrows()
        }

    return rows, processed


def save_checkpoint(rows, processed):

    pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, index=False)

    pd.DataFrame(
        [{"shard": s, "row_group": rg} for s, rg in sorted(processed)]
    ).to_csv(PROCESSED_PATH, index=False)


def build_row(record, generator, sample_id, destination, shard, rg_index):

    transcript = record.get("TTS Transcript")

    return {
        "sample_id": sample_id,
        "file_path": str(destination),
        "language": "Hindi",
        "language_type": "Hindi",
        "label": "fake",
        "source_dataset": "IndicSynth",
        "generator": generator,
        "generator_family": (
            "voice_conversion"
            if generator == "freevc24"
            else "text_to_speech"
        ),
        "source_speaker_id": normalize_speaker_id(
            record.get("Source Speaker_ID")
        ),
        "target_speaker_id": normalize_speaker_id(
            record.get("Target Speaker ID")
        ),
        "gender": record.get("Gender"),
        "source_reference_audio": record.get("Source Reference Audio"),
        "target_reference_audio": record.get("Target Reference Audio"),
        "transcript": "" if transcript is None else transcript,
        "codec": "wav",
        "fake_start": 0,
        "fake_end": "",
        "has_timestamp_labels": False,
        "license": "CC BY-NC 4.0",
        "source_shard": shard,
        "source_row_group": rg_index,
        "split": "",
    }


# ---------------------------------------------------------
# COLLECTION
# ---------------------------------------------------------

def collect(plan):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows, processed = load_checkpoint()

    if rows:
        print(f"\nResuming from checkpoint: {len(rows):,} clips already held")

    fs = HfFileSystem()
    caps_used = {}

    for generator, row_groups in plan.items():

        quota = QUOTAS[generator]
        cap = INITIAL_SPEAKER_CAP

        # Restore counters for this generator from the checkpoint.
        existing = [r for r in rows if r["generator"] == generator]

        per_speaker = Counter(
            str(r["target_speaker_id"]) for r in existing
        )

        collected = len(existing)

        print(f"\n--- {generator}: target {quota:,} clips "
              f"from up to {len(row_groups)} row groups ---")

        # With one speaker per row group, spreading the quota evenly over all
        # available row groups maximises speaker count.
        if generator == "xtts_v2":
            cap = min(cap, max(1, -(-quota // len(row_groups))))

        print(f"    speaker cap: {cap}")

        if collected:
            print(f"    resuming at {collected:,} clips, "
                  f"{len(per_speaker)} speakers")

        caps_used[generator] = cap

        stalled = 0

        for entry in row_groups:

            if collected >= quota:
                break

            shard = entry["shard"]
            rg_index = int(entry["row_group"])

            if (shard, rg_index) in processed:
                continue

            path = f"datasets/{REPO}/{shard}"

            try:
                with fs.open(path, "rb") as handle:
                    table = pq.ParquetFile(handle).read_row_group(rg_index)
            except Exception as e:
                print(f"    skipping {shard} rg{rg_index}: {e}")
                continue

            before = collected

            for record in table.to_pylist():

                if collected >= quota:
                    break

                if normalize_generator(
                    record.get("Generative Model")
                ) != generator:
                    continue

                target_speaker = normalize_speaker_id(
                    record.get("Target Speaker ID")
                )

                if target_speaker is None:
                    continue

                if per_speaker[target_speaker] >= cap:
                    continue

                audio_bytes = extract_audio_bytes(record)

                if audio_bytes is None:
                    continue

                sample_id = f"is_hi_{generator}_{collected:05d}"
                destination = OUTPUT_DIR / f"{sample_id}.wav"

                with open(destination, "wb") as f:
                    f.write(audio_bytes)

                rows.append(
                    build_row(
                        record, generator, sample_id,
                        destination, shard, rg_index,
                    )
                )

                per_speaker[target_speaker] += 1
                collected += 1

            processed.add((shard, rg_index))
            save_checkpoint(rows, processed)

            gained = collected - before
            stalled = stalled + 1 if gained == 0 else 0

            note = ""
            if stalled and stalled % 10 == 0:
                # Expected for freevc24 in the early shards - see module docstring.
                note = f"  (no new clips for {stalled} row groups)"

            print(
                f"    {shard.split('/')[-1]} rg{rg_index:>2} | "
                f"{collected:>5,}/{quota:,} clips | "
                f"{len(per_speaker):>4} speakers{note}"
            )

        # Cap escalation only if the speaker pool is genuinely exhausted.
        while collected < quota and cap < 400:

            previous = cap
            cap = min(cap * 2, 400)

            print(
                f"\n    Speaker pool exhausted at cap {previous} "
                f"({collected:,}/{quota:,} clips, "
                f"{len(per_speaker)} speakers). Raising cap to {cap} "
                f"and re-reading row groups."
            )

            for entry in row_groups:

                if collected >= quota:
                    break

                shard = entry["shard"]
                rg_index = int(entry["row_group"])

                path = f"datasets/{REPO}/{shard}"

                try:
                    with fs.open(path, "rb") as handle:
                        table = pq.ParquetFile(handle).read_row_group(rg_index)
                except Exception:
                    continue

                for record in table.to_pylist():

                    if collected >= quota:
                        break

                    if normalize_generator(
                        record.get("Generative Model")
                    ) != generator:
                        continue

                    target_speaker = normalize_speaker_id(
                        record.get("Target Speaker ID")
                    )

                    if target_speaker is None:
                        continue

                    if per_speaker[target_speaker] >= cap:
                        continue

                    audio_bytes = extract_audio_bytes(record)

                    if audio_bytes is None:
                        continue

                    sample_id = f"is_hi_{generator}_{collected:05d}"
                    destination = OUTPUT_DIR / f"{sample_id}.wav"

                    if destination.exists():
                        continue

                    with open(destination, "wb") as f:
                        f.write(audio_bytes)

                    rows.append(
                        build_row(
                            record, generator, sample_id,
                            destination, shard, rg_index,
                        )
                    )

                    per_speaker[target_speaker] += 1
                    collected += 1

                save_checkpoint(rows, processed)

            caps_used[generator] = cap

        print(
            f"\n    {generator}: {collected:,}/{quota:,} clips, "
            f"{len(per_speaker)} unique target speakers, cap {caps_used[generator]}"
        )

        if collected < quota:
            raise RuntimeError(
                f"{generator}: only {collected}/{quota} clips collected."
            )

    return pd.DataFrame(rows), caps_used


# ---------------------------------------------------------
# SPLITTING
# ---------------------------------------------------------

def assign_splits(df):
    """
    Greedy speaker-group assignment.

    A target speaker is ONE indivisible group across both generators, so a
    speaker appearing under xtts_v2 and freevc24 lands wholly in one split.

    Priorities, in order:
      1. zero target-speaker leakage  (structural: we assign whole groups)
      2. both generators present in every split
      3. reasonable generator balance
      4. approximate 70/15/15
    """

    generators = sorted(df["generator"].unique())

    counts = defaultdict(Counter)

    for (speaker, generator), n in (
        df.groupby(["target_speaker_id", "generator"]).size().items()
    ):
        counts[speaker][generator] = n

    totals = {
        g: int((df["generator"] == g).sum())
        for g in generators
    }

    targets = {
        split: {
            g: totals[g] * frac
            for g in generators
        }
        for split, frac in SPLIT_FRACTIONS.items()
    }

    current = {
        split: Counter()
        for split in SPLIT_FRACTIONS
    }

    assignment = {}

    # Largest speaker groups first - they are the hardest to place well.
    speakers = sorted(
        counts,
        key=lambda s: -sum(counts[s].values())
    )

    for speaker in speakers:

        group = counts[speaker]

        best_split = None
        best_score = None

        for split in SPLIT_FRACTIONS:

            fit = sum(
                min(
                    group[g],
                    max(0.0, targets[split][g] - current[split][g])
                )
                for g in generators
            )

            overshoot = sum(
                max(
                    0.0,
                    current[split][g] + group[g] - targets[split][g]
                )
                for g in generators
            )

            score = (fit - 0.5 * overshoot, -sum(current[split].values()))

            if best_score is None or score > best_score:
                best_score = score
                best_split = split

        assignment[speaker] = best_split

        for g in generators:
            current[best_split][g] += group[g]

    # Priority 2 repair: if some generator is missing from a split, move the
    # smallest donor speaker group that supplies it and can be spared.
    for split in SPLIT_FRACTIONS:
        for g in generators:

            if current[split][g] > 0:
                continue

            donors = sorted(
                (
                    s for s in counts
                    if counts[s][g] > 0
                    and assignment[s] != split
                    and current[assignment[s]][g] > counts[s][g]
                ),
                key=lambda s: sum(counts[s].values())
            )

            if not donors:
                print(f"    WARNING: cannot repair {g} in {split}")
                continue

            donor = donors[0]
            source = assignment[donor]

            for gg in generators:
                current[source][gg] -= counts[donor][gg]
                current[split][gg] += counts[donor][gg]

            assignment[donor] = split

            print(
                f"    repair: moved speaker {donor} "
                f"from {source} to {split} to supply {g}"
            )

    return df["target_speaker_id"].map(assignment)


# ---------------------------------------------------------
# REPORTING
# ---------------------------------------------------------

def describe_concentration(series, title):

    print(f"\n{title}")
    print(f"    min    {series.min()}")
    print(f"    median {series.median()}")
    print(f"    mean   {round(series.mean(), 2)}")
    print(f"    max    {series.max()}")


def report(df, caps_used):

    print("\n=========================================")
    print("INDICSYNTH V2 VALIDATION")
    print("=========================================")

    print("\nTotal samples:", len(df))

    print("\nGenerator counts:")
    for g, n in df["generator"].value_counts().items():
        print(f"    {g} = {n}")

    speakers_by_gen = {
        g: set(d["target_speaker_id"])
        for g, d in df.groupby("generator")
    }

    xtts_spk = speakers_by_gen.get("xtts_v2", set())
    freevc_spk = speakers_by_gen.get("freevc24", set())

    print("\nUnique target speakers:")
    print("    total   ", df["target_speaker_id"].nunique())
    print("    xtts_v2 ", len(xtts_spk))
    print("    freevc24", len(freevc_spk))
    print("    shared  ", len(xtts_spk & freevc_spk))

    print("\nSpeaker cap actually used:")
    for g, cap in caps_used.items():
        print(f"    {g} = {cap}")

    print("\nPer split:")

    for split in ["train", "validation", "test"]:

        part = df[df["split"] == split]
        by_gen = part["generator"].value_counts()

        print(f"\n  {split}")
        print(f"    samples          {len(part)}")
        print(f"    unique speakers  {part['target_speaker_id'].nunique()}")
        print(f"    xtts_v2          {int(by_gen.get('xtts_v2', 0))}")
        print(f"    freevc24         {int(by_gen.get('freevc24', 0))}")

    print("\nGenerator x Split:")
    print(pd.crosstab(df["generator"], df["split"]))

    describe_concentration(
        df["target_speaker_id"].value_counts(),
        "Clips per target speaker (overall):"
    )

    for g, part in df.groupby("generator"):
        describe_concentration(
            part["target_speaker_id"].value_counts(),
            f"Clips per target speaker ({g}):"
        )

    print("\n-----------------------------------------")
    print("REQUIREMENTS")
    print("-----------------------------------------")

    failures = []

    if len(df) != TOTAL_TARGET:
        failures.append(f"total samples {len(df)} != {TOTAL_TARGET}")

    for g, quota in QUOTAS.items():
        n = int((df["generator"] == g).sum())
        if n != quota:
            failures.append(f"{g} count {n} != {quota}")

    for g in QUOTAS:
        for split in ["train", "validation", "test"]:
            n = int(((df["generator"] == g) & (df["split"] == split)).sum())
            status = "OK" if n > 0 else "FAIL"
            print(f"    {g:>9} {split:<11} = {n:>5}   {status}")
            if n == 0:
                failures.append(f"{g} has 0 samples in {split}")

    split_speakers = {
        split: set(df[df["split"] == split]["target_speaker_id"])
        for split in ["train", "validation", "test"]
    }

    print("\n    Speaker overlap:")

    for a, b in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = len(split_speakers[a] & split_speakers[b])
        status = "OK" if overlap == 0 else "FAIL"
        print(f"        {a} & {b} = {overlap}   {status}")
        if overlap:
            failures.append(f"speaker leakage {a}/{b}: {overlap}")

    print("\n    Speaker diversity targets:")

    for g, spk in [("xtts_v2", xtts_spk), ("freevc24", freevc_spk)]:
        n = len(spk)
        grade = (
            "EXCELLENT" if n >= 100
            else "PREFERRED" if n >= 75
            else "BELOW TARGET"
        )
        print(f"        {g:>9} = {n:>4} speakers   {grade}")
        if n < 75:
            failures.append(f"{g} only {n} speakers (< 75)")

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

    if not INDEX_PATH.exists():
        raise SystemExit(
            f"Missing {INDEX_PATH}. "
            "Run scripts/index_indicsynth_shards.py first."
        )

    index = pd.read_csv(INDEX_PATH)

    print("Planning row groups from shard index...")
    plan = plan_row_groups(index)

    for g, entries in plan.items():
        print(f"    {g}: {len(entries)} candidate row groups")

    df, caps_used = collect(plan)

    print("\nAssigning speaker-disjoint, generator-balanced splits...")
    df["split"] = assign_splits(df)

    df.to_csv(METADATA_PATH, index=False)

    passed = report(df, caps_used)

    print("\nMetadata saved to:", METADATA_PATH)
    print("Audio saved to:", OUTPUT_DIR)
    print("\nAudio was NOT resampled, denoised, normalized or trimmed.")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
