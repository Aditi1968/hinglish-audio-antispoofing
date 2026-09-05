"""
SEA-Spoof Hindi External Evaluation V1.

Collects exactly 2,000 Hindi clips from SEA-Spoof's OFFICIAL `evaluation`
split: 1,000 bonafide (real) and 1,000 spoof (fake).

This is HELD-OUT EXTERNAL EVALUATION ONLY. It must never be mixed into the
20k training dataset, which is why it lives under data/evaluation/ and
metadata/evaluation/ rather than data/selected/.

Selection notes:
  - the 1,000 fakes are spread round-robin across the available spoof types
    and source models rather than taken as the first 1,000 rows
  - speaker diversity is maximised only if a real speaker field is populated;
    speaker IDs are never invented
  - audio bytes are written exactly as stored upstream (FLAC), with no
    resampling, denoising, normalization, trimming, conversion or augmentation

The collector is resumable: completed samples are appended to a checkpoint
after every row group, and a rerun skips whatever is already on disk.

Requires scripts/test_seaspoof.py to have cached the evaluation index.
"""

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

from huggingface_hub import HfFileSystem


REPO = "Jack-ppkdczgx/SEA-Spoof"
SOURCE_SPLIT = "evaluation"

QUOTAS = {
    "real": 1000,
    "fake": 1000,
}

LABEL_MAP = {
    "bonafide": "real",
    "spoof": "fake",
}

# Upstream corpora barred from the REAL half, to keep this external benchmark
# independent of the training data. Common Voice Hindi is already a training
# source (Dataset 1), so its bonafide clips cannot appear here.
#
# The surviving bonafide source is `indic_tts`, verified as genuine human
# speech rather than TTS output:
#   - bonafide rows carry source_model='bonafide' with no generator attributed
#   - SEA-Spoof separately labels audio SYNTHESIZED from that corpus as spoof,
#     under explicit generator names (hindi_indic_tts_xtts-v2, ...)
# Note the naming trap: `indic-tts` also appears as a SPOOF source_model
# (hindi_indic_indic-tts). Selection keys on source_dataset + label, never on
# a substring match.
EXCLUDED_REAL_UPSTREAM = {"common_voice"}

INDEX_PATH = Path("metadata/evaluation/seaspoof_eval_index.csv")

OUTPUT_DIR = Path("data/evaluation/seaspoof_hi")
METADATA_PATH = Path("metadata/evaluation/seaspoof_hi.csv")
CHECKPOINT_PATH = Path("metadata/evaluation/seaspoof_hi_checkpoint.csv")

LICENSE_NOTE = (
    "SEA-Spoof; gated repo (manual approval required); "
    "license: other - see repo LICENSE; research use"
)

SEED = 42


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def blank_to_none(value):
    """SEA-Spoof uses empty strings rather than nulls for missing fields."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    return text if text else None


def has_usable_speakers(frame):
    """True only if a real speaker field is actually populated."""

    if "speaker_or_voice" not in frame:
        return False

    values = frame["speaker_or_voice"].map(blank_to_none)

    return values.notna().sum() > 0


def round_robin(groups, quota):
    """
    Take one item from each group in turn until the quota is met.

    Guarantees the rarest spoof types are represented instead of being
    crowded out by whichever type happens to appear first.
    """

    pools = {
        key: list(items)
        for key, items in groups.items()
        if items
    }

    picked = []

    while pools and len(picked) < quota:

        for key in sorted(pools):

            if len(picked) >= quota:
                break

            pool = pools[key]

            if pool:
                picked.append(pool.pop(0))

        pools = {k: v for k, v in pools.items() if v}

    return picked


def diversified_pick(frame, quota, group_columns, spread_column=None):
    """
    Pick `quota` rows spread as evenly as possible across group_columns.

    Within each group rows are ordered by `spread_column` (e.g. speaker) in
    round-robin too, so no single speaker or shard dominates a group.
    """

    frame = frame.sort_values(["shard", "row_group", "row_in_group"])

    groups = defaultdict(list)

    for _, row in frame.iterrows():

        key = tuple(
            str(blank_to_none(row.get(c)) or "unknown")
            for c in group_columns
        )

        groups[key].append(row)

    # Interleave by spread_column inside each group.
    if spread_column and spread_column in frame:

        for key, rows in groups.items():

            by_value = defaultdict(list)

            for row in rows:
                by_value[
                    blank_to_none(row.get(spread_column)) or "unknown"
                ].append(row)

            interleaved = []
            pools = {k: list(v) for k, v in by_value.items()}

            while pools:
                for value in sorted(pools):
                    if pools[value]:
                        interleaved.append(pools[value].pop(0))
                pools = {k: v for k, v in pools.items() if v}

            groups[key] = interleaved

    return round_robin(groups, quota)


# ---------------------------------------------------------
# SELECTION
# ---------------------------------------------------------

def build_selection(index):

    hi = index[index["language"] == "hi"].copy()

    print(f"Hindi rows in official {SOURCE_SPLIT} split: {len(hi):,}")

    speakers_usable = has_usable_speakers(hi)

    print("Usable speaker field:", "YES" if speakers_usable else "NO")

    if not speakers_usable:
        print("    -> speaker_id will be left blank; IDs are NOT invented")

    selection = {}

    for original_label, project_label in LABEL_MAP.items():

        part = hi[hi["label"] == original_label]

        if project_label == "real" and EXCLUDED_REAL_UPSTREAM:

            before = len(part)
            part = part[~part["source_dataset"].isin(EXCLUDED_REAL_UPSTREAM)]

            print(
                f"\nreal: excluded {sorted(EXCLUDED_REAL_UPSTREAM)} -> "
                f"{before:,} bonafide rows reduced to {len(part):,}"
            )
            print("    remaining upstream sources:",
                  sorted(part["source_dataset"].unique()))

        quota = QUOTAS[project_label]

        if len(part) < quota:
            raise RuntimeError(
                f"Only {len(part)} usable {original_label} Hindi rows "
                f"available, need {quota}."
            )

        if project_label == "fake":
            # Spread fakes across every spoof type / category / model.
            group_columns = ["spoof_type", "category", "source_model"]
            # Vendor voice names exist only on part of the spoof half.
            spread_column = "speaker_or_voice" if speakers_usable else None
        else:
            # Bonafide has no spoof type and no speaker IDs at all, so the
            # best available proxy for spread is position in the corpus.
            group_columns = ["source_model", "source_dataset"]
            spread_column = "row_group"

        picked = diversified_pick(
            part,
            quota,
            group_columns,
            spread_column=spread_column,
        )

        print(
            f"\n{project_label}: selected {len(picked):,}/{quota:,} "
            f"grouped by {group_columns}"
        )

        selection[project_label] = picked

    return selection, speakers_usable


# ---------------------------------------------------------
# COLLECTION
# ---------------------------------------------------------

def purge_excluded_reals():
    """
    Drop every previously collected REAL sample if any of them came from a
    now-excluded upstream corpus.

    All reals are purged rather than just the offending ones, because
    sample_id is positional within the label: re-selecting shifts which row
    each ss_hi_real_XXXXX refers to. Fakes are untouched.
    """

    if not CHECKPOINT_PATH.exists():
        return

    done = pd.read_csv(CHECKPOINT_PATH, dtype=str, keep_default_na=False)

    if "upstream_dataset" not in done.columns:
        return

    offending = done[
        (done["label"] == "real")
        & (done["upstream_dataset"].isin(EXCLUDED_REAL_UPSTREAM))
    ]

    if offending.empty:
        return

    reals = done[done["label"] == "real"]

    print(
        f"Purging {len(reals):,} previously collected real samples "
        f"({len(offending):,} from excluded upstream sources)"
    )

    removed = 0

    for path in reals["file_path"]:
        target = Path(path)
        if target.exists():
            target.unlink()
            removed += 1

    print(f"    deleted {removed:,} audio files")

    done[done["label"] != "real"].to_csv(CHECKPOINT_PATH, index=False)


def load_checkpoint():

    if not CHECKPOINT_PATH.exists():
        return pd.DataFrame()

    done = pd.read_csv(CHECKPOINT_PATH, dtype=str)

    # Only trust rows whose audio actually survived on disk.
    keep = done["file_path"].map(
        lambda p: Path(p).exists() and Path(p).stat().st_size > 0
    )

    dropped = int((~keep).sum())

    if dropped:
        print(f"    checkpoint: discarding {dropped} rows with missing audio")

    return done[keep].reset_index(drop=True)


def collect(selection, speakers_usable):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    done = load_checkpoint()

    already = set(done["sample_id"]) if len(done) else set()

    if already:
        print(f"\nResuming: {len(already):,} samples already collected")

    rows = done.to_dict("records") if len(done) else []

    fs = HfFileSystem()

    # Group every outstanding target by row group so each is fetched once.
    pending = defaultdict(list)

    for project_label, picked in selection.items():
        for position, row in enumerate(picked):

            sample_id = f"ss_hi_{project_label}_{position:05d}"

            if sample_id in already:
                continue

            pending[(row["shard"], int(row["row_group"]))].append(
                (sample_id, project_label, row)
            )

    if not pending:
        print("Nothing left to collect.")
        return pd.DataFrame(rows)

    print(f"\nRow groups to fetch: {len(pending):,}\n")

    fetched = 0

    for (shard, rg_index), targets in sorted(pending.items()):

        path = f"datasets/{REPO}/{shard}"

        try:
            with fs.open(path, "rb") as handle:
                table = pq.ParquetFile(handle).read_row_group(rg_index)
        except Exception as e:
            print(f"    FAILED {shard} rg{rg_index}: {e}")
            continue

        records = table.to_pylist()

        for sample_id, project_label, meta in targets:

            position = int(meta["row_in_group"])

            if position >= len(records):
                print(f"    row {position} missing in {shard} rg{rg_index}")
                continue

            record = records[position]

            # Guard against any row-group offset drift.
            if blank_to_none(record.get("row_id")) != blank_to_none(
                meta.get("row_id")
            ):
                matches = [
                    r for r in records
                    if blank_to_none(r.get("row_id"))
                    == blank_to_none(meta.get("row_id"))
                ]
                if not matches:
                    print(f"    row_id mismatch for {sample_id}, skipping")
                    continue
                record = matches[0]

            audio = record.get("audio") or {}
            audio_bytes = audio.get("bytes")

            if not audio_bytes:
                print(f"    no audio bytes for {sample_id}, skipping")
                continue

            # Preserve the upstream container (FLAC); never convert.
            suffix = Path(audio.get("path") or "").suffix or ".flac"
            destination = OUTPUT_DIR / f"{sample_id}{suffix}"

            with open(destination, "wb") as f:
                f.write(audio_bytes)

            try:
                info = sf.info(str(destination))
                duration = round(info.duration, 4)
                original_sample_rate = info.samplerate
                codec = (info.format or suffix.lstrip(".")).lower()
            except Exception as e:
                print(f"    unreadable audio for {sample_id}: {e}")
                destination.unlink(missing_ok=True)
                continue

            original_label = record.get("label")

            rows.append({
                "sample_id": sample_id,
                "file_path": str(destination),
                "language": "Hindi",
                "label": project_label,
                "original_label": original_label,
                "source_dataset": "SEA-Spoof",
                "source_split": SOURCE_SPLIT,
                "project_role": "external_evaluation",
                "utterance_id": blank_to_none(record.get("utterance_id")),
                "speaker_id": (
                    blank_to_none(record.get("speaker_or_voice"))
                    if speakers_usable else None
                ),
                "spoof_type": blank_to_none(record.get("spoof_type")),
                "spoof_category": blank_to_none(record.get("category")),
                "transcript": record.get("text"),
                "duration": duration,
                "original_sample_rate": original_sample_rate,
                "codec": codec,
                "fake_start": 0 if project_label == "fake" else "",
                "fake_end": duration if project_label == "fake" else "",
                "has_timestamp_labels": False,
                "license_or_access_note": LICENSE_NOTE,
                "row_id": blank_to_none(record.get("row_id")),
                "source_model": blank_to_none(record.get("source_model")),
                "upstream_dataset": blank_to_none(record.get("source_dataset")),
            })

            fetched += 1

        # Checkpoint after every row group so an interruption loses nothing.
        pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, index=False)

        counts = Counter(r["label"] for r in rows)

        print(
            f"    {shard.split('/')[-1]} rg{rg_index:>3} | "
            f"real {counts['real']:>4}/1000 | "
            f"fake {counts['fake']:>4}/1000"
        )

    print(f"\nNewly fetched this run: {fetched:,}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate(df):

    print("\n" + "=" * 55)
    print("SEA-SPOOF HINDI EXTERNAL EVALUATION - VALIDATION")
    print("=" * 55)

    failures = []

    def check(condition, message):
        print(f"    {'OK  ' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    total = len(df)
    real = int((df["label"] == "real").sum())
    fake = int((df["label"] == "fake").sum())

    print("\nCOUNTS")
    check(total == 2000, f"total = {total} (expected 2000)")
    check(real == 1000, f"real = {real} (expected 1000)")
    check(fake == 1000, f"fake = {fake} (expected 1000)")

    hindi = int((df["language"] == "Hindi").sum())
    external = int((df["project_role"] == "external_evaluation").sum())

    check(hindi == 2000, f"language Hindi = {hindi} (expected 2000)")
    check(
        external == 2000,
        f"project_role external_evaluation = {external} (expected 2000)"
    )

    print("\nFILES")

    paths = df["file_path"].map(Path)
    exists = paths.map(lambda p: p.exists())
    sizes = paths.map(lambda p: p.stat().st_size if p.exists() else 0)

    check(int(exists.sum()) == 2000, f"audio files present = {int(exists.sum())}")
    check(
        int((sizes == 0).sum()) == 0,
        f"zero-byte files = {int((sizes == 0).sum())}"
    )

    dup_ids = int(df["sample_id"].duplicated().sum())
    dup_paths = int(df["file_path"].duplicated().sum())

    check(dup_ids == 0, f"duplicate sample IDs = {dup_ids}")
    check(dup_paths == 0, f"duplicate output paths = {dup_paths}")

    print("\nTRAINING-SOURCE INDEPENDENCE (real half)")

    reals_upstream = df[df["label"] == "real"]["upstream_dataset"]

    print(reals_upstream.value_counts(dropna=False).to_string())

    for excluded in sorted(EXCLUDED_REAL_UPSTREAM):
        n = int((reals_upstream == excluded).sum())
        check(n == 0, f"real samples from '{excluded}' = {n} (expected 0)")

    print("\nLABEL MAPPING")
    print(pd.crosstab(df["original_label"], df["label"]))

    print("\nSOURCE SPLIT USED")
    print(df["source_split"].value_counts().to_string())

    print("\nSPOOF TYPE DISTRIBUTION (fake only)")
    fakes = df[df["label"] == "fake"]
    print(fakes["spoof_type"].value_counts(dropna=False).to_string())

    print("\nSPOOF CATEGORY DISTRIBUTION (fake only)")
    print(fakes["spoof_category"].value_counts(dropna=False).to_string())

    if "source_model" in df:
        print("\nSOURCE MODEL DISTRIBUTION (fake only, top 15)")
        print(fakes["source_model"].value_counts(dropna=False)
              .head(15).to_string())

    print("\nSPEAKERS")
    speaker = df["speaker_id"].map(blank_to_none)

    if speaker.notna().any():
        print("    unique speakers:", speaker.nunique())
        print("    rows with speaker ID:", int(speaker.notna().sum()))
    else:
        print("    no speaker IDs upstream; field left blank (not invented)")

    print("\nAUDIO / DURATION STATS")

    duration = pd.to_numeric(df["duration"], errors="coerce")

    print(f"    total hours   {duration.sum() / 3600:.2f}")
    print(f"    min seconds   {duration.min():.2f}")
    print(f"    median        {duration.median():.2f}")
    print(f"    mean          {duration.mean():.2f}")
    print(f"    max           {duration.max():.2f}")

    print("\n    codec:")
    print(df["codec"].value_counts().to_string())

    print("\n    original_sample_rate:")
    print(df["original_sample_rate"].value_counts().to_string())

    print(f"\n    bytes on disk: {sizes.sum() / 1e9:.2f} GB")

    print("\nTIMESTAMPS")
    fake_ok = (
        (pd.to_numeric(fakes["fake_start"], errors="coerce") == 0).all()
        and pd.to_numeric(fakes["fake_end"], errors="coerce").notna().all()
    )
    reals = df[df["label"] == "real"]
    real_ok = (
        reals["fake_start"].map(lambda v: blank_to_none(v) is None).all()
        and reals["fake_end"].map(lambda v: blank_to_none(v) is None).all()
    )

    check(bool(fake_ok), "fake rows have fake_start=0 and fake_end=duration")
    check(bool(real_ok), "real rows have blank fake_start/fake_end")
    check(
        bool((~df["has_timestamp_labels"].astype(str)
              .str.lower().eq("true")).all()),
        "has_timestamp_labels is false everywhere"
    )

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
            f"Missing {INDEX_PATH}. Run scripts/test_seaspoof.py first."
        )

    index = pd.read_csv(INDEX_PATH, dtype=str, keep_default_na=False)

    purge_excluded_reals()

    selection, speakers_usable = build_selection(index)

    df = collect(selection, speakers_usable)

    column_order = [
        "sample_id", "file_path", "language", "label", "original_label",
        "source_dataset", "source_split", "project_role",
        "utterance_id", "speaker_id",
        "spoof_type", "spoof_category", "transcript",
        "duration", "original_sample_rate", "codec",
        "fake_start", "fake_end", "has_timestamp_labels",
        "license_or_access_note",
        "row_id", "source_model", "upstream_dataset",
    ]

    df = df[[c for c in column_order if c in df.columns]]
    df = df.sort_values("sample_id").reset_index(drop=True)

    df.to_csv(METADATA_PATH, index=False)

    passed = validate(df)

    print("\nMetadata:", METADATA_PATH)
    print("Audio:   ", OUTPUT_DIR)
    print("\nAudio was NOT resampled, denoised, normalized, trimmed,")
    print("converted or augmented. This dataset is external evaluation only")
    print("and must not enter the 20k training set.")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
