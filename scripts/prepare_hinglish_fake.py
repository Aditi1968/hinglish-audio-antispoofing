"""
FAKE Hinglish from nameissakthi/hindi-english-bilingual (Indic Parler-TTS, Rani).

LIVE INSPECTION RESULT - the README does not describe the usable data:

  README says "Hinglish (bi) = 10,094". That file count is real
  (9,995 root v8c_*.wav + 99 wavs/bi/), but those files are NOT all
  code-switched. By transcript script analysis:

      truly code-switched (Devanagari + Latin)   1,756
      pure Devanagari                            3,244
      pure Latin                                 5,094

  The `language` metadata field is also unreliable: 69 rows labelled "hi"
  and 1 labelled "en" are in fact code-switched, while thousands of rows
  inside the README's "bi" bucket are monolingual.

  So the only defensible definition of Hinglish here is the transcript
  itself, which yields 1,756 usable clips - not 5,000.

SELECTION_MODE controls what counts as Hinglish:

  "strict"  transcript contains BOTH Devanagari and Latin script.
            1,756 clips. Verifiable, no monolingual contamination.

  "bi_label" everything the dataset itself calls Hinglish (root v8c_* +
            wavs/bi/). 10,094 clips, but includes pure Hindi and pure
            English audio, which this project explicitly forbids.

Audio is copied byte-for-byte. No resampling / denoising / normalization /
trimming / augmentation.
"""

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

import json
import random
import re
import shutil
import time

import pandas as pd
import soundfile as sf

from huggingface_hub import hf_hub_download


REPO = "nameissakthi/hindi-english-bilingual"

SELECTION_MODE = "strict"

# Hard ceiling; the strict pool is smaller than this and that is the finding.
TARGET = 5000

OUTPUT_DIR = Path("data/selected/hinglish_fake")
METADATA_PATH = Path("metadata/hinglish_fake.csv")
SUMMARY_PATH = Path("metadata/hinglish_fake_summary.json")

SPLIT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

GENERATOR = "Indic_Parler_TTS"
VOICE = "Rani"
LICENSE = "CC BY 4.0"

SEED = 42
WORKERS = 3

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def is_code_switched(text):
    return bool(DEVANAGARI.search(text)) and bool(LATIN.search(text))


def normalize_text(text):
    """For duplicate detection only - never written to the audio."""
    return re.sub(r"\s+", " ", str(text).strip().lower())


# ---------------------------------------------------------
# SELECTION
# ---------------------------------------------------------

def load_rows():

    path = hf_hub_download(REPO, "metadata.jsonl", repo_type="dataset")

    rows = [
        json.loads(line)
        for line in open(path, encoding="utf-8")
        if line.strip()
    ]

    print(f"metadata.jsonl rows: {len(rows):,}")

    for row in rows:
        row["code_switched"] = is_code_switched(row["text"])
        row["bucket"] = (
            "root_v8c" if "/" not in row["file_name"]
            else row["file_name"].split("/")[1]
        )

    print("\nscript analysis of transcripts:")
    kinds = Counter(
        "code_switched" if r["code_switched"]
        else ("devanagari" if DEVANAGARI.search(r["text"]) else "latin")
        for r in rows
    )
    for k, v in kinds.most_common():
        print(f"    {k:<16} {v:,}")

    print("\ndataset's own 'language' field vs transcript reality:")
    print("    rows labelled 'bi'      :",
          sum(1 for r in rows if r["language"] == "bi"))
    print("    rows actually mixed     :",
          sum(1 for r in rows if r["code_switched"]))
    print("    mixed but NOT labelled bi:",
          sum(1 for r in rows if r["code_switched"] and r["language"] != "bi"))

    return rows


def select(rows):

    if SELECTION_MODE == "strict":
        pool = [r for r in rows if r["code_switched"]]
    elif SELECTION_MODE == "bi_label":
        pool = [r for r in rows if r["bucket"] in ("root_v8c", "bi")]
    else:
        raise SystemExit(f"unknown SELECTION_MODE {SELECTION_MODE!r}")

    print(f"\nSELECTION_MODE = {SELECTION_MODE!r} -> pool {len(pool):,}")

    # Drop duplicate transcripts: identical text from one TTS voice is a
    # near-duplicate waveform and would leak across splits.
    seen = set()
    unique = []
    duplicates = 0

    for row in sorted(pool, key=lambda r: r["file_name"]):

        key = normalize_text(row["text"])

        if key in seen:
            duplicates += 1
            continue

        seen.add(key)
        unique.append(row)

    print(f"    duplicate transcripts removed: {duplicates:,}")
    print(f"    unique usable clips:           {len(unique):,}")

    if len(unique) < TARGET:
        print(
            f"\n    *** SHORTFALL: {len(unique):,} available, "
            f"{TARGET:,} requested ***"
        )

    rng = random.Random(SEED)
    rng.shuffle(unique)

    return unique[:TARGET]


# ---------------------------------------------------------
# COLLECTION
# ---------------------------------------------------------

def fetch(job):

    index, row = job

    sample_id = f"hf_hi_en_{index:05d}"
    destination = OUTPUT_DIR / f"{sample_id}.wav"

    # Resume: a previous run already fetched this one.
    if destination.exists() and destination.stat().st_size > 0:
        try:
            return index, row, (destination, sf.info(str(destination))), None
        except Exception:
            destination.unlink(missing_ok=True)

    cached = None
    last_error = None

    # The hub rate-limits (429) on bursts of small-file requests.
    for attempt in range(6):

        try:
            cached = hf_hub_download(
                REPO, row["file_name"], repo_type="dataset"
            )
            break
        except Exception as e:
            last_error = e
            time.sleep(min(2 ** attempt, 30) + random.random())

    if cached is None:
        return index, row, None, f"download failed: {last_error}"

    # Byte-for-byte copy - no decode, no re-encode.
    shutil.copyfile(cached, destination)

    try:
        info = sf.info(str(destination))
    except Exception as e:
        destination.unlink(missing_ok=True)
        return index, row, None, f"unreadable audio: {e}"

    return index, row, (destination, info), None


def collect(picked):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(picked):,} clips...\n")

    rows = []
    failures = 0
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:

        for index, row, result, error in pool.map(
            fetch, list(enumerate(picked))
        ):

            done += 1

            if error:
                failures += 1
                print(f"    {row['file_name']}: {error}")
                continue

            destination, info = result

            rows.append({
                "sample_id": destination.stem,
                "file_path": str(destination),
                "language": "Hindi-English",
                "language_type": "Hinglish",
                "label": "fake",
                "source_dataset": "nameissakthi_hindi_english_bilingual",
                "generator": GENERATOR,
                "voice": VOICE,
                "source_file_id": row["file_name"],
                "transcript": row["text"],
                "duration": round(info.duration, 4),
                "original_sample_rate": info.samplerate,
                "codec": f"{info.format}/{info.subtype}",
                "channels": info.channels,
                "declared_language": row.get("language"),
                "lid": row.get("lid"),
                "split": "",
                "license": LICENSE,
            })

            if done % 250 == 0:
                print(f"    {done:,}/{len(picked):,}")

    print(f"\ncollected {len(rows):,} | failures {failures}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# SPLITTING
# ---------------------------------------------------------

def assign_splits(df):
    """
    This is ONE synthetic voice. There is no speaker grouping to preserve,
    so we do NOT pretend to build a speaker-disjoint split.

    The strongest grouping actually available is the transcript: identical
    or near-identical text rendered by the same voice would produce
    near-duplicate waveforms. Duplicate transcripts were already removed,
    so a seeded random split over distinct transcripts is honest here.
    """

    rng = random.Random(SEED)

    order = list(df.index)
    rng.shuffle(order)

    n = len(order)
    n_train = int(n * SPLIT_FRACTIONS["train"])
    n_val = int(n * SPLIT_FRACTIONS["validation"])

    assignment = {}

    for position, index in enumerate(order):
        if position < n_train:
            assignment[index] = "train"
        elif position < n_train + n_val:
            assignment[index] = "validation"
        else:
            assignment[index] = "test"

    return pd.Series(assignment).reindex(df.index)


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate(df, requested):

    print("\n" + "=" * 55)
    print("HINGLISH FAKE - VALIDATION")
    print("=" * 55)

    failures = []

    def check(condition, message):
        print(f"    {'OK  ' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    print("\nCOUNTS")
    check(len(df) == requested,
          f"total = {len(df)} (requested {requested})")
    check(bool((df["label"] == "fake").all()),
          f"all label = fake ({int((df['label'] == 'fake').sum())}/{len(df)})")
    check(bool((df["language_type"] == "Hinglish").all()),
          f"all language_type = Hinglish "
          f"({int((df['language_type'] == 'Hinglish').sum())}/{len(df)})")

    print("\nFILES")

    paths = df["file_path"].map(Path)
    exists = paths.map(lambda p: p.exists())
    sizes = paths.map(lambda p: p.stat().st_size if p.exists() else 0)

    check(int(exists.sum()) == len(df),
          f"audio files exist = {int(exists.sum())}/{len(df)}")
    check(int((sizes == 0).sum()) == 0,
          f"zero-byte files = {int((sizes == 0).sum())}")
    check(int(df["sample_id"].duplicated().sum()) == 0,
          f"duplicate sample IDs = {int(df['sample_id'].duplicated().sum())}")
    check(int(df["file_path"].duplicated().sum()) == 0,
          f"duplicate paths = {int(df['file_path'].duplicated().sum())}")

    print("\nCODE-SWITCHING VERIFICATION")
    mixed = df["transcript"].map(is_code_switched)
    check(bool(mixed.all()),
          f"transcripts containing both scripts = {int(mixed.sum())}/{len(df)}")

    print("\nSPLIT COUNTS")
    print(df["split"].value_counts().to_string())

    print("\nDURATION STATS")
    d = df["duration"]
    print(f"    total hours  {d.sum() / 3600:.2f}")
    print(f"    min          {d.min():.2f}")
    print(f"    median       {d.median():.2f}")
    print(f"    mean         {d.mean():.2f}")
    print(f"    max          {d.max():.2f}")

    print("\nAUDIO FORMAT")
    print("    original_sample_rate:")
    print(df["original_sample_rate"].value_counts().to_string())
    print("\n    codec:")
    print(df["codec"].value_counts().to_string())
    print("\n    channels:")
    print(df["channels"].value_counts().to_string())
    print(f"\n    bytes on disk: {sizes.sum() / 1e9:.3f} GB")

    print("\nVOICES / SPEAKERS")
    print(f"    voice field      : {sorted(df['voice'].unique())}")
    print(f"    generator        : {sorted(df['generator'].unique())}")
    print("    unique speakers  : 1 (single synthetic voice - no diversity)")

    print("\nTRANSCRIPTS")
    print(f"    present          {int(df['transcript'].notna().sum())}/{len(df)}")
    print(f"    unique           {df['transcript'].nunique()}")

    print("\nDECLARED LANGUAGE FIELD (upstream, unreliable)")
    print(df["declared_language"].value_counts(dropna=False).to_string())

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

    rows = load_rows()
    picked = select(rows)

    df = collect(picked)
    df["split"] = assign_splits(df)

    df = df.sort_values("sample_id").reset_index(drop=True)
    df.to_csv(METADATA_PATH, index=False)

    passed = validate(df, len(picked))

    summary = {
        "dataset": "Hinglish FAKE (Indic Parler-TTS / Rani)",
        "source_repo": REPO,
        "selection_mode": SELECTION_MODE,
        "language": "Hindi-English",
        "language_type": "Hinglish",
        "label": "fake",
        "generator": GENERATOR,
        "voice": VOICE,
        "requested_target": TARGET,
        "actually_available": int(len(df)),
        "total_samples": int(len(df)),
        "split_counts": {
            str(k): int(v) for k, v in df["split"].value_counts().items()
        },
        "duration_hours": round(float(df["duration"].sum() / 3600), 2),
        "unique_voices": 1,
        "license": LICENSE,
        "limitations": [
            "Single synthetic voice (Rani) and a single generator "
            "(Indic Parler-TTS). This is a weak V1 fake source.",
            "NOT valid evidence of unseen-speaker generalization.",
            "NOT valid evidence of unseen-generator generalization.",
            "Splits are random over distinct transcripts, NOT "
            "speaker-disjoint - there is only one speaker to be disjoint on.",
            "Upstream README claims 10,094 Hinglish utterances; only 1,756 "
            "are genuinely code-switched by transcript analysis.",
            "Upstream 'language' field is unreliable (69 'hi' and 1 'en' "
            "rows are actually code-switched).",
        ],
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nMetadata:", METADATA_PATH)
    print("Summary: ", SUMMARY_PATH)
    print("Audio:   ", OUTPUT_DIR)
    print("\nAudio copied byte-for-byte; not resampled, denoised, "
          "normalized, trimmed or augmented.")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
