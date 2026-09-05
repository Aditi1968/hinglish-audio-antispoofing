"""
Top up the FAKE Hinglish set from 1,756 to exactly 5,000.

Source A (already collected by scripts/prepare_hinglish_fake.py):
    nameissakthi/hindi-english-bilingual
    Indic Parler-TTS, voice "Rani", 1,756 clips, median 1.49 s

Source B (added here):
    lingamvamshikrishnareddy/octopus-tts-hinglish
    Microsoft Azure / Edge Neural TTS, voices hi-IN-MadhurNeural and
    hi-IN-SwaraNeural, 13,656 code-switched clips across 138 domains,
    ~4.5 s typical.

Adding source B fixes three weaknesses of the V1 primary source:
    - one generator          -> two generators
    - one voice              -> three voices
    - median 1.49 s clips    -> source B is ~4.5 s, closer to MUCS reals

Structure notes discovered live (nothing assumed):
    - hi/wav.scp stores the author's ABSOLUTE server paths
      (/root/chatbot/octo/...); only the "hi/wavs/..." tail is repo-relative
    - files are named *.wav but are actually MP3 (MPEG_LAYER_III) at 24 kHz.
      They are copied byte-for-byte and the true codec is recorded.
    - every transcript is rendered by BOTH voices, so transcript is a real
      grouping variable and both renditions must land in the same split.
    - the repo has NO README and states NO license.

Audio is copied byte-for-byte. No resampling / denoising / normalization /
trimming / augmentation.
"""

from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from pathlib import Path

import json
import random
import re
import shutil
import time

import pandas as pd
import soundfile as sf

from huggingface_hub import HfApi, hf_hub_download


REPO_B = "lingamvamshikrishnareddy/octopus-tts-hinglish"

TARGET_TOTAL = 5000

OUTPUT_DIR = Path("data/selected/hinglish_fake")
METADATA_PATH = Path("metadata/hinglish_fake.csv")
SUMMARY_PATH = Path("metadata/hinglish_fake_summary.json")

SPLIT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

GENERATOR_B = "Azure_Edge_Neural_TTS"
LICENSE_B = "not stated upstream (repo has no README/license file)"

SEED = 42
WORKERS = 3

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def is_code_switched(text):
    return bool(DEVANAGARI.search(text)) and bool(LATIN.search(text))


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


# ---------------------------------------------------------
# SOURCE B CATALOGUE
# ---------------------------------------------------------

def build_catalogue():

    repo_files = set(HfApi().list_repo_files(REPO_B, repo_type="dataset"))

    scp_path = hf_hub_download(REPO_B, "hi/wav.scp", repo_type="dataset")
    text_path = hf_hub_download(REPO_B, "hi/text", repo_type="dataset")

    paths = {}

    for line in open(scp_path, encoding="utf-8"):

        if not line.strip():
            continue

        utt, raw = line.split(maxsplit=1)

        # Strip the author's absolute server prefix.
        marker = raw.find("hi/wavs/")
        if marker < 0:
            continue

        paths[utt] = raw[marker:].strip()

    texts = {}

    for line in open(text_path, encoding="utf-8"):
        parts = line.rstrip("\n").split(maxsplit=1)
        if len(parts) == 2:
            texts[parts[0]] = parts[1]

    catalogue = []

    for utt, rel in paths.items():

        if rel not in repo_files or utt not in texts:
            continue

        text = texts[utt]

        if not is_code_switched(text):
            continue

        parts = rel.split("/")

        catalogue.append({
            "utt_id": utt,
            "file_name": rel,
            "text": text,
            "voice": parts[2],
            "domain": parts[3],
        })

    print(f"source B usable code-switched clips: {len(catalogue):,}")
    print(f"    voices : {Counter(c['voice'] for c in catalogue).most_common()}")
    print(f"    domains: {len(set(c['domain'] for c in catalogue))}")
    print(f"    unique transcripts: "
          f"{len(set(normalize_text(c['text']) for c in catalogue)):,}")

    return catalogue


def select_topup(catalogue, needed, existing_texts):
    """
    Pick whole transcript groups (both voice renditions together), spread
    round-robin across domains so no single domain dominates.
    """

    groups = defaultdict(list)

    for record in catalogue:

        key = normalize_text(record["text"])

        # Never reuse a transcript already present from source A.
        if key in existing_texts:
            continue

        groups[key].append(record)

    by_domain = defaultdict(list)

    for key, records in groups.items():
        by_domain[records[0]["domain"]].append((key, records))

    rng = random.Random(SEED)

    for domain in by_domain:
        rng.shuffle(by_domain[domain])

    domains = sorted(by_domain)
    picked = []
    exhausted = set()

    while len(picked) < needed and len(exhausted) < len(domains):

        for domain in domains:

            if len(picked) >= needed:
                break

            pool = by_domain[domain]

            if not pool:
                exhausted.add(domain)
                continue

            _, records = pool.pop()

            # Keep both voice renditions together; trim if it would overshoot.
            for record in records:
                if len(picked) < needed:
                    picked.append(record)

    print(f"\nselected {len(picked):,} top-up clips")
    print(f"    voices : {Counter(p['voice'] for p in picked).most_common()}")
    print(f"    domains: {len(set(p['domain'] for p in picked))}")

    return picked


# ---------------------------------------------------------
# COLLECTION
# ---------------------------------------------------------

def fetch(job):

    index, record = job

    sample_id = f"hf_hi_en_b_{index:05d}"
    destination = OUTPUT_DIR / f"{sample_id}.wav"

    if destination.exists() and destination.stat().st_size > 0:
        try:
            return index, record, (destination, sf.info(str(destination))), None
        except Exception:
            destination.unlink(missing_ok=True)

    cached = None
    last_error = None

    for attempt in range(6):
        try:
            cached = hf_hub_download(
                REPO_B, record["file_name"], repo_type="dataset"
            )
            break
        except Exception as e:
            last_error = e
            time.sleep(min(2 ** attempt, 30) + random.random())

    if cached is None:
        return index, record, None, f"download failed: {last_error}"

    shutil.copyfile(cached, destination)

    try:
        info = sf.info(str(destination))
    except Exception as e:
        destination.unlink(missing_ok=True)
        return index, record, None, f"unreadable audio: {e}"

    return index, record, (destination, info), None


def collect(picked):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(picked):,} source B clips...\n")

    rows = []
    failures = 0
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:

        for index, record, result, error in pool.map(
            fetch, list(enumerate(picked))
        ):

            done += 1

            if error:
                failures += 1
                print(f"    {record['file_name']}: {error}")
                continue

            destination, info = result

            rows.append({
                "sample_id": destination.stem,
                "file_path": str(destination),
                "language": "Hindi-English",
                "language_type": "Hinglish",
                "label": "fake",
                "source_dataset": "lingamvamshikrishnareddy_octopus_tts_hinglish",
                "generator": GENERATOR_B,
                "voice": record["voice"],
                "source_file_id": record["file_name"],
                "transcript": record["text"],
                "duration": round(info.duration, 4),
                "original_sample_rate": info.samplerate,
                "codec": f"{info.format}/{info.subtype}",
                "channels": info.channels,
                "declared_language": "hi-en",
                "lid": None,
                "domain": record["domain"],
                "split": "",
                "license": LICENSE_B,
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
    Group by transcript, not speaker.

    There are only three synthetic voices in total, so a speaker-disjoint
    split is meaningless here and we do not pretend otherwise. What IS real:
    the same transcript is rendered by multiple voices, so all renditions of
    one transcript must share a split or the text leaks across splits.
    """

    df = df.copy()
    df["_text_key"] = df["transcript"].map(normalize_text)

    rng = random.Random(SEED)
    assignment = {}

    # Stratify PER GENERATOR. Source B renders each transcript with two
    # voices (group size 2) while source A has singletons, so a single
    # global pass sorted by group size pushes all of source B into train
    # and leaves validation/test dominated by source A - which would make
    # generator and clip duration differ systematically between splits.
    for generator, part in df.groupby("generator"):

        sizes = part.groupby("_text_key").size().to_dict()

        targets = {
            split: len(part) * frac
            for split, frac in SPLIT_FRACTIONS.items()
        }

        current = Counter()

        keys = sorted(sizes)
        rng.shuffle(keys)

        for key in sorted(keys, key=lambda k: -sizes[k]):

            split = max(
                SPLIT_FRACTIONS,
                key=lambda s: targets[s] - current[s],
            )

            assignment[key] = split
            current[split] += sizes[key]

        print(f"    {generator}: "
              + " ".join(f"{s}={current[s]}" for s in SPLIT_FRACTIONS))

    return df["_text_key"].map(assignment)


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

def validate(df):

    print("\n" + "=" * 55)
    print("HINGLISH FAKE (COMBINED) - VALIDATION")
    print("=" * 55)

    failures = []

    def check(condition, message):
        print(f"    {'OK  ' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    print("\nCOUNTS")
    check(len(df) == TARGET_TOTAL,
          f"total = {len(df)} (expected {TARGET_TOTAL})")
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
          f"transcripts with both scripts = {int(mixed.sum())}/{len(df)}")

    print("\nSOURCE / GENERATOR / VOICE MIX")
    print(df["source_dataset"].value_counts().to_string())
    print()
    print(df["generator"].value_counts().to_string())
    print()
    print(df["voice"].value_counts().to_string())

    print("\nSPLIT COUNTS")
    print(df["split"].value_counts().to_string())

    print("\n    generator x split:")
    print(pd.crosstab(df["generator"], df["split"]))

    print("\nTRANSCRIPT LEAKAGE ACROSS SPLITS")

    keys = df["transcript"].map(normalize_text)
    groups = {
        split: set(keys[df["split"] == split])
        for split in ["train", "validation", "test"]
    }

    for a, b in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = len(groups[a] & groups[b])
        check(overlap == 0, f"transcript overlap {a} & {b} = {overlap}")

    print("\nDURATION STATS")
    d = df["duration"]
    print(f"    total hours  {d.sum() / 3600:.2f}")
    print(f"    min          {d.min():.2f}")
    print(f"    median       {d.median():.2f}")
    print(f"    mean         {d.mean():.2f}")
    print(f"    max          {d.max():.2f}")

    print("\n    by source:")
    for source, part in df.groupby("generator"):
        print(f"        {source:<24} n={len(part):>5} "
              f"median={part['duration'].median():.2f}s "
              f"total={part['duration'].sum() / 3600:.2f}h")

    print("\nAUDIO FORMAT")
    print("    original_sample_rate:")
    print(df["original_sample_rate"].value_counts().to_string())
    print("\n    codec:")
    print(df["codec"].value_counts().to_string())
    print("\n    channels:")
    print(df["channels"].value_counts().to_string())
    print(f"\n    bytes on disk: {sizes.sum() / 1e9:.3f} GB")

    print("\nTRANSCRIPTS")
    print(f"    present {int(df['transcript'].notna().sum())}/{len(df)}")
    print(f"    unique  {keys.nunique()}")

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

    if not METADATA_PATH.exists():
        raise SystemExit(
            f"Missing {METADATA_PATH}. Run scripts/prepare_hinglish_fake.py first."
        )

    source_a = pd.read_csv(METADATA_PATH)
    source_a = source_a[
        source_a["source_dataset"] == "nameissakthi_hindi_english_bilingual"
    ]

    print(f"source A clips already collected: {len(source_a):,}")

    needed = TARGET_TOTAL - len(source_a)
    print(f"top-up needed: {needed:,}\n")

    if needed <= 0:
        raise SystemExit("nothing to top up")

    catalogue = build_catalogue()

    existing_texts = set(source_a["transcript"].map(normalize_text))

    picked = select_topup(catalogue, needed, existing_texts)

    if len(picked) < needed:
        raise SystemExit(
            f"source B only yielded {len(picked)} of {needed} needed clips"
        )

    source_b = collect(picked)

    combined = pd.concat([source_a, source_b], ignore_index=True)
    combined["split"] = assign_splits(combined)

    combined = combined.sort_values("sample_id").reset_index(drop=True)
    combined.to_csv(METADATA_PATH, index=False)

    passed = validate(combined)

    summary = {
        "dataset": "Hinglish FAKE - Dataset V1",
        "language": "Hindi-English",
        "language_type": "Hinglish",
        "label": "fake",
        "total_samples": int(len(combined)),
        "sources": [
            {
                "repo": "nameissakthi/hindi-english-bilingual",
                "generator": "Indic_Parler_TTS",
                "voices": ["Rani"],
                "clips": int((combined["generator"] == "Indic_Parler_TTS").sum()),
                "license": "CC BY 4.0",
            },
            {
                "repo": REPO_B,
                "generator": GENERATOR_B,
                "voices": ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"],
                "clips": int((combined["generator"] == GENERATOR_B).sum()),
                "license": LICENSE_B,
            },
        ],
        "split_counts": {
            str(k): int(v) for k, v in combined["split"].value_counts().items()
        },
        "duration_hours": round(float(combined["duration"].sum() / 3600), 2),
        "unique_voices": int(combined["voice"].nunique()),
        "unique_generators": int(combined["generator"].nunique()),
        "limitations": [
            "Only 2 generators and 3 synthetic voices. This is a weak V1 "
            "fake source and is NOT evidence of unseen-generator or "
            "unseen-speaker generalization.",
            "Splits are transcript-disjoint, NOT speaker-disjoint - there "
            "are only three synthetic voices to be disjoint on.",
            "DURATION MISMATCH: source A clips are very short "
            "(median ~1.5 s) while MUCS real Hinglish is ~5.1 s. Clip "
            "duration could act as a shortcut feature. To be handled at the "
            "common preprocessing/merge step, not here.",
            "Source A upstream README claims 10,094 Hinglish utterances; "
            "only 1,756 are genuinely code-switched by transcript analysis.",
            "Source A 'language' field is unreliable (69 'hi' and 1 'en' "
            "rows are actually code-switched).",
            "Source B files use a .wav extension but contain MP3 "
            "(MPEG_LAYER_III) audio; bytes were preserved as published.",
            "Source B repo has no README and states no license.",
        ],
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nMetadata:", METADATA_PATH)
    print("Summary: ", SUMMARY_PATH)
    print("Audio:   ", OUTPUT_DIR)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
