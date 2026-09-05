from datasets import load_dataset, Audio
from sklearn.model_selection import GroupShuffleSplit

from collections import Counter
from pathlib import Path

import pandas as pd
import re
import shutil


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

# The Hindi config contains only two generators. Verified by reading the
# "Generative Model" column of all 107 Hindi parquet shards: shards 0-54 are
# freevc24, shards 54-106 are xtts_v2, and the transition inside shard 54 goes
# straight from freevc24 to xtts_v2 with no vits block between them.
# vits exists in IndicSynth (e.g. Bengali) but not in Hindi.
QUOTAS = {
    "xtts_v2": 2500,
    "freevc24": 2500,
}

TOTAL_TARGET = sum(QUOTAS.values())

OUTPUT_DIR = Path("data/selected/indicsynth_hi")
METADATA_PATH = Path("metadata/indicsynth_hi.csv")

SEED = 42


# ---------------------------------------------------------
# HELPER: NORMALIZE SPEAKER IDS
# ---------------------------------------------------------

def normalize_speaker_id(value):
    """
    Turns:
        1087.0 -> "1087"
        1087   -> "1087"

    This prevents speaker-ID mismatches.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    # float like 1087.0
    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    value = str(value).strip()

    # string like "1087.0"
    if re.fullmatch(r"\d+\.0+", value):
        value = value.split(".")[0]

    return value


# ---------------------------------------------------------
# HELPER: NORMALIZE GENERATOR NAME
# ---------------------------------------------------------

def normalize_generator(value):

    if value is None:
        return None

    value = str(value).strip().lower()

    value = value.replace("-", "_").replace(" ", "_")

    aliases = {
        "xttsv2": "xtts_v2",
        "xtts_v2": "xtts_v2",

        "vits": "vits",

        "freevc": "freevc24",
        "freevc_24": "freevc24",
        "freevc24": "freevc24",
    }

    return aliases.get(value, value)


# ---------------------------------------------------------
# HELPER: SAVE ORIGINAL AUDIO
# ---------------------------------------------------------

def save_audio(audio, destination):
    """
    Saves original encoded audio.

    NO:
        resampling
        denoising
        normalization
        conversion

    We preserve the original WAV.
    """

    if not isinstance(audio, dict):
        raise TypeError(
            "Audio is not a dictionary. "
            "Did you forget Audio(decode=False)?"
        )

    audio_bytes = audio.get("bytes")
    audio_path = audio.get("path")

    # Streaming normally gives us bytes
    if audio_bytes is not None:

        with open(destination, "wb") as f:
            f.write(audio_bytes)

        return

    # fallback if Hugging Face gives a local path
    if audio_path:

        source = Path(audio_path)

        if source.exists():
            shutil.copy2(source, destination)
            return

    raise RuntimeError(
        "Could not find audio bytes or usable path."
    )


# ---------------------------------------------------------
# HELPER: CREATE SPEAKER-DISJOINT SPLITS
# ---------------------------------------------------------

def create_splits(df):

    groups = df["target_speaker_id"]

    # --------------------------
    # 70% train / 30% temporary
    # --------------------------

    splitter1 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=SEED
    )

    train_index, temp_index = next(
        splitter1.split(
            df,
            groups=groups
        )
    )

    temp_df = df.iloc[temp_index]

    # --------------------------
    # Split remaining 30%:
    #
    # 15% validation
    # 15% test
    # --------------------------

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.50,
        random_state=SEED + 1
    )

    val_local, test_local = next(
        splitter2.split(
            temp_df,
            groups=temp_df["target_speaker_id"]
        )
    )

    splits = pd.Series(
        index=df.index,
        dtype="object"
    )

    splits.iloc[train_index] = "train"

    splits.iloc[
        temp_index[val_local]
    ] = "validation"

    splits.iloc[
        temp_index[test_local]
    ] = "test"

    return splits


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

print("\nLoading IndicSynth Hindi...")

dataset = load_dataset(
    "vdivyasharma/IndicSynth",
    "Hindi",
    split="train",
    streaming=True
)


# ---------------------------------------------------------
# DO NOT DECODE AUDIO
# ---------------------------------------------------------

dataset = dataset.cast_column(
    "audio",
    Audio(decode=False)
)


# ---------------------------------------------------------
# SHUFFLE
# ---------------------------------------------------------

print("Shuffling streaming shards...")

dataset = dataset.shuffle(
    seed=SEED,

    # Keep this SMALL for audio.
    buffer_size=128
)


# ---------------------------------------------------------
# PREPARE OUTPUT
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


counts = Counter()

rows = []

scanned = 0


# ---------------------------------------------------------
# STREAM DATA
# ---------------------------------------------------------

print("\nStarting collection...\n")


for sample in dataset:

    scanned += 1

    generator = normalize_generator(
        sample.get("Generative Model")
    )


    # Unknown generator?
    if generator not in QUOTAS:
        continue


    # Already filled this generator?
    if counts[generator] >= QUOTAS[generator]:

        # If everything is complete, stop.
        if all(
            counts[g] >= QUOTAS[g]
            for g in QUOTAS
        ):
            break

        continue


    # -----------------------------------------------------
    # SPEAKER IDS
    # -----------------------------------------------------

    source_speaker = normalize_speaker_id(
        sample.get("Source Speaker_ID")
    )

    target_speaker = normalize_speaker_id(
        sample.get("Target Speaker ID")
    )


    # We need this for speaker-disjoint splitting.
    if target_speaker is None:
        continue


    # -----------------------------------------------------
    # SAMPLE ID
    # -----------------------------------------------------

    sample_number = counts[generator]

    sample_id = (
        f"is_hi_{generator}_{sample_number:05d}"
    )


    # -----------------------------------------------------
    # SAVE AUDIO
    # -----------------------------------------------------

    destination = (
        OUTPUT_DIR /
        f"{sample_id}.wav"
    )


    try:

        save_audio(
            sample["audio"],
            destination
        )

    except Exception as e:

        print(
            f"Skipping row {scanned}: "
            f"audio save error: {e}"
        )

        continue


    # -----------------------------------------------------
    # TRANSCRIPT
    # -----------------------------------------------------

    transcript = sample.get(
        "TTS Transcript"
    )

    # FreeVC normally gives None
    if transcript is None:
        transcript = ""


    # -----------------------------------------------------
    # METADATA
    # -----------------------------------------------------

    row = {

        "sample_id":
            sample_id,

        "file_path":
            str(destination),

        "language":
            "Hindi",

        "language_type":
            "Hindi",

        "label":
            "fake",

        "source_dataset":
            "IndicSynth",

        "generator":
            generator,

        "generator_family":
            (
                "voice_conversion"
                if generator == "freevc24"
                else "text_to_speech"
            ),

        "source_speaker_id":
            source_speaker,

        "target_speaker_id":
            target_speaker,

        "gender":
            sample.get("Gender"),

        "source_reference_audio":
            sample.get(
                "Source Reference Audio"
            ),

        "target_reference_audio":
            sample.get(
                "Target Reference Audio"
            ),

        "transcript":
            transcript,

        # We know the streamed files are currently WAV,
        # but we haven't converted them.
        "codec":
            "wav",

        "fake_start":
            0,

        # Fill duration later during common preprocessing.
        "fake_end":
            "",

        "has_timestamp_labels":
            False,

        "license":
            "CC BY-NC 4.0",

        "split":
            ""
    }


    rows.append(row)

    counts[generator] += 1


    # -----------------------------------------------------
    # PROGRESS
    # -----------------------------------------------------

    if len(rows) % 100 == 0:

        print(
            f"Scanned: {scanned:,} | "
            f"Saved: {len(rows):,}/{TOTAL_TARGET:,}"
        )

        print(
            "   "
            + " | ".join(
                f"{g}: {counts[g]}/{QUOTAS[g]}"
                for g in QUOTAS
            )
        )


    # -----------------------------------------------------
    # FINISHED?
    # -----------------------------------------------------

    if all(
        counts[g] >= QUOTAS[g]
        for g in QUOTAS
    ):
        break


# ---------------------------------------------------------
# CHECK QUOTAS
# ---------------------------------------------------------

print("\nCollection completed.")

print("\nGenerator totals:")

for generator in QUOTAS:

    print(
        generator,
        counts[generator],
        "/",
        QUOTAS[generator]
    )


if not all(
    counts[g] >= QUOTAS[g]
    for g in QUOTAS
):

    raise RuntimeError(
        "Could not fill all generator quotas."
    )


# ---------------------------------------------------------
# CREATE DATAFRAME
# ---------------------------------------------------------

df = pd.DataFrame(rows)


# ---------------------------------------------------------
# CREATE SPEAKER-DISJOINT SPLITS
# ---------------------------------------------------------

print("\nCreating speaker-disjoint splits...")

df["split"] = create_splits(df)


# ---------------------------------------------------------
# VERIFY SPEAKER LEAKAGE
# ---------------------------------------------------------

speaker_split_counts = (

    df.groupby(
        "target_speaker_id"
    )["split"]

    .nunique()
)


leaking_speakers = speaker_split_counts[
    speaker_split_counts > 1
]


if len(leaking_speakers) > 0:

    raise RuntimeError(
        "Speaker leakage detected!"
    )


# ---------------------------------------------------------
# SAVE METADATA
# ---------------------------------------------------------

df.to_csv(
    METADATA_PATH,
    index=False
)


# ---------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------

print("\n===================================")
print("INDICSYNTH V1 COMPLETE")
print("===================================")

print(
    "\nTotal samples:",
    len(df)
)

print(
    "\nUnique target speakers:",
    df["target_speaker_id"].nunique()
)


print("\nGenerator counts:")

print(
    df["generator"]
    .value_counts()
)


print("\nSplit counts:")

print(
    df["split"]
    .value_counts()
)


print("\nGenerator x Split:")

print(
    pd.crosstab(
        df["generator"],
        df["split"]
    )
)


# ---------------------------------------------------------
# FINAL LEAKAGE TEST
# ---------------------------------------------------------

train_speakers = set(
    df[
        df["split"] == "train"
    ]["target_speaker_id"]
)

val_speakers = set(
    df[
        df["split"] == "validation"
    ]["target_speaker_id"]
)

test_speakers = set(
    df[
        df["split"] == "test"
    ]["target_speaker_id"]
)


print("\nSpeaker overlap:")

print(
    "Train <-> Validation:",
    len(
        train_speakers
        & val_speakers
    )
)

print(
    "Train <-> Test:",
    len(
        train_speakers
        & test_speakers
    )
)

print(
    "Validation <-> Test:",
    len(
        val_speakers
        & test_speakers
    )
)


print(
    "\nAudio has NOT been "
    "resampled, denoised or normalized."
)

print(
    "\nMetadata saved to:",
    METADATA_PATH
)

print(
    "Audio saved to:",
    OUTPUT_DIR
)
