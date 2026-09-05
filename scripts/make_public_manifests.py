#!/usr/bin/env python3
"""
Generate the publishable subset of the V1 manifests: metadata/ -> metadata/public/.

The full manifests are LOCAL ONLY (gitignored). They carry, per row:

  * transcript                      verbatim text from Common Voice, Kathbath, MUCS and
                                    IndicSynth. Publishing it republishes their content.
  * speaker_id / target_speaker_id  pseudonymous identifiers for real people. The Kathbath
    / source_speaker_id             and IndicSynth values share ONE namespace (see
                                    DATASET_REPORT.md 7.4a), so publishing them links a
                                    human speaker to the synthetic clones of their voice.
  * source_file_id / session_id     15-digit AI4Bharat utterance IDs; same linkage risk.
  * gender / gender_normalized      a protected attribute.
  * group_id (raw)                  for the two Hinglish-fake sources the group key IS the
                                    verbatim transcript, i.e. literally
                                    "<source_dataset>::<the whole sentence spoken in the clip>".
                                    It is therefore hashed, never published raw.

This script emits only the agreed public columns and a salted hash of group_id.

The salt lives in metadata/public/.group_salt, is generated once, and is gitignored.
Without it the hashed group_id cannot be reversed by dictionary attack against the
public upstream corpora. Keep the file if you want stable IDs across regenerations;
delete it and every group hash changes.

Read-only with respect to data/. Writes only under metadata/public/.

Usage:  python scripts/make_public_manifests.py [--check]
        --check  verify existing public files are current; exit 1 if not
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
META = REPO / "metadata"
PUB = META / "public"
SALT_FILE = PUB / ".group_salt"

# Columns that must NEVER reach metadata/public/ under any name.
FORBIDDEN = {
    "transcript", "tnorm",
    "speaker_id", "target_speaker_id", "source_speaker_id",
    "source_file_id", "session_id", "utterance_id", "row_id",
    "gender", "gender_normalized",
    "voice",  # names the specific Azure/Parler voice actor persona
    "source_reference_audio", "target_reference_audio",
    "file_path",  # points into data/selected/, i.e. the pre-canonical original
}


def get_salt() -> bytes:
    """Load the group-hash salt, creating it on first run."""
    PUB.mkdir(parents=True, exist_ok=True)
    if SALT_FILE.exists():
        return SALT_FILE.read_bytes().strip()
    salt = secrets.token_hex(32).encode()
    SALT_FILE.write_bytes(salt)
    print(f"  generated new salt -> {SALT_FILE.relative_to(REPO)} (gitignored; keep it)")
    return salt


def hash_group(series: pd.Series, salt: bytes) -> pd.Series:
    """Salted, truncated SHA-256 of a group key. Stable for equal inputs."""
    cache: dict[str, str] = {}

    def h(v):
        k = "" if pd.isna(v) else str(v)
        if k not in cache:
            cache[k] = hashlib.sha256(salt + b"|" + k.encode("utf-8")).hexdigest()[:16]
        return cache[k]

    return series.map(h)


def relpath(series: pd.Series) -> pd.Series:
    """Windows-style manifest path -> forward-slash repo-relative path."""
    return series.astype(str).str.replace("\\", "/", regex=False).str.lstrip("./")


def audit(df: pd.DataFrame, name: str) -> None:
    """Fail loudly if anything forbidden survived into a public frame."""
    bad = sorted(set(df.columns) & FORBIDDEN)
    if bad:
        raise SystemExit(f"REFUSING TO WRITE {name}: forbidden column(s) present: {bad}")
    for col in df.columns:
        if df[col].dtype == object:
            s = df[col].dropna().astype(str)
            # a raw group_id would still contain "::"
            if col.endswith("group_id") and s.str.contains("::").any():
                raise SystemExit(f"REFUSING TO WRITE {name}: {col} looks un-hashed")
            if s.str.contains(r"[A-Za-z]:[\\/]{1,2}Users[\\/]", regex=True).any():
                raise SystemExit(f"REFUSING TO WRITE {name}: absolute user path in {col}")


def serialise(df: pd.DataFrame) -> str:
    """The exact CSV text we would write. Used for both writing and --check, so the
    comparison never trips over CSV round-trip dtype changes (empty string -> NaN)."""
    return df.to_csv(index=False, lineterminator="\n")


def write(df: pd.DataFrame, name: str) -> None:
    audit(df, name)
    out = PUB / name
    out.write_text(serialise(df), encoding="utf-8", newline="")
    print(f"  wrote {out.relative_to(REPO)}  ({len(df):,} rows x {len(df.columns)} cols)")


def build_development(salt: bytes) -> pd.DataFrame:
    """The canonical 20,000. Exactly the agreed column set."""
    m = pd.read_csv(META / "dataset_v1_master_v2.csv", low_memory=False)
    return pd.DataFrame({
        "sample_id":      m["sample_id"],
        "label":          m["label"],
        "language_type":  m["language_type"],
        "source_dataset": m["source_dataset"],
        "generator":      m["generator"].fillna(""),
        "split":          m["split"],
        # duration/codec/sha256 all describe the CANONICAL file that `path` points at
        "duration":       m["processed_duration"],
        "codec":          m["processed_codec"],
        "sha256":         m["processed_sha256"],
        "group_id":       hash_group(m["group_id"], salt),
        "path":           relpath(m["processed_file_path"]),
    })


def build_augmented(salt: bytes) -> pd.DataFrame:
    """13,963 training augmentations. Adds parent_sample_id + augmentation_condition,
    without which the file cannot be joined or used; neither is identifying."""
    a = pd.read_csv(META / "dataset_v1_augmented.csv", low_memory=False)
    return pd.DataFrame({
        "sample_id":              a["augmentation_id"],
        "parent_sample_id":       a["parent_sample_id"],
        "label":                  a["label"],
        "language_type":          a["language_type"],
        "source_dataset":         a["source_dataset"],
        "generator":              a["generator"].fillna(""),
        "split":                  a["split"],
        "augmentation_condition": a["augmentation_condition"],
        "duration":               a["duration"],
        "codec":                  a["codec"],
        "sha256":                 a["sha256"],
        "group_id":               hash_group(a["group_id"], salt),
        "path":                   relpath(a["augmented_file_path"]),
    })


def build_robustness(salt: bytes) -> pd.DataFrame:
    """2,000 robustness rows, test parents only."""
    r = pd.read_csv(META / "evaluation" / "robustness_v1.csv", low_memory=False)
    return pd.DataFrame({
        "sample_id":              r["robustness_id"],
        "parent_sample_id":       r["parent_sample_id"],
        "label":                  r["label"],
        "language_type":          r["language_type"],
        "source_dataset":         r["source_dataset"],
        "generator":              r["generator"].fillna(""),
        "split":                  r["split"],
        "augmentation_condition": r["augmentation_condition"],
        "duration":               r["duration"],
        "codec":                  r["codec"],
        "sha256":                 r["sha256"],
        "group_id":               hash_group(r["group_id"], salt),
        "path":                   relpath(r["augmented_file_path"]),
    })


def build_seaspoof(salt: bytes, with_hashes: bool = True) -> pd.DataFrame:
    """2,000 external-eval rows. SEA-Spoof is gated and non-redistributable, so this
    manifest deliberately carries no transcript, no utterance_id and no upstream row_id
    -- it is an index into audio the user must obtain from the authors themselves."""
    s = pd.read_csv(META / "evaluation" / "seaspoof_hi.csv", low_memory=False)
    paths = relpath(s["file_path"])
    sha = pd.Series([""] * len(s), index=s.index)
    if with_hashes:
        vals = []
        for p in paths:
            fp = REPO / p
            if fp.is_file():
                h = hashlib.sha256()
                with open(fp, "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 18), b""):
                        h.update(chunk)
                vals.append(h.hexdigest())
            else:
                vals.append("")
        sha = pd.Series(vals, index=s.index)
    return pd.DataFrame({
        "sample_id":      s["sample_id"],
        "label":          s["label"],
        "language_type":  "Hindi",
        "source_dataset": s["source_dataset"],
        # source_model names the TTS system, not a person; it is the generator here
        "generator":      s["source_model"].fillna(""),
        "split":          "evaluation",
        "spoof_type":     s["spoof_type"].fillna(""),
        "duration":       s["duration"],
        "codec":          s["codec"],
        "sha256":         sha,
        # every row is its own group: no speaker grouping is available upstream
        "group_id":       hash_group(s["sample_id"], salt),
        "path":           paths,
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify public files match a fresh build; do not overwrite")
    ap.add_argument("--no-seaspoof-hashes", action="store_true",
                    help="skip hashing the 2,000 SEA-Spoof files (faster)")
    args = ap.parse_args()

    if not (META / "dataset_v1_master_v2.csv").exists():
        print("metadata/dataset_v1_master_v2.csv not found -- run from the repo root.")
        return 1

    salt = get_salt()
    print("building public manifests ...")
    # --check must compare like with like, so it always hashes SEA-Spoof
    seaspoof_hashes = args.check or not args.no_seaspoof_hashes
    frames = {
        "dataset_v1_public.csv":           build_development(salt),
        "dataset_v1_augmented_public.csv": build_augmented(salt),
        "robustness_v1_public.csv":        build_robustness(salt),
        "seaspoof_hi_public.csv":          build_seaspoof(salt, seaspoof_hashes),
    }

    if args.check:
        stale = []
        for name, df in frames.items():
            p = PUB / name
            if not p.exists():
                stale.append(f"{name}: missing")
                continue
            audit(df, name)
            with open(p, "r", encoding="utf-8", newline="") as fh:
                current = fh.read()
            if current != serialise(df):
                stale.append(f"{name}: out of date")
        if stale:
            print("STALE (re-run without --check):\n  " + "\n  ".join(stale))
            return 1
        print("all public manifests are current")
        return 0

    for name, df in frames.items():
        write(df, name)

    total = sum(len(d) for d in frames.values())
    print(f"\ndone: {len(frames)} files, {total:,} rows")
    print("published columns carry no transcript, speaker, utterance or gender field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
