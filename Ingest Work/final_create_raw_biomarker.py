"""
create_raw_biomarker_csv.py — Biomarker extraction from clean .npy files
=========================================================================
Reads clean .npy files written by data_pipeline.py and computes
biomarkers per file → saves a CSV for XGBoost training.

Changes from previous version:
  1. Reads from nema_final_used/clean_train/{dataset}/ (not preprocessed_npy)
  2. Labels come from split manifest CSV (not label.npy) — filename-keyed,
     no positional index, no label mismatch bugs
  3. Only processes TRAIN + REVIEW quality files (reads quality_tag from manifest)
  4. Only processes files in the train split (reads split_train_latest.csv)
  5. Amplitude check: warns if file looks like wrong units (still in μV)
  6. Cleaned up all dead code and commented-out blocks
  7. Output CSV goes to local/outputs/biomarkers/round1_biomarkers.csv
"""

import io
import os
import tempfile
from datetime import datetime
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

from scripts12.biomarkers.biomarkers_apiclean import compute_biomarkers
from scripts12.utils.eeg_utils import convert_window, to_regions
from scripts12.biomarkers.spectral_biomarkers import extract_new_features
from scripts12.config import BUCKET, DATASETS, local

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Where data_pipeline.py writes clean .npy files
CLEAN_PREFIX   = "nema_final_used/clean_train"

# Where split_manifest.py writes split CSVs
SPLIT_PREFIX   = "nema_final_used/splits"

# Which splits to include in the biomarker CSV
# train + val: XGBoost will do its own CV internally
INCLUDE_SPLITS = {"train", "val"}

# Quality tags to include (exclude REJECT)
INCLUDE_QUALITY = {"TRAIN", "REVIEW"}

# Biomarker window: 1 second at 128Hz
WINDOW_SIZE    = 1024  # 1 second at 256Hz
S_FREQ         = 256

# Round 1 datasets — override DATASETS from config if needed
ROUND1_DATASETS = [
    "ADFSU",
    "DS004504",
    "BrainLat",
    # "test_hardware",
    "P-ADIC",
    "Isfahan",
    "APAVA",
    "FIGSHARE-128Hz",
    "FIGSHARE-256Hz",
    "ADSZ-AD"
    "CAUEEG"
]

OUTPUT_CSV = os.path.join(
    local, "outputs", "biomarkers", "round1_biomarkers.csv"
)
LOG_FILE = os.path.join(
    local, "logs", "create_raw_biomarker_csv.log"
)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE),   exist_ok=True)

s3 = boto3.client("s3")

# ─── LOGGING ──────────────────────────────────────────────────────────────────

def log(msg):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ─── S3 HELPERS ───────────────────────────────────────────────────────────────

# def s3_read_csv(key):
#     data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
#     return pd.read_csv(io.BytesIO(data))
def s3_read_csv(key, dtype=None):
    data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(data), dtype=dtype)

def s3_list_npy(prefix):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".npy"):
                keys.append(obj["Key"])
    return sorted(keys)

# ─── LOAD SPLIT MANIFEST ──────────────────────────────────────────────────────

def load_label_lookup():
    """
    Load the train + val split manifests from split_manifest.py output.
    Returns a dict: {stem → {"true_label": int, "quality_tag": str, "split": str}}
    Only includes files with TRAIN or REVIEW quality and valid labels (0 or 1).
    """
    lookup = {}
    for split_name in INCLUDE_SPLITS:
        key = f"{SPLIT_PREFIX}/split_{split_name}_latest.csv"
        try:
            df = s3_read_csv(key, dtype={"stem": str, "subject_id": str})
            # df = s3_read_csv(key)
            # print(df.head(-20))
            log(f"  Loaded split '{split_name}': {len(df)} rows from s3://{BUCKET}/{key}")
        except Exception as e:
            log(f"  [WARN] Could not load split '{split_name}': {e} — trying manifest folder")
            # Fallback: look for any manifest CSV for this dataset
            continue

        for _, row in df.iterrows():
            # Change to:
            stem_raw = row.get("stem", Path(str(row.get("clean_npy",""))).stem)
            # Pad numeric stems to 5 digits to match CAUEEG filename format
            stem = str(stem_raw).strip()
            if stem.isdigit():
                stem = stem.zfill(5)
            # stem  = str(row.get("stem", Path(str(row.get("clean_npy",""))).stem)).strip()
            # print(stem, "--=-=")
            label = row.get("true_label")
            # print(label, "-jkokolanl---------")
            qtag  = str(row.get("quality_tag", "TRAIN")).strip()
            # print(qtag, "---")
            if label not in [0, 1]:
                continue
            if qtag not in INCLUDE_QUALITY:
                continue

            lookup[stem] = {
                "true_label":  int(label),
                "quality_tag": qtag,
                "split":       split_name,
                "dataset":     str(row.get("dataset", "")),
            }

    log(f"Label lookup: {len(lookup)} files  "
        f"(label=1: {sum(1 for v in lookup.values() if v['true_label']==1)}  "
        f"label=0: {sum(1 for v in lookup.values() if v['true_label']==0)})")
    return lookup

# ─── WINDOWING ────────────────────────────────────────────────────────────────

def create_windows(eeg, window_size=WINDOW_SIZE):
    """
    (19, T) → (N_windows, window_size, 19)
    No overlap for biomarkers (each second is independent).
    """
    eeg_t   = eeg.T                  # (T, 19)
    windows = []
    for start in range(0, eeg_t.shape[0] - window_size + 1, window_size):
        windows.append(eeg_t[start:start + window_size])
    return np.asarray(windows, dtype=np.float32)  # (N, 128, 19)

# ─── BIOMARKER EXTRACTION ─────────────────────────────────────────────────────

def extract_biomarkers_from_file(key):
    """
    Download one clean .npy, compute biomarkers per window, return mean row.
    Returns dict of biomarker values, or None on failure.
    """
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        s3.download_file(BUCKET, key, tmp_path)
        eeg = np.load(tmp_path, allow_pickle=True)
        if eeg.dtype == object:
            eeg = eeg.item()
        eeg = np.array(eeg, dtype=np.float32)

        # Shape guard
        if eeg.ndim != 2:
            raise ValueError(f"Expected 2D, got {eeg.shape}")
        if eeg.shape[0] != 19 and eeg.shape[1] == 19:
            eeg = eeg.T
        if eeg.shape[0] != 19:
            raise ValueError(f"Expected 19ch, got shape {eeg.shape}")

        # Unit check — warn if looks like microvolts
        if eeg.std() > 1.0:
            log(f"  [WARN] {Path(key).stem}: std={eeg.std():.2f} may still be μV "
                f"— check data_pipeline.py unit conversion")

        # Window
        windows = create_windows(eeg, WINDOW_SIZE)
        if len(windows) == 0:
            raise ValueError(f"No windows extracted (duration too short)")

        biomarker_rows = []
        for window in windows:
            try:
                # window: (128, 19)
                if window.shape != (WINDOW_SIZE, 19):
                    continue

                emotiv    = convert_window(window)
                regions   = to_regions(emotiv)

                bio       = compute_biomarkers(regions.T, S_FREQ)
                conn      = extract_new_features(regions.T)

                if bio is None or conn is None:
                    continue

                combined  = {**bio, **conn}
                combined  = {k: round(float(v), 6) for k, v in combined.items()
                             if v is not None and not (isinstance(v, float)
                                                       and (v != v or abs(v) == float("inf")))}
                biomarker_rows.append(combined)

            except Exception:
                continue

        if not biomarker_rows:
            raise ValueError("All windows failed biomarker computation")

        return pd.DataFrame(biomarker_rows).mean().to_dict()

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ─── MAIN ─────────────────────────────────────────────────────────────────────

log("=" * 80)
log("BIOMARKER GENERATION — ROUND 1")
log(f"Datasets: {ROUND1_DATASETS}")
log(f"Splits:   {INCLUDE_SPLITS}")
log(f"Quality:  {INCLUDE_QUALITY}")
log("=" * 80)

# Load label lookup from split manifests
label_lookup = load_label_lookup()

if not label_lookup:
    log("[ERROR] No labels loaded. Run split_manifest.py first.")
    raise SystemExit(1)

all_rows = []

for dataset in ROUND1_DATASETS:

    log(f"\n{'─'*60}")
    log(f"Dataset: {dataset}")
    log(f"{'─'*60}")

    # S3 prefix for clean .npy files written by data_pipeline.py
    npy_prefix = f"{CLEAN_PREFIX}/{dataset}/"
    # print("---data ",npy_prefix)
    # List all .npy files for this dataset
    npy_keys = s3_list_npy(npy_prefix)
    log(f"  Found {len(npy_keys)} .npy files on S3")

    if not npy_keys:
        log(f"  [SKIP] No files found at s3://{BUCKET}/{npy_prefix}")
        log(f"         Run data_pipeline.py --dataset {dataset} first")
        continue
        
    
    
    # Filter to only files in the label lookup (correct split + quality)
    # in_split = [(k, label_lookup[Path(k).stem])
    #             for k in npy_keys
    #             if Path(k).stem in label_lookup
    #             and label_lookup[Path(k).stem]["dataset"] == dataset]
    in_split = [(k, label_lookup[Path(k).stem])
            for k in npy_keys
            if Path(k).stem in label_lookup
            and label_lookup[Path(k).stem]["dataset"].strip().upper()
                == dataset.strip().upper()]
    
    # print(in_split, "0-0-0------")
    log(f"  In split (label+quality filter): {len(in_split)}/{len(npy_keys)} files")

    if not in_split:
        log(f"  [SKIP] No files match label lookup for this dataset")
        log(f"         Check that split manifests contain '{dataset}' entries")
        continue

    n_ok = n_err = n_skip = 0

    for idx, (key, meta) in enumerate(in_split):
        stem  = Path(key).stem
        label = meta["true_label"]
        qtag  = meta["quality_tag"]

        try:
            final_row = extract_biomarkers_from_file(key)

            if final_row is None:
                n_skip += 1
                continue

            # NaN check — skip files where majority of biomarkers are NaN
            nan_count = sum(1 for v in final_row.values()
                            if isinstance(v, float) and v != v)
            if nan_count > len(final_row) * 0.3:
                log(f"  [WARN] {stem}: {nan_count} NaN biomarkers — skipping")
                n_skip += 1
                continue

            final_row["dataset"]    = dataset
            final_row["true_label"] = label
            final_row["subject_id"] = stem
            final_row["file_source"]= key
            final_row["quality_tag"]= qtag
            final_row["split"]      = meta["split"]

            all_rows.append(final_row)
            n_ok += 1

            if (idx + 1) % 25 == 0 or (idx + 1) == len(in_split):
                log(f"  {dataset}: {idx+1}/{len(in_split)}  "
                    f"ok={n_ok}  skip={n_skip}  err={n_err}")

        except Exception as e:
            log(f"  FAILED: {stem} :: {e}")
            n_err += 1

    log(f"  Completed {dataset}: ok={n_ok}  skip={n_skip}  err={n_err}")

# ─── SAVE ─────────────────────────────────────────────────────────────────────

df = pd.DataFrame(all_rows)

df.to_csv(OUTPUT_CSV, index=False)

log("=" * 80)
log("RUN COMPLETE")
log(f"Total rows : {len(df)}")
log(f"label=1    : {int((df['true_label']==1).sum())}")
log(f"label=0    : {int((df['true_label']==0).sum())}")
log(f"Datasets   : {df['dataset'].value_counts().to_dict()}")
log(f"Features   : {len(df.columns)} columns")
log(f"NaN cols   : {int(df.isnull().any(axis=1).sum())} rows with any NaN")
log(f"CSV saved  : {OUTPUT_CSV}")
log("=" * 80)