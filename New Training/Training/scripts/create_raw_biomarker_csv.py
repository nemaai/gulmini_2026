import os
import tempfile
from datetime import datetime

import boto3
import numpy as np
import pandas as pd

from biomarkers_apiclean import compute_biomarkers
from eeg_utils import convert_window, to_regions

# =====================================================
# CONFIG
# =====================================================

BUCKET = "dementia-research2025"

DATASETS = [
    "REEG-BACA-19",
    "AD Auditory",
    "ADFSU",
    "ADFTD",
    "ADSZ",
    "APAVA-19",
    "BrainLat"
]

OUTPUT_CSV = (
    "/home/ubuntu/training/outputs/"
    "raw_biomarkers_v2.csv"
)

LOG_FILE = (
    "/home/ubuntu/training/logs/"
    "create_raw_biomarker_csv.log"
)

# =====================================================
# LOGGING
# =====================================================

os.makedirs(
    os.path.dirname(LOG_FILE),
    exist_ok=True
)

os.makedirs(
    os.path.dirname(OUTPUT_CSV),
    exist_ok=True
)

def log(msg):

    ts = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = f"[{ts}] {msg}"

    print(line)

    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# =====================================================
# START
# =====================================================

log("=" * 80)
log("NEW RAW BIOMARKER GENERATION RUN")
log("=" * 80)

s3 = boto3.client("s3")


all_rows = []

# =====================================================
# PROCESS DATASETS
# =====================================================

for dataset in DATASETS:

    try:

        log(f"Processing dataset: {dataset}")

        # -------------------------
        # LABEL FILE
        # -------------------------

        label_key = (
            f"New-Training-DB/"
            f"{dataset}/Label/label.npy"
        )

        tmp_label = tempfile.NamedTemporaryFile(
            suffix=".npy",
            delete=False
        )

        s3.download_file(
            BUCKET,
            label_key,
            tmp_label.name
        )

        labels = np.load(tmp_label.name)

        os.unlink(tmp_label.name)

        # -------------------------
        # FEATURE FILES
        # -------------------------

        prefix = (
            f"New-Training-DB/"
            f"{dataset}/Feature/"
        )

        paginator = s3.get_paginator(
            "list_objects_v2"
        )

        feature_files = []

        for page in paginator.paginate(
            Bucket=BUCKET,
            Prefix=prefix
        ):

            for obj in page.get(
                "Contents",
                []
            ):

                key = obj["Key"]

                if key.endswith(".npy"):
                    feature_files.append(key)

        feature_files = sorted(feature_files)

        log(
            f"{dataset}: "
            f"{len(feature_files)} feature files | "
            f"{len(labels)} labels"
        )

        # -------------------------
        # PROCESS FILES
        # -------------------------

        for idx, key in enumerate(feature_files):

            try:

                tmp_feat = tempfile.NamedTemporaryFile(
                    suffix=".npy",
                    delete=False
                )

                s3.download_file(
                    BUCKET,
                    key,
                    tmp_feat.name
                )

                data = np.load(
                    tmp_feat.name
                )

                os.unlink(
                    tmp_feat.name
                )

                biomarker_rows = []

                for window in data:

                    try:

                        if (
                            len(window.shape) != 2
                            or window.shape[1] != 19
                        ):
                            continue

                        emotiv = convert_window(
                            window
                        )

                        regions = to_regions(
                            emotiv
                        )

                        biomarkers = (
                            compute_biomarkers(
                                regions.T,
                                256
                            )
                        )

                        biomarker_rows.append(
                            biomarkers
                        )

                    except Exception:
                        continue

                if len(
                    biomarker_rows
                ) == 0:

                    log(
                        f"SKIPPED: "
                        f"{key} "
                        f"(no valid windows)"
                    )

                    continue

                final_row = (
                    pd.DataFrame(
                        biomarker_rows
                    )
                    .mean()
                    .to_dict()
                )

                # ---------------------
                # LABELS
                # ---------------------

                if idx >= len(labels):

                    log(
                        f"Label mismatch: "
                        f"{key}"
                    )

                    continue

                label_row = labels[idx]

                final_row[
                    "dataset"
                ] = dataset

                final_row[
                    "true_label"
                ] = int(
                    label_row[0]
                )

                final_row[
                    "subject_id"
                ] = int(
                    label_row[1]
                )

                final_row[
                    "file_source"
                ] = key

                all_rows.append(
                    final_row
                )

                if idx % 25 == 0:

                    log(
                        f"{dataset}: "
                        f"{idx}/"
                        f"{len(feature_files)}"
                    )

            except Exception as e:

                log(
                    f"FAILED FILE: "
                    f"{key} :: {str(e)}"
                )

        log(
            f"Completed dataset: "
            f"{dataset}"
        )

    except Exception as e:

        log(
            f"FAILED DATASET: "
            f"{dataset} :: {str(e)}"
        )

# =====================================================
# SAVE CSV
# =====================================================

df = pd.DataFrame(
    all_rows
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

log("=" * 80)
log("RUN COMPLETE")
log(f"Rows created: {len(df)}")
log(f"CSV saved: {OUTPUT_CSV}")
log(f"Shape: {df.shape}")
log("=" * 80)