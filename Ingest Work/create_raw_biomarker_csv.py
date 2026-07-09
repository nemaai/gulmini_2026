# source "/Users/nidhi/Documents/Test Server/STOKCS/.venv/bin/activate"
import os
import tempfile
from datetime import datetime

import boto3
import numpy as np
import pandas as pd
import json

from scripts12.biomarkers.biomarkers_apiclean import compute_biomarkers
from scripts12.utils.eeg_utils import convert_window, to_regions
from scripts12.biomarkers.spectral_biomarkers import extract_new_features
from scripts12.config import *
# =====================================================
# CONFIG
# =====================================================

# BUCKET = "dementia-research2025"
prefix_og = "nema_final_used/preprocessed_npy/"
folder_feature = "/Features/"
new_data = True

# prefix_og = "New-Training-DB/"
# folder_feature = "/Feature/"
# new_data = False


OUTPUT_CSV = (
    local +"/outputs/external/"
    "padic_external_cohort_raw_biomarkers_v2.csv"
)

LOG_FILE = (
    local+"/logs/"
    "create_raw_biomarker_csv.log"
)

s_freq = 128

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



import numpy as np

def create_windows(
    eeg,
    window_size=128,
    overlap=0
):

    eeg = eeg.T

    windows = []

    step = (
        window_size - overlap
    )

    for start in range(
        0,
        eeg.shape[0] - window_size + 1,
        step
    ):

        stop = start + window_size

        windows.append(
            eeg[start:stop]
        )

    return np.asarray(
        windows,
        dtype=np.float32
    )

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
            prefix_og+
            f"{dataset}/Label/label.npy"
        )
        # print(label_key, "--=-==lablw ")
        # print(tmp_label.name)

        
        tmp_label = tempfile.NamedTemporaryFile(
            suffix=".npy",
            delete=False
        )

        s3.download_file(
            BUCKET,
            label_key,
            tmp_label.name
        )
        # print(tmp_label.name)
        # if not new_data:

            # print("--old--")
        labels = np.load(tmp_label.name)
        # if new_data== True:
        # # labels = np.load(tmp_label.name)
        #     labels = np.locdad(tmp_label.name, allow_pickle=True)
        #     # print(labels,"----")

        #     labels = [
        #         row.tolist()
        #         for row in labels
        #     ]
        

        # for row in labels:
        #     print(row)
                # for row in labels:
                #     print(row)
        # print(labels.shape)
        # print(labels.dtype)
        # print(labels[:5])
        # print(labels, "----")
        os.unlink(tmp_label.name)

        # -------------------------
        # FEATURE FILES
        # -------------------------

        prefix = (
            prefix_og+
            f"{dataset}" + folder_feature
        )
        # prefix = (
        #     prefix_og+
        #     f"{dataset}/"
        # )
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
        # print(feature_files, "--feature----")

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
                
                if not new_data :
                    # print("--old_data ")
                    data = np.load(
                        tmp_feat.name
                    )
                if new_data == True:
                    data = np.load(
                    tmp_feat.name, allow_pickle=True
                )
                    # print("---hello")
                    data = create_windows(data, window_size=128)
                    # data = [
                    #             row.tolist()
                    #             for row in data
                    #         ]
                    # print( "=---" , len(data))
                os.unlink(
                    tmp_feat.name
                )

                biomarker_rows = []
                # print(data, "-0-90")
                for window in data:
                    # print( "----qindoew", window.shape, type(window), len(window[0]), len(window[1]))
                    try:
                        # print(window.shape , window.shape[1])
                        if (
                            len(window.shape) != 2
                            or window.shape[1] != 19
                        ):
                            print(window.shape, window.shape[1])
                            continue

                        emotiv = convert_window(
                            window
                        )

                        regions = to_regions(
                            emotiv
                        )
                        # print("00000kk")

                        biomarkers = (
                            compute_biomarkers(
                                regions.T,
                                s_freq
                            )
                        )
                        # print("--done old e")
                        connectivity_features=extract_new_features(regions.T)
                        
                        # print(len(connectivity_features),"---")

                        # biomarker_rows.append(
                        #     biomarkers
                        # )
                        # print(len(combined_features), "----")
                        combined_features = {
                            **biomarkers,
                            **connectivity_features
                        }
                        combined_features = {
                            k: round(float(v), 6)
                            for k, v in combined_features.items()
                        }
                        # print(combined_features, )
                        biomarker_rows.append(
                            combined_features
                        )
                    except Exception:
                        continue
                # print(len(biomarker_rows), "----")
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

                if idx >= len(labels+1):

                    log(
                        f"Label mismatch: "
                        f"{key}"
                    )

                    continue

                label_row = labels[idx]
                print(label_row, "---0-")
                final_row[
                    "dataset"
                ] = dataset

                final_row[
                    "true_label"
                ] = int(labels[idx]
                    # label_row[0]
                )

                final_row[
                    "subject_id"
                ] = int(
                    idx #label_row[1]
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
                import traceback
                traceback.print_exc()
                log(
                    f"FAILED FILE: "
                    f"{key} :: {str(e)}"
                )

        log(
            f"Completed dataset: "
            f"{dataset}"
        )
        # break
    

    except Exception as e:

        log(
            f"FAILED DATASET: "
            f"{dataset} :: {str(e)}"
        )
    # break

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

