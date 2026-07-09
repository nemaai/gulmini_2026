"""
Supported Formats
-----------------
1. Allengers Referential (40 channels)
2. Standard Referential (19/32/51 channels)
3. High Density Referential (>64 channels)

"""

import argparse
import io
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import boto3
import mne
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

mne.set_log_level("WARNING")


# =============================================================================
# CONFIGURATION
# =============================================================================

BUCKET = "dementia-research2025"

OUTPUT_ROOT = "nema_final_used/all_npy_raw"

MIN_SFREQ = 64

MASTER_19 = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5","P3","Pz","P4","T6",
    "O1","O2",
]


CHANNEL_ALIASES = {

    "FP1":"Fp1",
    "FP2":"Fp2",

    "FZ":"Fz",
    "CZ":"Cz",
    "PZ":"Pz",

    "T7":"T3",
    "T8":"T4",

    "P7":"T5",
    "P8":"T6",

}


NON_EEG = {

    "PG1","PG2",

    "DC1","DC2","DC3","DC4","DC5",
    "DC6","DC7","DC8","DC9","DC10",

    "BP1","BP2","BP3","BP4",
    "BP5","BP6","BP7","BP8",

    "ECG","EKG",

    "EMG1","EMG2","EMG3","EMG4",

    "EOG","EOG H","EOG V",

    "A1","A2","M1","M2",

    "X1","X2","X3","X4"

}


s3 = boto3.client("s3")


# =============================================================================
# COMMON HELPERS
# =============================================================================

def log(msg):

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}",
        flush=True,
    )


def s3_upload_npy(arr, bucket, key):

    buf = io.BytesIO()

    np.save(buf, arr)

    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=buf.getvalue(),
        ContentType="application/octet-stream",
    )


def s3_exists(bucket, key):

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True

    except ClientError:
        return False


def inspect_edf(raw):

    return {

        "sfreq": raw.info["sfreq"],

        "channels": raw.ch_names,

        "n_channels": len(raw.ch_names),

        "duration": raw.n_times / raw.info["sfreq"],

    }


# =============================================================================
# EDF TYPE DETECTION
# =============================================================================
def detect_edf_type(raw):
    """
    Detect EDF recording type.

    Returns
    -------
    allengers
    referential
    hd_referential
    bipolar
    cz_reference
    unknown
    """

    ch_names = raw.ch_names

    sfreq = raw.info["sfreq"]

    n_channels = len(ch_names)

    # -------------------------------------------------------
    # High Density
    # -------------------------------------------------------

    if n_channels > 64:

        return "hd_referential"

    # -------------------------------------------------------
    # Allengers
    # -------------------------------------------------------

    if any(ch.startswith("PG") for ch in ch_names):

        return "allengers"

    # -------------------------------------------------------
    # Bipolar
    # -------------------------------------------------------

    bipolar = 0

    for ch in ch_names:

        if "-" in ch.replace("EEG ", ""):

            bipolar += 1

    if bipolar > 5:

        return "bipolar"

    # -------------------------------------------------------
    # Cz Reference
    # -------------------------------------------------------

    cz_ref = 0

    for ch in ch_names:

        if ch.upper().endswith("-CZ"):

            cz_ref += 1

    if cz_ref > 5:

        return "cz_reference"

    # -------------------------------------------------------
    # Standard Referential
    # -------------------------------------------------------

    return "referential"

# =============================================================================
# CHANNEL STANDARDISATION
# =============================================================================
def standardise_channel_names(raw):
    """
    Rename channels into standard MASTER_19 convention.
    """

    rename = {}

    for ch in raw.ch_names:

        name = ch.strip()

        name = name.replace("-Ref", "")
        name = name.replace("-REF", "")
        name = name.replace("EEG ", "")

        if name in CHANNEL_ALIASES:

            name = CHANNEL_ALIASES[name]

        rename[ch] = name

    raw.rename_channels(rename)

    return raw

# =============================================================================
# MASTER 19 SELECTION
# =============================================================================
def select_master19(raw):
    """
    Keep only MASTER_19 channels.
    """

    raw = standardise_channel_names(raw)

    available = {}

    for ch in raw.ch_names:

        if ch not in NON_EEG:

            available[ch] = ch

    missing = []

    picks = []

    for ch in MASTER_19:

        if ch in available:

            picks.append(available[ch])

        else:

            missing.append(ch)

    if missing:

        raise ValueError(
            f"Missing channels : {missing}"
        )

    raw.pick_channels(
        picks,
        ordered=True,
    )

    return raw

# =============================================================================
# HANDLER 1 - ALLENGERS
# =============================================================================
def handle_allengers(raw):
    """
    Allengers 40-channel EDF.
    """

    log("    Handler : Allengers")

    raw.pick("eeg")

    raw = select_master19(raw)

    eeg = raw.get_data()

    eeg = eeg.astype(np.float32)

    return eeg

# =============================================================================
# HANDLER 2 - REFERENTIAL
# =============================================================================
def handle_referential(raw):
    """
    Generic referential EDF.
    """

    log("    Handler : Referential")

    raw.pick("eeg")

    raw = select_master19(raw)

    eeg = raw.get_data()

    eeg = eeg.astype(np.float32)

    return eeg

# =============================================================================
# HANDLER 3 - HIGH DENSITY
# =============================================================================
def handle_hd_referential(raw):
    """
    High-density referential EDF.
    """

    log("    Handler : High Density")

    raw.pick("eeg")

    raw = select_master19(raw)

    eeg = raw.get_data()

    eeg = eeg.astype(np.float32)

    return eeg

# =============================================================================
# CONVERT SINGLE EDF
# =============================================================================
def convert_one_edf(
    bucket,
    edf_key,
    output_key,
):
    """
    Convert one EDF recording into a standardised
    (19, T) float32 NumPy array.
    """

    log(f"Converting : {Path(edf_key).name}")

    with tempfile.TemporaryDirectory() as tmp:

        local_file = Path(tmp) / Path(edf_key).name

        s3.download_file(
            bucket,
            edf_key,
            str(local_file),
        )

        raw = mne.io.read_raw_edf(
            str(local_file),
            preload=True,
            verbose=False,
        )

        metadata = inspect_edf(raw)

        edf_type = detect_edf_type(raw)

        log(f"    Type      : {edf_type}")
        log(f"    Channels  : {metadata['n_channels']}")
        log(f"    SFreq     : {metadata['sfreq']} Hz")

        # ----------------------------------------------------
        # Route to correct handler
        # ----------------------------------------------------

        if edf_type == "allengers":

            eeg = handle_allengers(raw)

        elif edf_type == "referential":

            eeg = handle_referential(raw)

        elif edf_type == "hd_referential":

            eeg = handle_hd_referential(raw)

        else:

            raise ValueError(
                f"Unsupported EDF type : {edf_type}"
            )

    # --------------------------------------------------------
    # Unit sanity check
    # --------------------------------------------------------

    if eeg.std() > 1:

        log("    Converting μV → V")

        eeg = eeg / 1e6

    eeg = eeg.astype(np.float32)

    if eeg.shape[0] != 19:

        raise ValueError(
            f"Expected 19 channels. Got {eeg.shape[0]}"
        )

    s3_upload_npy(
        eeg,
        bucket,
        output_key,
    )

    return {

        "shape": eeg.shape,

        "duration":

            round(
                eeg.shape[1] / metadata["sfreq"],
                2,
            ),

        "sfreq": metadata["sfreq"],

        "edf_type": edf_type,

    }

# =============================================================================
# PROCESS SINGLE FILE
# =============================================================================
def process_one(
    bucket,
    edf_key,
    output_prefix,
):
    """
    Process one EDF.
    """

    stem = Path(edf_key).stem

    output_key = f"{output_prefix}/{stem}.npy"

    if s3_exists(bucket, output_key):

        log(f"[SKIP] {stem}")

        return None

    try:

        info = convert_one_edf(

            bucket=bucket,

            edf_key=edf_key,

            output_key=output_key,

        )

        log(
            f"    ✓ Saved ({info['edf_type']}) "
            f"{info['shape']}"
        )

        return {

            "subject_id": stem,

            "edf_type": info["edf_type"],

            "shape": str(info["shape"]),

            "duration": info["duration"],

            "sfreq": info["sfreq"],

            "clean_npy": output_key,

        }

    except Exception as e:

        log(f"[ERROR] {stem}")

        log(str(e))

        traceback.print_exc()

        return None
    
# =============================================================================
# BATCH INGESTION
# =============================================================================
def ingest_dataset(
    bucket,
    src_prefix,
    dataset,
):
    """
    Batch ingestion for an EDF dataset.
    """

    output_prefix = f"{OUTPUT_ROOT}/{dataset}"

    log("")
    log("=" * 70)
    log(f"Dataset : {dataset}")
    log(f"Source  : s3://{bucket}/{src_prefix}")
    log(f"Output  : s3://{bucket}/{output_prefix}")
    log("=" * 70)

    edf_keys = s3_list_keys(
        bucket,
        src_prefix,
        suffix=".edf",
    )

    log(f"Found {len(edf_keys)} EDF files")

    manifest = []

    success = 0
    skipped = 0
    failed = 0

    for index, edf_key in enumerate(edf_keys):

        log("")
        log(f"[{index+1}/{len(edf_keys)}]")

        row = process_one(

            bucket=bucket,

            edf_key=edf_key,

            output_prefix=output_prefix,

        )

        if row is None:

            skipped += 1
            continue

        manifest.append(row)

        success += 1

    manifest = pd.DataFrame(manifest)

    manifest_key = f"{output_prefix}/manifest.csv"

    s3_upload_csv(

        manifest,

        bucket,

        manifest_key,

    )

    log("")
    log("=" * 70)
    log("INGESTION COMPLETE")
    log("=" * 70)

    log(f"Success : {success}")
    log(f"Skipped : {skipped}")
    log(f"Failed  : {failed}")

    log(f"Manifest : s3://{bucket}/{manifest_key}")

    return manifest

# =============================================================================
# S3 HELPERS
# =============================================================================
def s3_list_keys(

    bucket,

    prefix,

    suffix=None,

):

    paginator = s3.get_paginator("list_objects_v2")

    keys = []

    for page in paginator.paginate(

        Bucket=bucket,

        Prefix=prefix,

    ):

        for obj in page.get("Contents", []):

            key = obj["Key"]

            if suffix:

                if key.lower().endswith(suffix):

                    keys.append(key)

            else:

                keys.append(key)

    return sorted(keys)

def s3_upload_csv(

    df,

    bucket,

    key,

):

    buffer = io.BytesIO()

    df.to_csv(

        buffer,

        index=False,

    )

    s3.put_object(

        Bucket=bucket,

        Key=key,

        Body=buffer.getvalue(),

        ContentType="text/csv",

    )

# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description="Unified EDF Ingestion",

    )

    parser.add_argument(

        "--bucket",

        default=BUCKET,

    )

    parser.add_argument(

        "--dataset",

        required=True,

    )

    parser.add_argument(

        "--src_prefix",

        required=True,

    )

    args = parser.parse_args()

    ingest_dataset(

        bucket=args.bucket,

        src_prefix=args.src_prefix,

        dataset=args.dataset,

    )