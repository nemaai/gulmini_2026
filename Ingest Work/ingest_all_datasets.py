"""
ingest_all_datasets.py — Unified EEG Dataset Ingestion to S3
=============================================================
Single script that handles every dataset in the NEMA training pipeline.
Each dataset has its own named section clearly labelled.

Datasets covered:
  ── Dementia ────────────────────────────────────────────────────────
  DS003800       | AD Auditory (Tehran) | .set       | 250Hz     | 19ch       | label=1
  DS004504/ADFTD | AHEPA Greece AD+FTD  | .set       | 500Hz     | 19ch       | label=1
  ADFSU          | Florida State AD     | .txt       | 128Hz     | 19ch       | label=1    # (GUL)
  BrainLat       | Latin America AD/FTD | .set       | 512Hz     | 128ch→19ch
  P-ADIC / ALZ   | Israeli AD+MCI       | .mat       | 500Hz     | 19ch       | label=1
  P-ADIC / HC    | Israeli Controls     | .mat       | 500Hz     | 19ch       | label=0
  Isfahan        | Iran MCI             | .set/.edf  | 200Hz     | 19ch       | label=1
  APAVA          | FieldTrip            | .mat       | 256Hz     | 16ch→19ch  | label=1    # (GUL)
  FIGSHARE       | AD Cohort B          | .mat       | 128/256Hz | 19ch       | label=AD/MCI/HC  # (GUL)
  ADSZ           | Alzheimer            | .out       | Unknown   | 19ch       | label=AD/HC       # (GUL)
  GENEEG         | WAVi MCI Dataset     | .eeg       | Unknown   | 17ch→19ch  | label=MCI/HC     # (GUL)
  TD-BRAIN       | SMC Controls         | BrainVision| 500Hz     | 33ch→19ch  | label=0          # (GUL)

  ── Controls ────────────────────────────────────────────────────────
  BACA / ds005385| Dortmund Longitudinal| .edf | 1000Hz | 64ch→19ch
  PEARL-Neuro      | Polish Healthy Controls   | BrainVision | 1000Hz     | 127ch→19ch  | label=0          # (GUL)

  ── Excluded ───────────────────────────────────────────────────────────────
Olfactory        | AD/MCI Olfactory Task     | .mat        | 200Hz      | 4ch         | EXCLUDED (task-based, non-resting, 4-channel)  # (GUL)

Output per file:
  s3://{BUCKET}/nema_final_used/all_npy_raw/{dataset}/{stem}.npy
  (19, T) float32, original units preserved — feed into data_pipeline.py next)

Run order:
  1. python ingest_all_datasets.py --dataset <name>   (or --all)
  2. python data_pipeline.py   (standardise: resample + preprocess + QC)
  3. python split_manifest.py  (create train/val/test splits)

Usage:
  python ingest_all_datasets.py --dataset BACA-ses2 --bucket dementia-research2025
  python ingest_all_datasets.py --dataset ADFSU    --local_root /data/ADFSU_EEG_data
  python ingest_all_datasets.py --all --bucket dementia-research2025
"""

import argparse
import io
import os
import sys
import tempfile
import traceback
from math import gcd
from pathlib import Path
from datetime import datetime, timezone

import boto3
import h5py
import mne
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

from collections import defaultdict
import tempfile
import traceback
import mne

from scipy.io import loadmat  # (GUL)
from convert_edf_batch import convert_raw_edf  #(GUL)


mne.set_log_level("WARNING")

# ─── GLOBAL CONFIG ────────────────────────────────────────────────────────────

BUCKET      = "dementia-research2025"
S3_RAW_ROOT = "nema_final_used/final_npy_raw"     # output root
S3_SRC_ROOT = "nema_final_used/original_data"                  # where raw source files live

s3 = boto3.client("s3")

# Standard 19-channel 10-20 order — all datasets must match this after ingestion
MASTER_19 = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5","P3","Pz","P4","T6",
    "O1","O2",
]

# Channel aliases across different EEG systems
CHANNEL_ALIASES = {
    "T7":"T3","T8":"T4","P7":"T5","P8":"T6",
    "FP1":"Fp1","FP2":"Fp2",
    "CZ":"Cz","FZ":"Fz","PZ":"Pz",
}

# ─── S3 HELPERS ───────────────────────────────────────────────────────────────

def s3_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key); return True
    except ClientError:
        return False

def s3_download(bucket, key, local_path):
    s3.download_file(bucket, key, str(local_path))

def s3_upload_npy(arr, bucket, key):
    buf = io.BytesIO()
    np.save(buf, arr)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(),
                  ContentType="application/octet-stream")

def s3_upload_csv(df, bucket, key):
    buf = io.BytesIO(df.to_csv(index=False).encode())
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue(),
                  ContentType="text/csv")

def s3_list_keys(bucket, prefix, suffix=None):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if suffix is None or k.endswith(suffix):
                keys.append(k)
    return sorted(keys)

def s3_read_tsv(bucket, key):
    data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(data), sep="\t")

def s3_read_excel(bucket, key):
    data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_excel(io.BytesIO(data))



def s3_read_csv(bucket, key):
    data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_csv(io.BytesIO(data))

# ─── CHANNEL HELPERS ─────────────────────────────────────────────────────────

def normalise_ch(name):
    return CHANNEL_ALIASES.get(name, CHANNEL_ALIASES.get(name.upper(), name))

def select_19ch_from_raw(raw):
    """
    Select and reorder MASTER_19 channels from an MNE Raw object.
    Applies aliases (T7→T3 etc). Raises if any channel missing.
    """
    norm_map = {normalise_ch(ch): ch for ch in raw.ch_names}
    picks    = []
    missing  = []
    for tch in MASTER_19:
        if tch in norm_map:
            picks.append(norm_map[tch])
        else:
            missing.append(tch)
    if missing:
        raise ValueError(f"Channels not found: {missing}\n"
                         f"Available: {raw.ch_names[:20]}")
    raw.pick_channels(picks, ordered=True)
    rename = {norm_map[t]: t for t in MASTER_19 if norm_map[t] != t}
    if rename:
        raw.rename_channels(rename)
    return raw

def select_19ch_from_npy(eeg, source_channels):
    """Reorder numpy array to MASTER_19 from known source channel list."""
    norm  = [normalise_ch(c) for c in source_channels]
    order = []
    for tch in MASTER_19:
        for si, sch in enumerate(norm):
            if sch.lower() == tch.lower():
                order.append(si); break
        else:
            raise ValueError(f"Channel '{tch}' not in source: {source_channels}")
    return eeg[order]

def reconstruct_midline(name, signals):
    """Reconstruct Fz/Cz/Pz as average of adjacent pair if missing."""
    pairs = {"Fz":("F3","F4"), "Cz":("C3","C4"), "Pz":("P3","P4")}
    if name in pairs:
        a, b = pairs[name]
        if a in signals and b in signals:
            return (signals[a] + signals[b]) / 2.0
    raise ValueError(f"Cannot reconstruct {name} — required channels missing")

# ─── LOGGING ─────────────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ─── LABEL MAPS ──────────────────────────────────────────────────────────────

BINARY_LABEL = {
    "AD":1,"MCI":1,"FTD":1,"DEMENTIA":1,"1":1,
    "HC":0,"CONTROL":0,"HEALTHY":0,"NORMAL":0,"0":0,
    "PD":None,"MS":None,"SZ":None,"DEP":None,  # excluded
}

def to_binary(dx):
    return BINARY_LABEL.get(str(dx).strip().upper(), None)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 1 — DS003800 (AD Auditory, Tehran)
#  Source: EEGLAB .set files, 19ch, 250Hz, eyes-open rest + auditory task
#  Label:  all subjects = dementia (label=1)
#          participants.tsv has Group: "A"=AD, "C"=HC, "F"=FTD
#  Note:   rest data for sub-06 and sub-13 are missing
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_DS003800(bucket=BUCKET):
    """
    DS003800 — AD Auditory Tehran dataset
    EEGLAB .set, 19ch, 250Hz
    Reads participants.tsv for labels (A=AD, C=HC, F=FTD)
    """
    dataset   = "ds003800-download"
    src_pfx   = f"{S3_SRC_ROOT}/{dataset}"
    out_pfx   = f"{S3_RAW_ROOT}/DS003800"

    log(f"=== DS003800 (AD Auditory Tehran) ===")

    # Load participant labels
    label_map = {}
    try:
        pts = s3_read_tsv(bucket, f"{src_pfx}/participants.tsv")
        dx_col = next((c for c in ["Group","diagnosis","Diagnosis"] if c in pts.columns), None)
        if dx_col:
            LABEL_MAP_AUDITORY = {"A":"AD","C":"HC","F":"FTD","-":None}
            for _, row in pts.iterrows():
                sid = str(row["participant_id"]).strip()
                dx  = LABEL_MAP_AUDITORY.get(str(row[dx_col]).strip(), None)
                if dx:
                    label_map[sid] = to_binary(dx)
    except Exception as e:
        log(f"  [WARN] Could not load participants.tsv: {e}")

    rows = []
    for key in s3_list_keys(bucket, src_pfx, suffix=".set"):
        try:
            subject_id = next((p for p in Path(key).parts if p.startswith("sub-")), None)
            file_stem  = Path(key).stem
            out_key    = f"{out_pfx}/{file_stem}.npy"

            if s3_exists(bucket, out_key):
                log(f"  [SKIP] {file_stem} already exists"); continue

            log(f"  {file_stem}  label={label_map.get(subject_id)}")

            with tempfile.TemporaryDirectory() as tmp:
                local_set = Path(tmp) / Path(key).name
                s3_download(bucket, key, local_set)
                # Try FDT companion
                fdt_key = key[:-4] + ".fdt"
                if s3_exists(bucket, fdt_key):
                    s3_download(bucket, fdt_key, Path(tmp) / Path(fdt_key).name)

                raw  = mne.io.read_raw_eeglab(str(local_set), preload=True, verbose=False)
                # DS003800 already has 19 channels in standard order
                eeg  = raw.get_data().astype(np.float32)
                sfreq = raw.info["sfreq"]

            s3_upload_npy(eeg, bucket, out_key)
            rows.append({"subject_id": subject_id, "stem": file_stem,
                         "dataset": "DS003800", "true_label": label_map.get(subject_id),
                         "original_sfreq": sfreq, "shape": str(eeg.shape),
                         "clean_npy": out_key})
            log(f"    → {eeg.shape}  {sfreq}Hz  saved")

        except Exception as e:
            log(f"  [ERROR] {key}: {e}"); traceback.print_exc()

    _save_manifest(rows, bucket, out_pfx, "DS003800")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 2 — DS004504 / ADFTD (AHEPA Greece)
#  Source: EEGLAB .set, 19ch, 500Hz, eyes-closed rest
#  Labels: participants.tsv Group: "Normal"=HC, "Mild AD"=AD,
#          "Moderate AD"=AD, "MCI"=MCI, "FTD"=FTD
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_DS004504(bucket=BUCKET):
    """
    DS004504 / ADFTD — AHEPA Greece dataset
    EEGLAB .set, 19ch, 500Hz, eyes-closed
    """
    dataset = "ds004504"
    src_pfx = f"{S3_SRC_ROOT}/{dataset}"
    out_pfx = f"{S3_RAW_ROOT}/DS004504"

    log("=== DS004504 (AHEPA Greece AD+FTD) ===")

    LABEL_MAP_ADFTD = {
        "Normal":"HC", "Mild AD":"AD", "Moderate AD":"AD",
        "MCI":"MCI", "FTD":"FTD", "AD":"AD",
    }
    label_map = {}
    try:
        pts = s3_read_tsv(bucket, f"{src_pfx}/participants.tsv")
        dx_col = next((c for c in ["Group","Diagnosis","diagnosis"] if c in pts.columns), None)
        if dx_col:
            for _, row in pts.iterrows():
                sid = str(row["participant_id"]).strip()
                raw_dx = str(row[dx_col]).strip()
                mapped = LABEL_MAP_ADFTD.get(raw_dx, raw_dx)
                label_map[sid] = to_binary(mapped)
    except Exception as e:
        log(f"  [WARN] participants.tsv: {e}")

    rows = []
    for key in s3_list_keys(bucket, src_pfx, suffix=".set"):
        try:
            subject_id = next((p for p in Path(key).parts if p.startswith("sub-")), Path(key).stem)
            file_stem  = Path(key).stem
            out_key    = f"{out_pfx}/{file_stem}.npy"

            if s3_exists(bucket, out_key):
                log(f"  [SKIP] {file_stem}"); continue

            log(f"  {file_stem}  label={label_map.get(subject_id)}")

            with tempfile.TemporaryDirectory() as tmp:
                local_set = Path(tmp) / Path(key).name
                s3_download(bucket, key, local_set)
                fdt_key = key[:-4] + ".fdt"
                if s3_exists(bucket, fdt_key):
                    s3_download(bucket, fdt_key, Path(tmp) / Path(fdt_key).name)

                raw   = mne.io.read_raw_eeglab(str(local_set), preload=True, verbose=False)
                raw   = select_19ch_from_raw(raw)
                eeg   = raw.get_data().astype(np.float32)
                sfreq = raw.info["sfreq"]

            s3_upload_npy(eeg, bucket, out_key)
            rows.append({"subject_id": subject_id, "stem": file_stem,
                         "dataset": "DS004504", "true_label": label_map.get(subject_id),
                         "original_sfreq": sfreq, "shape": str(eeg.shape),
                         "clean_npy": out_key})
            log(f"    → {eeg.shape}  {sfreq}Hz")

        except Exception as e:
            log(f"  [ERROR] {key}: {e}"); traceback.print_exc()

    _save_manifest(rows, bucket, out_pfx, "DS004504")
    return rows

############### NIDHI'S VERSION ############################################################
# # ═══════════════════════════════════════════════════════════════════════════════
# #  DATASET 3 — ADFSU (Florida State University)
# #  Source: Plain .txt files per channel, 128Hz native, 19ch
# #  Structure: root/AD|HEALTHY/state/patient_id/*.txt (one file per channel)
# #  Missing channels Fz/Cz/Pz reconstructed as average of neighbours
# #  Note: data is in microvolts from txt files
# # ═══════════════════════════════════════════════════════════════════════════════

# def ingest_ADFSU(local_root, bucket=BUCKET):
#     """
#     ADFSU — Florida State University AD dataset
#     Local .txt files (one per channel), 128Hz, 19ch
#     Reconstructs Fz/Cz/Pz from F3+F4, C3+C4, P3+P4 if missing

#     Args:
#         local_root: path to ADFSU_EEG_data directory
#     """
#     out_pfx = f"{S3_RAW_ROOT}/ADFSU"
#     log("=== ADFSU (Florida State University AD) ===")

#     root    = Path(local_root)
#     rows    = []

#     LABEL_MAP_ADFSU = {"AD":"1", "HEALTHY":"0"}

#     for dx_dir in sorted(root.iterdir()):
#         if not dx_dir.is_dir(): continue
#         dx    = LABEL_MAP_ADFSU.get(dx_dir.name.upper(), dx_dir.name)
#         label = to_binary(dx)

#         for state_dir in sorted(dx_dir.iterdir()):
#             if not state_dir.is_dir(): continue
#             state = state_dir.name
            
#             # Keep eyes-closed only — eyes-open causes alpha suppression
#             # which is physiologically incompatible with your other datasets
#             if state.lower() in ("eyes_open", "eyesopen", "eo", "open",
#                                 "eyes open", "ec1"):
#                 log(f"    [SKIP] {state} — eyes-open excluded")
#                 continue

#             # Check if it's eyes-closed
#             if state.lower() not in ("eyes_closed", "eyesclosed", "ec",
#                                     "closed", "eyes closed", "ec2", "rest"):
#                 # Unknown state — log but include with a warning
#                 log(f"    [WARN] Unknown state '{state}' — including, verify  manually")


#             for patient_dir in sorted(state_dir.iterdir()):
#                 if not patient_dir.is_dir(): continue
#                 subject_id = f"{dx_dir.name}_{state}_{patient_dir.name}"
#                 out_key    = f"{out_pfx}/{subject_id}.npy"
                
#                 if s3_exists(bucket, out_key):
#                     log(f"  [SKIP] {subject_id}"); continue

#                 try:
#                     # Load all channel .txt files
#                     signals = {}
#                     for txt in sorted(patient_dir.glob("*.txt")):
#                         ch_name = txt.stem.strip()
#                         sig     = np.loadtxt(str(txt), dtype=np.float32)
#                         signals[ch_name] = sig

#                     if not signals:
#                         log(f"  [SKIP] {subject_id}: no .txt files found"); continue

#                     # Build MASTER_19 — reconstruct midline if needed
#                     harmonised = []
#                     missing    = []
#                     for ch in MASTER_19:
#                         if ch in signals:
#                             harmonised.append(signals[ch])
#                         else:
#                             try:
#                                 harmonised.append(reconstruct_midline(ch, signals))
#                                 missing.append(ch)
#                             except ValueError:
#                                 log(f"  [WARN] {subject_id}: cannot reconstruct {ch}")
#                                 harmonised.append(np.zeros_like(list(signals.values())[0]))
#                                 missing.append(ch)

#                     eeg = np.vstack(harmonised).astype(np.float32)
#                     if eeg.shape[0] != 19:
#                         raise ValueError(f"Expected 19 channels, got {eeg.shape[0]}")

#                     s3_upload_npy(eeg, bucket, out_key)
#                     rows.append({"subject_id": subject_id, "stem": subject_id,
#                                  "dataset": "ADFSU", "true_label": label,
#                                  "condition": state,
#                                  "original_sfreq": 128, "shape": str(eeg.shape),
#                                  "missing_channels": ",".join(missing),
#                                  "clean_npy": out_key})
#                     log(f"  {subject_id}  {eeg.shape}  missing={missing or 'none'}")

#                 except Exception as e:
#                     log(f"  [ERROR] {subject_id}: {e}"); traceback.print_exc()

#     _save_manifest(rows, bucket, out_pfx, "ADFSU")
#     return rows
#####################################################################################################3

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 3 — ADFSU (Florida State University) (GUL)
#  Source: Plain .txt files stored on S3
#  Structure:
#      ADFSU_EEG_data/
#          AD/
#              Eyes_closed/
#              Eyes_open/
#          Healthy/
#              Eyes_closed/
#              Eyes_open/
#
#  One txt per channel
#  Missing Fz/Cz/Pz reconstructed from neighbours
#  Eyes-open excluded
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_ADFSU(bucket=BUCKET):

    out_pfx = f"{S3_RAW_ROOT}/ADFSU"
    src_pfx = f"{S3_SRC_ROOT}/ADFSU_EEG_data"

    log("=== ADFSU (Florida State University AD) ===")

    rows = []

    LABEL_MAP_ADFSU = {
        "AD": "1",
        "Healthy": "0",
    }

    for dx_name in ["AD", "Healthy"]:

        dx = LABEL_MAP_ADFSU[dx_name]
        label = to_binary(dx)


        # Keep only Eyes_closed recordings to maintain consistency with
        # the other resting-state datasets used in the training pipeline.
        for state in ["Eyes_closed"]:

            patient_root = f"{src_pfx}/{dx_name}/{state}"

            patient_ids = set()

            for key in s3_list_keys(bucket, patient_root):

                rel = key.replace(patient_root + "/", "")

                if "/" not in rel:
                    continue

                patient_ids.add(rel.split("/")[0])

            log(f"  {dx_name} / {state} : {len(patient_ids)} patients")

            for patient in sorted(patient_ids):

                subject_id = f"{dx_name}_{state}_{patient}"

                out_key = f"{out_pfx}/{subject_id}.npy"

                if s3_exists(bucket, out_key):
                    log(f"  [SKIP] {subject_id}")
                    continue

                try:

                    signals = {}

                    with tempfile.TemporaryDirectory() as tmp:

                        patient_prefix = (
                            f"{patient_root}/{patient}"
                        )

                        for key in s3_list_keys(bucket, patient_prefix):

                            if not key.endswith(".txt"):
                                continue

                            local_txt = (
                                Path(tmp) /
                                Path(key).name
                            )

                            s3_download(
                                bucket,
                                key,
                                local_txt
                            )

                            ch_name = local_txt.stem.strip()

                            signals[ch_name] = np.loadtxt(
                                local_txt,
                                dtype=np.float32,
                            )

                    if not signals:
                        log(f"  [SKIP] {subject_id}: no txt files")
                        continue
                    
                    # Build MASTER_19 (GUL)
                    harmonised = []
                    missing = []

                    for ch in MASTER_19:

                        if ch in signals:

                            harmonised.append(signals[ch])

                        else:

                            try:

                                harmonised.append(
                                    reconstruct_midline(ch, signals)
                                )

                                missing.append(ch)

                            except ValueError:

                                log(
                                    f"  [WARN] {subject_id}: "
                                    f"cannot reconstruct {ch}"
                                )

                                harmonised.append(
                                    np.zeros_like(
                                        next(iter(signals.values()))
                                    )
                                )

                                missing.append(ch)

                    eeg = np.vstack(harmonised).astype(np.float32)

                    if eeg.shape[0] != 19:
                        raise ValueError(
                            f"Expected 19 channels, got {eeg.shape[0]}"
                        )

                    s3_upload_npy(
                        eeg,
                        bucket,
                        out_key,
                    )

                    rows.append({

                        "subject_id": subject_id,
                        "stem": subject_id,
                        "dataset": "ADFSU",
                        "true_label": label,
                        "condition": state,
                        "original_sfreq": 256,
                        "shape": str(eeg.shape),
                        "missing_channels": ",".join(missing) if missing else "none",
                        "clean_npy": out_key,

                    })

                    log(
                        f"  {subject_id}  "
                        f"{eeg.shape}  "
                        f"missing={missing or 'none'}"
                    )

                except Exception as e:

                    log(f"  [ERROR] {subject_id}: {e}")

                    traceback.print_exc()

    _save_manifest(
        rows,
        bucket,
        out_pfx,
        "ADFSU",
    )

    return rows

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 4 — BrainLat (Latin America)
#  Source: EEGLAB .set, 128-channel BioSemi Active II, various sfreq
#  Channel selection: nearest-neighbour mapping BioSemi128 → 10-20 standard
#  Labels: cohort folder name: 1_AD, 2_FTD, 3_PD, 4_MS, 5_HC
#  Note: PD and MS excluded (not dementia, not healthy aging)
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_BrainLat(local_root, bucket=BUCKET):
    """
    BrainLat — Latin America multi-disease EEG
    EEGLAB .set, 128-channel BioSemi → 19-channel via nearest neighbour
    Excludes PD and MS subjects

    Args:
        local_root: path to BrainLat directory containing 1_AD, 2_FTD etc.
    """
    out_pfx = f"{S3_RAW_ROOT}/BrainLat"
    log("=== BrainLat (Latin America AD/FTD/HC) ===")

    LABEL_MAP_BRAINLAT = {
        "1_AD":1,"2_bvFTD":1,"3_PD":None,"4_MS":None,"5_HC":"0"
    }

    # Build BioSemi128 → 10-20 nearest-neighbour channel map (once)
    log("  Building BioSemi128 → 10-20 channel map …")
    try:
        biosemi = mne.channels.make_standard_montage("biosemi128")
        ten20   = mne.channels.make_standard_montage("standard_1020")
        brainlat_map = {}
        for tch in MASTER_19:
            if tch not in ten20.get_positions()["ch_pos"]:
                continue
            target_xyz = ten20.get_positions()["ch_pos"][tch]
            best_ch, best_dist = None, 999
            for bio_ch, bio_xyz in biosemi.get_positions()["ch_pos"].items():
                d = np.linalg.norm(target_xyz - bio_xyz)
                if d < best_dist:
                    best_dist = d; best_ch = bio_ch
            brainlat_map[tch] = best_ch
        log(f"  Channel map: {brainlat_map}")
    except Exception as e:
        log(f"  [ERROR] Could not build channel map: {e}")
        return []

    root = Path(local_root)
    rows = []

    for cohort_dir in sorted(root.iterdir()):
        if not cohort_dir.is_dir(): continue
        dx_raw = cohort_dir.name
        dx     = LABEL_MAP_BRAINLAT.get(dx_raw)
        if dx is None:
            log(f"  [SKIP] {dx_raw} (excluded class)"); continue
        label  = to_binary(dx)
        log(f"  Cohort: {dx_raw} ({dx}, label={label})")

        for set_file in sorted(cohort_dir.rglob("*_eeg.set")):
            subject_id = set_file.parent.parent.name
            out_key    = f"{out_pfx}/{subject_id}.npy"

            if s3_exists(bucket, out_key):
                log(f"    [SKIP] {subject_id}"); continue

            try:
                raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
                available = raw.ch_names

                # Select BioSemi channels corresponding to MASTER_19
                picks   = []
                missing = []
                for tch in MASTER_19:
                    bio_ch = brainlat_map.get(tch)
                    if bio_ch and bio_ch in available:
                        picks.append(available.index(bio_ch))
                    else:
                        missing.append(tch)

                if missing:
                    log(f"    [WARN] {subject_id}: missing {missing}"); continue

                data = raw.get_data()[picks].astype(np.float32)
                # Rebuild with MASTER_19 names for downstream pipeline
                info   = mne.create_info(MASTER_19, sfreq=raw.info["sfreq"], ch_types="eeg")
                raw19  = mne.io.RawArray(data, info, verbose=False)
                eeg    = raw19.get_data().astype(np.float32)
                sfreq  = raw.info["sfreq"]

                s3_upload_npy(eeg, bucket, out_key)
                rows.append({"subject_id": subject_id, "stem": subject_id,
                             "dataset": "BrainLat", "true_label": label,
                             "original_sfreq": sfreq, "shape": str(eeg.shape),
                             "clean_npy": out_key})
                log(f"    {subject_id}  {eeg.shape}  {sfreq}Hz")

            except Exception as e:
                log(f"    [ERROR] {subject_id}: {e}"); traceback.print_exc()

    _save_manifest(rows, bucket, out_pfx, "BrainLat")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 5 — P-ADIC Dementia (ALZ + MCI)
#  Source: MATLAB .mat (HDF5 v7.3), 19ch, 500Hz
#  Structure: f["alz_r"]["G"][0, i] → reference to (19,T) EEG per subject
#             f["mci_r"]["G"][0, i] → same for MCI
#  Units: microvolts (Nihon Kohden) — convert to Volts before saving
#  Labels: ALZ→1, MCI→1
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_PADIC_dementia(local_mat_dir, bucket=BUCKET):
    """
    P-ADIC — Israeli EEG dataset (AD + MCI subjects)
    MATLAB v7.3 HDF5, 19ch, 500Hz, μV → converted to V

    Args:
        local_mat_dir: directory containing alz_c1_new.mat and mci_c1_new.mat
    """
    import h5py
    from scipy import signal as scipy_signal

    out_pfx = f"{S3_RAW_ROOT}/P-ADIC"
    log("=== P-ADIC Dementia (ALZ + MCI, Israel) ===")

    mat_dir = Path(local_mat_dir)

    # Each .mat has a group key, G is (3, N_subjects)
    # Row 0 of G = HDF5 reference to (19,T) or (T,19) EEG array
    TASKS = [
        ("alz_c1_new.mat", "alz_r", "AD", 1),
        ("mci_c1_new.mat", "mci_r", "MCI", 1),
    ]

    rows = []

    for fname, group_key, prefix, label in TASKS:
        mat_path = mat_dir / fname
        if not mat_path.exists():
            log(f"  [SKIP] {fname} not found at {mat_path}"); continue

        log(f"  Loading {fname} ({prefix}, label={label}) …")

        try:
            with h5py.File(str(mat_path), "r") as f:
                grp        = f[group_key]
                G          = grp["G"]
                n_subjects = G.shape[1]
                log(f"  Group='{group_key}'  G.shape={G.shape}  ({n_subjects} subjects)")

                for i in range(n_subjects):
                    stem    = f"{prefix}_{i:03d}"
                    out_key = f"{out_pfx}/{stem}.npy"

                    if s3_exists(bucket, out_key):
                        log(f"    [SKIP] {stem}"); continue

                    try:
                        ref = G[0, i]
                        eeg = f[ref][()].astype(np.float32)

                        # Ensure (19, T)
                        if eeg.ndim != 2:
                            log(f"    [SKIP] {stem}: ndim={eeg.ndim}"); continue
                        if eeg.shape[0] != 19:
                            if eeg.shape[1] == 19:
                                eeg = eeg.T
                            else:
                                log(f"    [SKIP] {stem}: shape {eeg.shape}"); continue

                        duration = eeg.shape[1] / 500
                        if duration < 10:
                            log(f"    [SKIP] {stem}: too short ({duration:.1f}s)"); continue

                        # μV → V (Nihon Kohden stores in microvolts)
                        if eeg.std() > 1.0:
                            eeg = (eeg / 1e6).astype(np.float32)

                        s3_upload_npy(eeg, bucket, out_key)
                        rows.append({"subject_id": stem, "stem": stem,
                                     "dataset": "P-ADIC", "true_label": label,
                                     "original_sfreq": 500, "shape": str(eeg.shape),
                                     "original_units": "microvolts_converted_to_V",
                                     "clean_npy": out_key})
                        log(f"    [{i+1}/{n_subjects}] {stem}  {eeg.shape}  ({duration:.1f}s)")

                    except Exception as e:
                        log(f"    [ERROR] subject {i}: {e}")

        except Exception as e:
            log(f"  [ERROR] {fname}: {e}"); traceback.print_exc()

    _save_manifest(rows, bucket, out_pfx, "P-ADIC_dementia")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 6 — P-ADIC Controls (HC)
#  Source: MATLAB v7.3 HDF5, controls_c1_new.mat
#  Structure: EEG arrays are in f["#refs#"] as (T,19) float64
#             Iterate all #refs# entries, keep those with shape (T, 19) and T>1000
#  Units: microvolts → Volts
#  Labels: HC → 0
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_PADIC_controls(local_mat_dir, bucket=BUCKET):
    """
    P-ADIC — Israeli EEG dataset (healthy controls)
    MATLAB v7.3 HDF5, 19ch, 500Hz, μV → V
    Controls stored in #refs# group, shape (T, 19) per subject

    Args:
        local_mat_dir: directory containing controls_c1_new.mat
    """
    import h5py

    out_pfx  = f"{S3_RAW_ROOT}/P-ADIC"
    mat_path = Path(local_mat_dir) / "controls_c1_new.mat"
    log("=== P-ADIC Controls (HC, Israel) ===")

    if not mat_path.exists():
        log(f"  [SKIP] controls_c1_new.mat not found at {mat_path}")
        return []

    rows = []

    with h5py.File(str(mat_path), "r") as f:
        refs     = f["#refs#"]
        all_keys = sorted(refs.keys())
        log(f"  Total #refs# entries: {len(all_keys)}")

        # Filter: shape (T, 19), T > 1000, numeric dtype
        eeg_keys = [
            k for k in all_keys
            if refs[k].ndim == 2
            and refs[k].shape[1] == 19
            and refs[k].shape[0] > 1000
            and refs[k].dtype.kind in ("f","i","u")
        ]
        log(f"  EEG arrays (T×19): {len(eeg_keys)}")

        for i, k in enumerate(eeg_keys):
            stem    = f"HC_{i:03d}"
            out_key = f"{out_pfx}/{stem}.npy"

            if s3_exists(bucket, out_key):
                log(f"    [SKIP] {stem}"); continue

            try:
                # (T, 19) → transpose to (19, T)
                eeg = refs[k][()].T.astype(np.float32)
                assert eeg.shape[0] == 19

                duration = eeg.shape[1] / 500
                if duration < 10:
                    log(f"    [SKIP] {stem}: too short ({duration:.1f}s)"); continue

                # μV → V
                if eeg.std() > 1.0:
                    eeg = (eeg / 1e6).astype(np.float32)

                s3_upload_npy(eeg, bucket, out_key)
                rows.append({"subject_id": stem, "stem": stem,
                             "dataset": "P-ADIC", "true_label": 0,
                             "original_sfreq": 500, "shape": str(eeg.shape),
                             "original_units": "microvolts_converted_to_V",
                             "refs_key": k, "clean_npy": out_key})
                log(f"    [{i+1}/{len(eeg_keys)}] {stem}  {eeg.shape}  ({duration:.1f}s)")

            except Exception as e:
                log(f"    [ERROR] {k}: {e}")

    _save_manifest(rows, bucket, out_pfx, "P-ADIC_controls")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 7 — Isfahan MCI (Iran)
#  Source: EEGLAB .set or EDF, 200Hz, 19ch standard 10-20
#  Labels: from participants.tsv or folder structure
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_Isfahan(bucket=BUCKET):
    """
    Isfahan MCI — Iran resting-state EEG dataset
    EEGLAB .set or EDF, 200Hz, 19ch
    Labels from participants.tsv (MCI=1, HC=0)
    """
    dataset = "Isfahan MCI _EEG_2"
    src_pfx = f"{S3_SRC_ROOT}/{dataset}"
    out_pfx = f"{S3_RAW_ROOT}/Isfahan"
    log("=== Isfahan MCI (Iran) ===")

    LABEL_MAP_ISFAHAN = {
        "MCI":"MCI","HC":"HC","CONTROL":"HC","HEALTHY":"HC",
        "1":"MCI","0":"HC", "NORMAL":"HC"
    }
    label_map = {}
    try:
        pts = s3_read_excel(bucket, f"{src_pfx}/States.xlsx")
        
        dx_col = next((c for c in [ "status"] if c in pts.columns), None)
        # print(dx_col , "-----")
        if dx_col:
            for _, row in pts.iterrows():
                sid = str(row["status"]).strip()
                dx  = LABEL_MAP_ISFAHAN.get(str(row[dx_col]).strip().upper(), None)
                if dx:
                    label_map[sid] = to_binary(dx)
                    # print(label_map[sid], "-=-")
    except Exception as e:
        log(f"  [WARN] participants.tsv: {e}")

    rows = []
    # Try both .set and .edf files
    for suffix in [".set", ".edf"]:
        for key in s3_list_keys(bucket, src_pfx, suffix=suffix):
            try:
                subject_id = next((p for p in Path(key).parts if p.startswith("sub-")),
                                  Path(key).stem)
                file_stem  = Path(key).stem
                out_key    = f"{out_pfx}/{file_stem}.npy"

                if s3_exists(bucket, out_key):
                    log(f"  [SKIP] {file_stem}"); continue

                log(f"  {file_stem}  label={label_map.get(subject_id)}")

                with tempfile.TemporaryDirectory() as tmp:
                    local_f = Path(tmp) / Path(key).name
                    s3_download(bucket, key, local_f)

                    if suffix == ".set":
                        fdt_key = key[:-4] + ".fdt"
                        if s3_exists(bucket, fdt_key):
                            s3_download(bucket, fdt_key, Path(tmp)/Path(fdt_key).name)
                        raw = mne.io.read_raw_eeglab(str(local_f), preload=True, verbose=False)
                    else:
                        raw = mne.io.read_raw_edf(str(local_f), preload=True, verbose=False)

                    raw   = select_19ch_from_raw(raw)
                    eeg   = raw.get_data().astype(np.float32)
                    sfreq = raw.info["sfreq"]

                s3_upload_npy(eeg, bucket, out_key)
                rows.append({"subject_id": subject_id, "stem": file_stem,
                             "dataset": "Isfahan", "true_label": label_map.get(subject_id),
                             "original_sfreq": sfreq, "shape": str(eeg.shape),
                             "clean_npy": out_key})
                log(f"    → {eeg.shape}  {sfreq}Hz")

            except Exception as e:
                log(f"  [ERROR] {key}: {e}"); traceback.print_exc()

    _save_manifest(rows, bucket, out_pfx, "Isfahan")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 8 — BACA / ds005385 (Dortmund Longitudinal)
#  Source: EDF, 64ch, 1000Hz, healthy controls, session 1 + session 2
#  Channel selection: 19 standard 10-20 from 64-channel BioSemi
#  Condition: eyes-closed pre-task only
#  Labels: all HC → 0
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_BACA_old(bucket=BUCKET, session="ses-1"):
    """
    BACA / ds005385 — Dortmund 5-year longitudinal healthy control EEG
    EDF, 64ch, 1000Hz → select 19ch, extract eyes-closed pre-task
    All subjects label=0 (healthy control)

    Args:
        session: "ses-1" (training) or "ses-2" (external validation)
    """
    dataset = "ds005385"
    src_pfx = f"{S3_SRC_ROOT}/{dataset}"
    out_pfx = f"{S3_RAW_ROOT}/BACA_{session.replace('-','')}"
    log(f"=== BACA ds005385 ({session}, Dortmund Longitudinal) ===")

    rows = []
    # EDF files follow BIDS naming: sub-XX_ses-X_task-EyesClosed_...edf
    for key in s3_list_keys(bucket, src_pfx, suffix=".edf"):
        if session not in key: continue
        if "EyesClosed" not in key and "eyesclosed" not in key.lower(): continue

        file_stem  = Path(key).stem
        subject_id = next((p for p in Path(key).parts if p.startswith("sub-")),
                          file_stem)
        session_id = next((p for p in Path(key).parts if p.startswith("ses-")), session)
        out_key    = f"{out_pfx}/{file_stem}.npy"

        if s3_exists(bucket, out_key):
            log(f"  [SKIP] {file_stem}"); continue

        log(f"  {file_stem}")

        try:
            with tempfile.TemporaryDirectory() as tmp:
                local_f = Path(tmp) / Path(key).name
                s3_download(bucket, key, local_f)
                raw = mne.io.read_raw_edf(str(local_f), preload=True, verbose=False)
                raw = select_19ch_from_raw(raw)
                eeg = raw.get_data().astype(np.float32)
                sfreq = raw.info["sfreq"]

            s3_upload_npy(eeg, bucket, out_key)
            rows.append({"subject_id": subject_id, "stem": file_stem,
                         "dataset": f"BACA_{session}", "true_label": 0,
                         "session": session_id, "condition": "eyes_closed_rest",
                         "original_sfreq": sfreq, "shape": str(eeg.shape),
                         "clean_npy": out_key})
            log(f"    → {eeg.shape}  {sfreq}Hz")

        except Exception as e:
            log(f"  [ERROR] {key}: {e}"); traceback.print_exc()

    _save_manifest(rows, bucket, out_pfx, f"BACA_{session}")
    return rows


def ingest_BACA(bucket=BUCKET):
    """
    BACA / ds005385

    TRAIN:
        Subjects having ONLY ses-1

    TEST:
        Subjects having BOTH ses-1 and ses-2
        (both sessions are exported)

    This prevents subject leakage.
    """

    dataset = "ds005385"
    src_pfx = f"{S3_SRC_ROOT}/{dataset}"

    train_out = f"{S3_RAW_ROOT}/BACA_train"
    test_out = f"{S3_RAW_ROOT}/BACA_longitudinal"

    log("=== BACA Longitudinal Split ===")

    # ---------------------------------------------------
    # Discover sessions per subject
    # ---------------------------------------------------

    subject_files = defaultdict(dict)

    for key in s3_list_keys(bucket, src_pfx, suffix=".edf"):

        if "EyesClosed" not in key and "eyesclosed" not in key.lower():
            continue

        parts = Path(key).parts

        subject = next(
            p for p in parts
            if p.startswith("sub-")
        )

        session = next(
            p for p in parts
            if p.startswith("ses-")
        )

        subject_files[subject][session] = key

    # ---------------------------------------------------
    # Build cohorts
    # ---------------------------------------------------

    train_subjects = []
    paired_subjects = []

    for subject, sessions in subject_files.items():

        has1 = "ses-1" in sessions
        has2 = "ses-2" in sessions

        if has1 and has2:
            paired_subjects.append(subject)

        elif has1:
            train_subjects.append(subject)

    log(f"Training subjects      : {len(train_subjects)}")
    log(f"Longitudinal subjects  : {len(paired_subjects)}")

    # ---------------------------------------------------
    # Export helper
    # ---------------------------------------------------

    def export(key, out_prefix, dataset_name, session):

        rows = []

        file_stem = Path(key).stem

        subject = next(
            p for p in Path(key).parts
            if p.startswith("sub-")
        )

        out_key = f"{out_prefix}/{file_stem}.npy"

        if s3_exists(bucket, out_key):
            return []

        try:

            with tempfile.TemporaryDirectory() as tmp:

                local = Path(tmp) / Path(key).name

                s3_download(bucket, key, local)

                raw = mne.io.read_raw_edf(
                    str(local),
                    preload=True,
                    verbose=False
                )

                raw = select_19ch_from_raw(raw)

                eeg = raw.get_data().astype(np.float32)

                sfreq = raw.info["sfreq"]

            s3_upload_npy(
                eeg,
                bucket,
                out_key
            )

            rows.append({
                "subject_id": subject,
                "stem": file_stem,
                "dataset": dataset_name,
                "true_label": 0,
                "session": session,
                "condition": "eyes_closed_rest",
                "original_sfreq": sfreq,
                "shape": str(eeg.shape),
                "clean_npy": out_key
            })

            log(f"  ✓ {file_stem}")

        except Exception as e:

            log(f"[ERROR] {key}")

            traceback.print_exc()

        return rows

    # ---------------------------------------------------
    # TRAIN
    # ---------------------------------------------------

    train_rows = []

    for subject in train_subjects:

        key = subject_files[subject]["ses-1"]

        train_rows.extend(
            export(
                key,
                train_out,
                "BACA_train",
                "ses-1"
            )
        )

    # ---------------------------------------------------
    # TEST (paired only)
    # ---------------------------------------------------

    test_rows = []

    for subject in paired_subjects:

        for session in ["ses-1", "ses-2"]:

            test_rows.extend(
                export(
                    subject_files[subject][session],
                    test_out,
                    "BACA_longitudinal",
                    session
                )
            )

    # ---------------------------------------------------
    # Save manifests
    # ---------------------------------------------------

    _save_manifest(
        train_rows,
        bucket,
        train_out,
        "BACA_train"
    )

    _save_manifest(
        test_rows,
        bucket,
        test_out,
        "BACA_longitudinal"
    )

    log("=================================")
    log(f"Training EEGs      : {len(train_rows)}")
    log(f"Longitudinal EEGs  : {len(test_rows)}")
    log("=================================")

    return train_rows, test_rows

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 9 — PEARL-Neuro (Polish Healthy Controls)
#  Source: BrainVision .vhdr/.eeg/.vmrk on S3, 128ch, 1000Hz
#  Channel selection: 19 standard 10-20 from 128-channel BrainProducts
#  Condition: eyes-closed segment (last 6 min of rest task)
#  Age filter: 40–80 years (default 50–63 = full PEARL age range)
#  Labels: all HC → 0
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_PEARL(bucket=BUCKET, age_min=50, age_max=80):
    """
    PEARL-Neuro — Polish healthy aging EEG (Brain Products, 128ch, 1000Hz)
    Streams BrainVision triplet from S3, selects 19ch, extracts eyes-closed
    All subjects label=0

    Args:
        age_min, age_max: age filter (PEARL subjects are 50–63)
    """
    src_pfx = f"{S3_SRC_ROOT}/pearl_neuro"
    out_pfx = f"{S3_RAW_ROOT}/PEARL-Neuro"
    log("=== PEARL-Neuro (Polish Healthy Controls) ===")

    TARGET_PEARL = [
        "Fp1","Fp2","F7","F3","Fz","F4","F8",
        "T7","C3","Cz","C4","T8",
        "P7","P3","Pz","P4","P8",
        "O1","O2",
    ]
    RENAME_PEARL = {"T7":"T3","T8":"T4","P7":"T5","P8":"T6"}
    MIN_EC_SEC   = 60.0

    # Load age lookup
    age_lookup = {}
    try:
        pts = s3_read_tsv(bucket, f"{src_pfx}/participants.tsv")
        pts.columns = pts.columns.str.lower()
        id_col = next((c for c in pts.columns if "participant" in c or "subject" in c), pts.columns[0])
        if "age" in pts.columns:
            for _, row in pts.iterrows():
                age_lookup[str(row[id_col]).strip()] = float(row["age"])
    except Exception as e:
        log(f"  [WARN] participants.tsv: {e}")

    # Discover subjects
    paginator = s3.get_paginator("list_objects_v2")
    subjects  = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=src_pfx, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            folder = cp["Prefix"].rstrip("/").split("/")[-1]
            if folder.startswith("sub-"):
                subjects.add(folder)
    subjects = sorted(subjects)
    log(f"  Found {len(subjects)} subjects")

    rows = []
    for sub in subjects:
        age = age_lookup.get(sub, age_lookup.get(sub.replace("sub-",""), None))
        if age is not None and not (age_min <= float(age) <= age_max):
            continue

        eeg_prefix = f"{src_pfx}/{sub}/eeg"
        stem       = f"{sub}_rest_ec"
        out_key    = f"{out_pfx}/{stem}.npy"

        if s3_exists(bucket, out_key):
            log(f"  [SKIP] {sub}"); continue

        log(f"  {sub}  age={age}")

        try:
            base = f"{eeg_prefix}/{sub}_task-rest_eeg"
            with tempfile.TemporaryDirectory() as tmp:
                for ext in [".vhdr",".eeg",".vmrk"]:
                    k = base + ext
                    local = Path(tmp) / f"{sub}_task-rest_eeg{ext}"
                    try:
                        s3_download(bucket, k, local)
                    except ClientError:
                        if ext == ".vhdr":
                            raise FileNotFoundError(f".vhdr not found: {k}")

                raw = mne.io.read_raw_brainvision(
                    str(Path(tmp) / f"{sub}_task-rest_eeg.vhdr"),
                    preload=True, verbose=False
                )

                # Select 19 channels
                available = raw.ch_names
                ch_map = {}
                for tch in TARGET_PEARL:
                    for cand in [tch, tch.upper()]:
                        if cand in available:
                            ch_map[tch] = cand; break
                    if tch not in ch_map:
                        for src_ch, dst in RENAME_PEARL.items():
                            if dst == tch and src_ch in available:
                                ch_map[tch] = src_ch; break

                missing = [t for t in TARGET_PEARL if t not in ch_map]
                if missing:
                    raise ValueError(f"Channels missing: {missing}")

                raw.pick_channels([ch_map[t] for t in TARGET_PEARL], ordered=True)
                rename = {ch_map[t]: t for t in TARGET_PEARL if ch_map[t] != t}
                if rename:
                    raw.rename_channels(rename)
                # Rename to MASTER_19 names
                raw.rename_channels({t: RENAME_PEARL.get(t, t) for t in TARGET_PEARL
                                     if t in RENAME_PEARL})

                # Extract eyes-closed segment
                sfreq     = raw.info["sfreq"]
                total_sec = raw.n_times / sfreq

                # PEARL-Neuro task-rest does not expose explicit Eyes Closed markers.
                # Use the final 6 minutes of the resting recording.

                sfreq = raw.info["sfreq"]
                total_sec = raw.n_times / sfreq

                ec_onset = int(max(0, total_sec - 360.0) * sfreq)

                log(
                    f"    Using final {(raw.n_times - ec_onset)/sfreq:.0f}s "
                    "of resting recording"
                )

                eeg_ec, _ = raw[:, ec_onset:]

                eeg_ec, _ = raw[:, ec_onset:]
                duration  = eeg_ec.shape[1] / sfreq
                if duration < MIN_EC_SEC:
                    raise ValueError(f"Eyes-closed too short: {duration:.1f}s")

                eeg = eeg_ec.astype(np.float32)
                # MNE BrainVision output is in Volts — no conversion needed

            s3_upload_npy(eeg, bucket, out_key)
            rows.append({"subject_id": sub, "stem": stem,
                         "dataset": "PEARL-Neuro", "true_label": 0,
                         "age": age, "condition": "eyes_closed_rest",
                         "original_sfreq": sfreq, "shape": str(eeg.shape),
                         "clean_npy": out_key})
            log(f"    → {eeg.shape}  {sfreq}Hz  {duration:.0f}s EC")

        except Exception as e:
            log(f"  [ERROR] {sub}: {e}"); traceback.print_exc()

    _save_manifest(rows, bucket, out_pfx, "PEARL-Neuro")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  Gul - DATASET 10 — APAVA (Alzheimer's Progression Assessment via EEG)
#  Source: MATLAB FieldTrip (.mat), 256Hz native, 16ch
#  Structure: APAVA/AD_Data/preproctrials*.mat
#  Missing channels Fz/Cz/Pz reconstructed as average of neighbours
#  One output .npy generated per file
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_APAVA(bucket=BUCKET):
    """
    APAVA AD dataset
    One .mat -> One .npy (all trials concatenated)
    """

    dataset = "APAVA"
    src_pfx = f"{S3_SRC_ROOT}/{dataset}/AD_Data"
    out_pfx = f"{S3_RAW_ROOT}/APAVA"

    log("=== APAVA (AD Dataset) ===")

    rows = []

    for key in s3_list_keys(bucket, src_pfx, suffix=".mat"):

        if "preproctrials" not in Path(key).name.lower():
            continue

        try:

            subject_id = Path(key).stem

            with tempfile.TemporaryDirectory() as tmp:

                local_mat = Path(tmp) / Path(key).name

                s3_download(bucket, key, local_mat)

                mat = loadmat(
                    str(local_mat),
                    squeeze_me=True,
                    struct_as_record=False,
                )

            data = mat["data"]

            labels = list(data.label)
            sfreq = int(data.fsample)

            all_trials = []
            missing_all = set()

            for trial in data.trial:

                signals = {}

                for i, ch in enumerate(labels):
                    signals[ch] = trial[i].astype(np.float32)

                # Reconstruct missing midline channels
                for ch in ["Fz", "Cz", "Pz"]:
                    if ch not in signals:
                        signals[ch] = reconstruct_midline(ch, signals)

                harmonised = []
                missing = []

                for ch in MASTER_19:

                    if ch in signals:
                        harmonised.append(signals[ch])

                    else:
                        harmonised.append(
                            np.zeros_like(next(iter(signals.values())))
                        )
                        missing.append(ch)

                eeg = np.vstack(harmonised).astype(np.float32)

                if eeg.shape[0] != 19:
                    raise ValueError(
                        f"Expected 19 channels, got {eeg.shape[0]}"
                    )

                all_trials.append(eeg)
                missing_all.update(missing)

            # -------------------------------------------------
            # Concatenate all trials into one recording
            # -------------------------------------------------

            combined = np.concatenate(
                all_trials,
                axis=1,
            )

            stem = subject_id
            out_key = f"{out_pfx}/{stem}.npy"

            if s3_exists(bucket, out_key):
                log(f"  [SKIP] {stem}")
                continue

            s3_upload_npy(
                combined,
                bucket,
                out_key,
            )

            rows.append({
                "subject_id": subject_id,
                "stem": stem,
                "dataset": "APAVA",
                "true_label": 1,
                "original_sfreq": sfreq,
                "shape": str(combined.shape),
                "missing_channels": ",".join(sorted(missing_all)),
                "clean_npy": out_key,
            })

            log(
                f"  {stem}  {combined.shape}  "
                f"missing={sorted(missing_all) or 'none'}"
            )

        except Exception as e:

            log(f"  [ERROR] {subject_id}: {e}")
            traceback.print_exc()

    _save_manifest(rows, bucket, out_pfx, "APAVA")

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 11 — FIGSHARE AD COHORT B (GUL)
#  Source: MAT files stored on S3
#
#  Structure:
#      AD/
#          AD1/
#              1.mat
#              2.mat
#              ...
#      CONTROL/
#          normal1/
#          normal2/
#      MCI/
#          ...
#
#  Each MAT contains:
#      export -> (samples,19)
#
#  Output:
#      (19,samples) float32
#
#  MCI skipped to preserve binary classification
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_FIGSHARE(bucket=BUCKET):

    # out_pfx = f"{S3_RAW_ROOT}"
    src_pfx = f"{S3_SRC_ROOT}/Figshare AD Cohort B "

    log("=== FIGSHARE AD COHORT B ===")

    rows_128 = []      
    rows_256 = []      
    

    LABEL_MAP = {
        "AD": "1",
        "MCI": "1",
        "CONTROL": "0",
    }

    for dx_name in ["AD", "MCI", "CONTROL"]:

        label = to_binary(LABEL_MAP[dx_name])

        dx_root = f"{src_pfx}/{dx_name}"

        subject_ids = set()

        for key in s3_list_keys(bucket, dx_root):

            rel = key.replace(dx_root + "/", "")

            if "/" not in rel:
                continue

            subject_ids.add(rel.split("/")[0])

        log(f"  {dx_name}: {len(subject_ids)} subjects")

        for subject in sorted(subject_ids):

            # Process one subject per class for testing (GUL)
            # if (
            #     (dx_name == "AD" and subject != "AD1") or
            #     (dx_name == "MCI" and subject != "MCI1") or
            #     (dx_name == "CONTROL" and subject != "normal1")
            # ):
            #     continue

            subject_root = f"{dx_root}/{subject}"

            for key in s3_list_keys(bucket, subject_root):

                if not key.endswith(".mat"):
                    continue

                # Skip FIR filter files
                if Path(key).name.lower().endswith("fir.mat"):
                    continue

                stem = Path(key).stem
                prefix = "HC" if dx_name == "CONTROL" else ""

                subject_id = f"{prefix}{subject}_{stem}"

                # Sampling frequency from README

                if dx_name == "AD":

                    num = int(subject.replace("AD", ""))

                    sfreq = 256 if num <= 34 else 128

                elif dx_name == "MCI":

                    num = int(subject.replace("MCI", ""))

                    sfreq = 256 if num <= 4 else 128

                elif dx_name == "CONTROL":

                    sfreq = 128

                else:

                    sfreq = None


                out_key = (
                    f"{S3_RAW_ROOT}/FIGSHARE-{sfreq}Hz/"
                    f"{prefix}{subject}_{stem}.npy"
                )

                if s3_exists(bucket, out_key):
                    log(f"  [SKIP] {subject_id}")
                    continue

                try:

                    with tempfile.TemporaryDirectory() as tmp:

                        local_mat = Path(tmp) / Path(key).name

                        s3_download(
                            bucket,
                            key,
                            local_mat,
                        )

                        mat = loadmat(local_mat)

                        eeg = mat["export"].T.astype(np.float32)

                        # Skip recordings that are not 19-channel
                        if eeg.shape[0] != 19:

                            log(
                                f"  [SKIP] {subject_id}: "
                                f"{eeg.shape[0]} channels"
                            )

                            continue



                    s3_upload_npy(
                        eeg,
                        bucket,
                        out_key,
                    )

                    record = {

                        "subject_id": subject_id,
                        "stem": subject_id,
                        "dataset": "FIGSHARE",
                        "true_label": label,
                        "condition": dx_name,
                        "original_sfreq": sfreq,
                        "shape": str(eeg.shape),
                        "missing_channels": "none",
                        "clean_npy": out_key,

                    }

                    if sfreq == 128:
                        rows_128.append(record)
                    else:
                        rows_256.append(record)

                    log(
                        f"  {subject_id}  "
                        f"{eeg.shape}  "
                        f"{sfreq}Hz"
                    )

                except Exception as e:

                    log(f"  [ERROR] {subject_id}: {e}")

                    traceback.print_exc()

    _save_manifest(
        rows_128,
        bucket,
        f"{S3_RAW_ROOT}/FIGSHARE-128Hz",
        "FIGSHARE_128Hz",
    )

    _save_manifest(
        rows_256,
        bucket,
        f"{S3_RAW_ROOT}/FIGSHARE-256Hz",
        "FIGSHARE_256Hz",
    )

    return rows_128 + rows_256


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 12 — ADSZ Alzheimer (GUL)
#  Source: Local .out files
#  Structure:
#      ADSZ/
#          alzheimer/
#              AD/
#              Healthy/
#
#  Each .out file contains EEG with shape (1024,19)
#  Saved as ADSZ-AD/
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_ADSZ(bucket=BUCKET):

    src_pfx = f"{S3_SRC_ROOT}/ADSZ/alzheimer"

    log("=== ADSZ (Alzheimer) ===")

    rows = []

    LABEL_MAP = {
        "AD": "1",
        "Healthy": "0",
    }

    for dx_name in ["AD", "Healthy"]:

        label = to_binary(LABEL_MAP[dx_name])

        folder = f"{src_pfx}/{dx_name}"

        files = [
            k for k in s3_list_keys(bucket, folder)
            if k.lower().endswith(".out")
        ]

        files = sorted(files)

        log(f"  {dx_name}: {len(files)} files")

        for key in files:

            stem = Path(key).stem

            prefix = "HC" if dx_name == "Healthy" else "AD"

            subject_id = f"{prefix}{stem}"

            out_key = (
                f"{S3_RAW_ROOT}/ADSZ-AD/"
                f"{subject_id}.npy"
            )

            if s3_exists(bucket, out_key):

                log(f"  [SKIP] {subject_id}")

                continue

            try:

                with tempfile.TemporaryDirectory() as tmp:

                    local_out = Path(tmp) / Path(key).name

                    s3_download(
                        bucket,
                        key,
                        local_out,
                    )

                    eeg = np.loadtxt(
                        local_out,
                        dtype=np.float32,
                    ).T

                if eeg.shape[0] != 19:

                    log(
                        f"  [SKIP] {subject_id}: "
                        f"{eeg.shape[0]} channels"
                    )

                    continue

                s3_upload_npy(
                    eeg,
                    bucket,
                    out_key,
                )

                rows.append({

                    "subject_id": subject_id,
                    "stem": subject_id,
                    "dataset": "ADSZ",
                    "true_label": label,
                    "condition": dx_name,
                    "original_sfreq": "unknown",
                    "shape": str(eeg.shape),
                    "missing_channels": "none",
                    "clean_npy": out_key,

                })

                log(
                    f"  {subject_id}  "
                    f"{eeg.shape}"
                )

            except Exception as e:

                log(f"  [ERROR] {subject_id}: {e}")

                traceback.print_exc()

    _save_manifest(
        rows,
        bucket,
        f"{S3_RAW_ROOT}/ADSZ-AD",
        "ADSZ_AD",
    )

    return rows

# ═══════════════════════════════════════════════════════════════════════════════
#  # GUL - DATASET 13 — GENEEG (WAVi EEG Dataset)
#  Source: HuggingFace GENEEG (.eeg/.art/.evt)
#  Subjects: Mild Cognitive Impairment (MCI) + Healthy Controls
#  Format: Plain text .eeg (17 channels)
#  Channel mapping: 17ch → MASTER_19 (T5/T6 zero-filled)
#  Labels:
#      Control_raw_files → 0
#      MCI_raw_files     → 1
#  Artifact/events: Ignored during ingestion (.art/.evt retained as source)
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_GENEEG(bucket=BUCKET):
    """
    GENEEG (WAVi)

    Reads raw 17-channel EEG text files and converts them into the
    project's standard 19-channel MASTER_19 format.
    """

    dataset = "GENEEG"
    out_pfx = f"{S3_RAW_ROOT}/{dataset}"

    log("=== GENEEG ===")

    CHANNELS_GENEEG = [
        "Fp1", "Fp2",
        "F3", "F4",
        "F7", "F8",
        "C3", "C4",
        "P3", "P4",
        "O1", "O2",
        "T3", "T4",
        "Fz", "Cz", "Pz",
    ]

    folders = [
        ("Control_raw_files", 0),
        ("MCI_raw_files", 1),
    ]

    rows = []

    for folder, label in folders:

        src_pfx = f"{S3_SRC_ROOT}/{dataset}/{folder}"

        log(f"\nProcessing {folder} (label={label})")

        for key in s3_list_keys(bucket, src_pfx, suffix=".eeg"):

            try:

                stem = Path(key).stem
                subject_id = stem
                out_key = f"{out_pfx}/{stem}.npy"

                if s3_exists(bucket, out_key):
                    log(f"  [SKIP] {stem}")
                    continue

                with tempfile.TemporaryDirectory() as tmp:

                    local_eeg = Path(tmp) / Path(key).name

                    s3_download(bucket, key, local_eeg)

                    eeg = np.loadtxt(local_eeg).T.astype(np.float32)

                if eeg.shape[0] != 17:
                    raise ValueError(
                        f"Expected 17 channels, got {eeg.shape[0]}"
                    )

                signals = {
                    ch: eeg[i]
                    for i, ch in enumerate(CHANNELS_GENEEG)
                }

                harmonised = []
                missing = []

                for ch in MASTER_19:

                    if ch in signals:

                        harmonised.append(signals[ch])

                    elif ch in ("T5", "T6"):

                        harmonised.append(
                            np.zeros_like(eeg[0], dtype=np.float32)
                        )

                        missing.append(ch)

                    else:

                        raise ValueError(
                            f"Unexpected missing channel: {ch}"
                        )

                eeg19 = np.vstack(harmonised).astype(np.float32)

                s3_upload_npy(eeg19, bucket, out_key)

                rows.append({
                    "subject_id": subject_id,
                    "stem": stem,
                    "dataset": dataset,
                    "true_label": label,
                    "original_sfreq": None,   # Update if sampling frequency is confirmed
                    "shape": str(eeg19.shape),
                    "missing_channels": ",".join(missing),
                    "clean_npy": out_key,
                })

                log(
                    f"  {stem}  {eeg19.shape}  "
                    f"label={label}  missing={missing or 'none'}"
                )

            except Exception as e:

                log(f"  [ERROR] {Path(key).name}: {e}")
                traceback.print_exc()

    _save_manifest(rows, bucket, out_pfx, dataset)

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  # GUL - DATASET 14 — TD-BRAIN (SMC Controls)
#  Source: BrainVision (.vhdr/.eeg/.vmrk) on S3
#  Subjects: Subjective Memory Complaints (SMC) only
#  Condition: Eyes-Closed Resting State (task-restEC)
#  Channel selection: 19 standard 10-20 channels from 33-channel recording
#  Sampling rate: 500 Hz
#  Labels: all SMC → 0
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_TDBRAIN(bucket=BUCKET):
    """
    TD-BRAIN

    Includes only SMC (Subjective Memory Complaints)
    resting-state Eyes Closed recordings.

    Output:
        final_npy_raw/TD-BRAIN/
    """

    src_pfx = f"{S3_SRC_ROOT}/TD-BRAIN-SAMPLE"
    out_pfx = f"{S3_RAW_ROOT}/TD-BRAIN"

    log("=== TD-BRAIN (SMC Controls) ===")

    TARGET = [
        "Fp1","Fp2",
        "F7","F3","Fz","F4","F8",
        "T7","C3","Cz","C4","T8",
        "P7","P3","Pz","P4","P8",
        "O1","O2",
    ]

    RENAME = {
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6",
    }

    rows = []

    # ------------------------------------------------------------------
    # Load participants.tsv
    # ------------------------------------------------------------------

    keep_subjects = set()

    try:

        pts = s3_read_tsv(
            bucket,
            f"{src_pfx}/participants.tsv"
        )

        pts.columns = pts.columns.str.strip()

        for _, row in pts.iterrows():

            indication = str(
                row["indication"]
            ).strip().upper()

            if indication == "SMC":

                keep_subjects.add(
                    str(row["participant_id"]).strip()
                )

        log(f"Keeping {len(keep_subjects)} SMC subjects")

    except Exception as e:

        log(f"[ERROR] participants.tsv : {e}")
        traceback.print_exc()
        return []

    # ------------------------------------------------------------------
    # Process each SMC subject
    # ------------------------------------------------------------------

    for subject in sorted(keep_subjects):

        for session in ["ses-1", "ses-2", "ses-3"]:

            eeg_prefix = (
                f"{src_pfx}/{subject}/{session}/eeg"
            )

            base = (
                f"{eeg_prefix}/"
                f"{subject}_{session}_task-restEC_eeg"
            )

            stem = f"{subject}_{session}_restEC"

            out_key = (
                f"{out_pfx}/{stem}.npy"
            )

            if s3_exists(bucket, out_key):

                log(f"  [SKIP] {stem}")
                continue

            try:

                with tempfile.TemporaryDirectory() as tmp:

                    for ext in [
                        ".vhdr",
                        ".eeg",
                        ".vmrk",
                    ]:

                        key = base + ext

                        local = (
                            Path(tmp) /
                            f"{subject}_{session}_task-restEC_eeg{ext}"
                        )

                        s3_download(
                            bucket,
                            key,
                            local,
                        )

                    raw = mne.io.read_raw_brainvision(
                        str(
                            Path(tmp) /
                            f"{subject}_{session}_task-restEC_eeg.vhdr"
                        ),
                        preload=True,
                        verbose=False,
                    )

                    available = raw.ch_names

                    ch_map = {}

                    for tch in TARGET:

                        for cand in [tch, tch.upper()]:

                            if cand in available:

                                ch_map[tch] = cand
                                break

                    missing = [
                        t
                        for t in TARGET
                        if t not in ch_map
                    ]

                    if missing:

                        raise ValueError(
                            f"Missing channels: {missing}"
                        )
                    
                    raw.pick(
                        [ch_map[t] for t in TARGET],
                        ordered=True,
                    )

                    rename = {
                        ch_map[t]: t
                        for t in TARGET
                        if ch_map[t] != t
                    }

                    if rename:
                        raw.rename_channels(rename)

                    # Rename to MASTER_19 names
                    raw.rename_channels({
                        t: RENAME.get(t, t)
                        for t in TARGET
                        if t in RENAME
                    })

                    eeg, _ = raw[:]

                    eeg = eeg.astype(np.float32)

                s3_upload_npy(
                    eeg,
                    bucket,
                    out_key,
                )

                rows.append({
                    "subject_id": subject,
                    "stem": stem,
                    "dataset": "TD-BRAIN",
                    "true_label": 0,
                    "condition": "SMC_restEC",
                    "session": session,
                    "original_sfreq": 500,
                    "shape": str(eeg.shape),
                    "clean_npy": out_key,
                })

                log(
                    f"  {stem}  "
                    f"{eeg.shape}  "
                    f"500Hz"
                )

            except ClientError as e:

                if e.response["Error"]["Code"] in (
                    "404",
                    "NoSuchKey",
                ):
                    continue

                log(f"  [ERROR] {stem}: {e}")
                traceback.print_exc()

            except Exception as e:

                log(f"  [ERROR] {stem}: {e}")
                traceback.print_exc()

    _save_manifest(
        rows,
        bucket,
        out_pfx,
        "TD-BRAIN",
    )

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 15 — DS005048 (40Hz Auditory Entrainment)
#  Source: EEGLAB v7.3 (.set + .fdt) on S3
#  Recording: 40Hz Auditory Entrainment
#  Sampling: 250Hz
#  Channels: 19 standard 10-20
#  Labels:
#      Normal  -> 0
#      Mild AD -> 1
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_DS005048(bucket=BUCKET):
    """
    DS005048

    EEGLAB v7.3 dataset.

    EEG is stored inside .fdt while metadata is
    stored inside .set (HDF5).

    Labels:
        Normal  -> 0
        Mild AD -> 1
    """

    dataset = "DS005048"

    src_pfx = f"{S3_SRC_ROOT}/ds005048"
    out_pfx = f"{S3_RAW_ROOT}/DS005048"

    log("=== DS005048 (40Hz Auditory Entrainment) ===")

    LABEL_MAP = {
        "Normal": 0,
        "Mild AD": 1,
    }

    RENAME = {
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6",
    }

    # ------------------------------------------------------------
    # Participant labels
    # ------------------------------------------------------------

    label_map = {}

    try:

        pts = s3_read_tsv(
            bucket,
            f"{src_pfx}/participants.tsv",
        )

        for _, row in pts.iterrows():

            sid = str(
                row["participant_id"]
            ).strip()

            dx = str(
                row["Group"]
            ).strip()

            if dx in LABEL_MAP:

                label_map[sid] = LABEL_MAP[dx]

        log(
            f"Loaded labels for {len(label_map)} subjects"
        )

    except Exception as e:

        log(
            f"[ERROR] participants.tsv : {e}"
        )

        traceback.print_exc()

        return []

    rows = []

    # ------------------------------------------------------------
    # EEG files
    # ------------------------------------------------------------

    for key in s3_list_keys(
        bucket,
        src_pfx,
        suffix=".set",
    ):
        # if "sub-01" not in key:
        #     continue

        try:

            subject_id = next(
                (
                    p
                    for p in Path(key).parts
                    if p.startswith("sub-")
                ),
                None,
            )

            label = label_map.get(subject_id)

            if label is None:

                log(
                    f"  [SKIP] {subject_id}"
                )

                continue

            stem = Path(key).stem

            out_key = (
                f"{out_pfx}/{stem}.npy"
            )

            if s3_exists(
                bucket,
                out_key,
            ):

                log(
                    f"  [SKIP] {stem}"
                )

                continue

            log(
                f"  {stem}  label={label}"
            )

            with tempfile.TemporaryDirectory() as tmp:

                local_set = (
                    Path(tmp) /
                    Path(key).name
                )

                s3_download(
                    bucket,
                    key,
                    local_set,
                )

                fdt_key = key[:-4] + ".fdt"

                local_fdt = (
                    Path(tmp) /
                    Path(fdt_key).name
                )

                s3_download(
                    bucket,
                    fdt_key,
                    local_fdt,
                )

                with h5py.File(
                    local_set,
                    "r",
                ) as f:

                    sfreq = int(
                        f["srate"][0, 0]
                    )

                    nchan = int(
                        f["nbchan"][0, 0]
                    )

                    nsamp = int(
                        f["pnts"][0, 0]
                    )

                    refs = (
                        f["chanlocs"]["labels"][:]
                    )

                    channels = []

                    for ref in refs.flatten():

                        arr = f[ref][()]

                        name = "".join(
                            chr(int(x))
                            for x in arr.flatten()
                            if int(x) != 0
                        )

                        channels.append(name)

                # --------------------------------------------------
                # Read EEG from .fdt
                # --------------------------------------------------

                eeg = np.fromfile(
                    local_fdt,
                    dtype=np.float32,
                )

                expected = nchan * nsamp

                if eeg.size != expected:

                    raise ValueError(
                        f"Expected {expected} values, "
                        f"got {eeg.size}"
                    )

                # EEGLAB stores continuous data
                # channel-major in .fdt
                eeg = eeg.reshape(
                    nchan,
                    nsamp,
                )

                if eeg.shape[0] != 19:

                    raise ValueError(
                        f"Expected 19 channels, "
                        f"got {eeg.shape[0]}"
                    )

                # --------------------------------------------------
                # Rename channels to MASTER_19
                # --------------------------------------------------

                channels = [
                    RENAME.get(ch, ch)
                    for ch in channels
                ]

                expected_channels = [
                    "Fp1","Fp2",
                    "F7","F3","Fz","F4","F8",
                    "T3","C3","Cz","C4","T4",
                    "T5","P3","Pz","P4","T6",
                    "O1","O2",
                ]

                if channels != expected_channels:

                    log(
                        f"  [WARN] Channel order differs:"
                    )
                    log(
                        f"    {channels}"
                    )

                eeg = eeg.astype(np.float32)

            s3_upload_npy(
                eeg,
                bucket,
                out_key,
            )

            rows.append({

                "subject_id": subject_id,
                "stem": stem,
                "dataset": dataset,
                "true_label": label,

                "original_sfreq": sfreq,

                "shape": str(eeg.shape),

                "clean_npy": out_key,

            })

            log(
                f"    → {eeg.shape}  "
                f"{sfreq}Hz  saved"
            )

        except Exception as e:

            log(
                f"  [ERROR] {key}: {e}"
            )

            traceback.print_exc()

    _save_manifest(
        rows,
        bucket,
        out_pfx,
        dataset,
    )

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 16 — Custom EDF Dataset
#  Source: EDF
#  Conversion: convert_raw_edf() (shared with convert_edf_batch.py)
#  Output: Standard 19-channel NPY
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_CUSTOM_EDF(bucket=BUCKET, local_test_edf=None):
    """
    Generic EDF dataset loader.

    Normal mode:
        Reads EDFs from S3.

    Test mode:
        Converts a single local EDF without requiring S3.
    """

    dataset = "CUSTOM_EDF"

    src_pfx = f"{S3_SRC_ROOT}/{dataset}"
    out_pfx = f"{S3_RAW_ROOT}/{dataset}"

    log(f"=== {dataset} ===")

    # ------------------------------------------------------------------
    # Example label loading (optional)
    #
    # Expected labels.csv format:
    #
    # subject_id,label
    # sub-001,1
    # sub-002,0
    #
    # If labels.csv is not present,
    # ingestion continues with true_label=None.
    # ------------------------------------------------------------------

    label_map = {}

    try:

        labels = s3_read_csv(
            bucket,
            f"{src_pfx}/labels.csv",
        )

        for _, row in labels.iterrows():

            label_map[
                str(row["subject_id"]).strip()
            ] = int(row["label"])

        log(f"Loaded labels for {len(label_map)} subjects")

    except Exception:

        log("No labels.csv found. Proceeding without labels.")

    rows = []

    # ------------------------------------------------------------------
    # LOCAL TEST
    # ------------------------------------------------------------------

    if local_test_edf is not None:

        log(f"[LOCAL TEST] {local_test_edf}")

        raw = convert_raw_edf(local_test_edf)

        eeg = raw.get_data().astype(np.float32)

        print("=" * 60)
        print("Converted")
        print("=" * 60)
        print("Shape :", eeg.shape)
        print("dtype :", eeg.dtype)
        print("sfreq :", raw.info["sfreq"])
        print("Channels :", raw.ch_names)
        print("=" * 60)

        return

    # ------------------------------------------------------------------
    # S3 
    # ------------------------------------------------------------------

    for key in s3_list_keys(
        bucket,
        src_pfx,
        suffix=".edf",
    ):

        try:

            subject_id = next(
                (
                    p
                    for p in Path(key).parts
                    if p.startswith("sub-")
                ),
                None,
            )

            stem = Path(key).stem

            out_key = f"{out_pfx}/{stem}.npy"

            if s3_exists(bucket, out_key):
                log(f"  [SKIP] {stem}")
                continue

            log(f"  {stem}  label={label_map.get(subject_id)}")

            with tempfile.TemporaryDirectory() as tmp:

                local_edf = Path(tmp) / Path(key).name

                s3_download(
                    bucket,
                    key,
                    local_edf,
                )

                raw = convert_raw_edf(str(local_edf))

                eeg = raw.get_data().astype(np.float32)

                sfreq = raw.info["sfreq"]

            s3_upload_npy(
                eeg,
                bucket,
                out_key,
            )

            rows.append(
                {
                    "subject_id": subject_id,
                    "stem": stem,
                    "dataset": dataset,
                    "true_label": label_map.get(subject_id),
                    "original_sfreq": sfreq,
                    "shape": str(eeg.shape),
                    "clean_npy": out_key,
                }
            )

            log(f"    → {eeg.shape} {sfreq}Hz saved")

        except Exception as e:

            log(f"[ERROR] {key}: {e}")

            traceback.print_exc()

    _save_manifest(
        rows,
        bucket,
        out_pfx,
        dataset,
    )

    return rows


# ─── MANIFEST HELPER ─────────────────────────────────────────────────────────

def _save_manifest(rows, bucket, out_pfx, name):
    if not rows:
        log(f"  No rows for {name}"); return
    df  = pd.DataFrame(rows)
    key = f"{out_pfx}/manifest_{name}.csv"
    s3_upload_csv(df, bucket, key)
    log(f"\n  Manifest ({name}) → s3://{bucket}/{key}")
    log(f"  Total: {len(df)}  "
        f"label=1: {(df.true_label==1).sum()}  "
        f"label=0: {(df.true_label==0).sum()}")

def ingest_Allengers(bucket=BUCKET):
    """
    Allengers EEG Dataset
    Input : EDF
    Output: NPY
    """

    dataset = "test_hardware"
    src_pfx = f"{S3_SRC_ROOT}/{dataset}"
    out_pfx = f"{S3_RAW_ROOT}/test_hardware"

    log("=== Allengers EEG ===")

    rows = []

    for key in s3_list_keys(bucket, src_pfx, suffix=".edf"):

        try:
            file_stem = Path(key).stem
            subject_id = file_stem

            out_key = f"{out_pfx}/{file_stem}.npy"

            if s3_exists(bucket, out_key):
                log(f"  [SKIP] {file_stem}")
                continue

            log(f"  {file_stem}")

            with tempfile.TemporaryDirectory() as tmp:

                local_file = Path(tmp) / Path(key).name
                s3_download(bucket, key, local_file)

                raw = mne.io.read_raw_edf(
                    str(local_file),
                    preload=True,
                    verbose=False
                )

                # Keep EEG channels only
                raw.pick("eeg")

                # Optional preprocessing
                # raw.filter(0.5, 45)
                # raw.notch_filter(50)
                # raw.resample(128)

                # Select standard 19 channels
                raw = select_19ch_from_raw(raw)

                eeg = raw.get_data().astype(np.float32)
                sfreq = raw.info["sfreq"]
                channels = raw.ch_names
                print(sfreq, "--sfreq", channels)

            s3_upload_npy(eeg, bucket, out_key)

            rows.append({
                "subject_id": subject_id,
                "stem": file_stem,
                "dataset": "Allengers",
                "true_label": None,
                "original_sfreq": sfreq,
                "shape": str(eeg.shape),
                "clean_npy": out_key
            })

            log(f"    → {eeg.shape}   {sfreq} Hz")

        except Exception as e:
            log(f"  [ERROR] {key}: {e}")
            traceback.print_exc()

    _save_manifest(rows, bucket, out_pfx, "Allengers")

    return rows

# ─── CLI ──────────────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "DS003800":       ("S3-only",  ingest_DS003800), #done
    "DS004504":       ("S3-only",  ingest_DS004504), #done
    "ADFSU":          ("S3-only",    ingest_ADFSU), #done # (GUL)
    "BrainLat":       ("local",    ingest_BrainLat), #DONE
    "P-ADIC-dem":     ("local",    ingest_PADIC_dementia), #done
    "P-ADIC-ctrl":    ("local",    ingest_PADIC_controls), #done
    "Isfahan":        ("S3-only",  ingest_Isfahan), #done
    "BACA":      ("S3-only",   ingest_BACA), #done
    # "BACA-ses2":      ("S3-only",  lambda b=BUCKET: ingest_BACA(b, "ses-2")),
    "PEARL":          ("S3-only",  ingest_PEARL),
    "test_hardware": ("S3-only",ingest_Allengers),
    "APAVA": ("S3-only", ingest_APAVA),   # (GUL),
    "FIGSHARE": ("S3-only", ingest_FIGSHARE),   # (GUL),
    "ADSZ": ("S3-only", ingest_ADSZ),    # (GUL)
    "GENEEG": ("S3-only", ingest_GENEEG),        # (GUL)
    "TD-BRAIN": ("S3-only", ingest_TDBRAIN),     # (GUL)
    "DS005048": ("S3-only", ingest_DS005048),     # (GUL)
    "CUSTOM_EDF": ("S3-only", ingest_CUSTOM_EDF),   # (GUL)
} 


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="NEMA Unified Dataset Ingestion — raw sources → S3 .npy"
    )
    p.add_argument("--dataset", default=None,
                   choices=list(DATASET_REGISTRY.keys()) + ["all"],
                   help="Dataset to ingest (or 'all')")
    p.add_argument("--bucket",  default=BUCKET)
    p.add_argument("--local_root", default=None,
                   help="Local data root for datasets requiring local files "
                        "(ADFSU, BrainLat, P-ADIC)")
    args = p.parse_args()

    if args.dataset is None:
        p.print_help()
        print(f"\nAvailable datasets: {list(DATASET_REGISTRY.keys())}")
        sys.exit(0)

    targets = list(DATASET_REGISTRY.keys()) if args.dataset == "all" else [args.dataset]

    for ds in targets:
        kind, fn = DATASET_REGISTRY[ds]
        log(f"\n{'='*60}\nIngesting: {ds}  ({kind})\n{'='*60}")
        try:
            if kind == "local":
                if not args.local_root:
                    log(f"  [SKIP] {ds} requires --local_root"); continue
                fn(args.local_root, bucket=args.bucket)
            else:
                fn(bucket=args.bucket)

        except Exception as e:
            log(f"  [FATAL] {ds}: {e}")
            traceback.print_exc()

    log("\nIngestion complete. Next: python data_pipeline.py --all_datasets")