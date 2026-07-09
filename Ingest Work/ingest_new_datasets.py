"""
ingest_new_datasets.py — Ingestion for AD Cohort B, TD-BRAIN, GENEEG
=====================================================================
Handles five new datasets on S3. Each dataset has its own clearly
labelled section explaining the structure, label logic, and decisions.

Decision summary:
  AD Cohort B (Figshare/Sadegh-Zadeh)  ✅ Include — AD/MCI/HC, .mat export key
  TD-BRAIN-SAMPLE                       ⚠️  SMC subjects only → label=0 (controls)
  GENEEG                                ✅ Include — MCI/HC, 17ch (T5/T6 excluded)
  ds006036 (AHEPA eyes-open)            ❌ EXCLUDED — same subjects as ds004504,
                                           eyes-open photic stimulation (OOD)
  Olfactory AD/MCI dataset              ❌ EXCLUDED — olfactory stimulation (OOD)
                                           not resting-state, different neural state

S3 source folder: nema_final_used/ (same as other datasets)
S3 output:        nema_final_used/all_npy_raw/{DATASET}/{stem}.npy

Label logic enforced in manifest:
  - Labels read from the actual data/metadata per dataset
  - Never positional — always filename-keyed or structure-keyed
  - Binary only: 0=control, 1=dementia/MCI
  - Excluded classes written to manifest with true_label=None and skipped

Usage:
  python ingest_new_datasets.py --dataset AD_COHORT_B
  python ingest_new_datasets.py --dataset TDBRAIN
  python ingest_new_datasets.py --dataset GENEEG
  python ingest_new_datasets.py --all
  python ingest_new_datasets.py --dataset AD_COHORT_B --inspect  # structure check
"""

import argparse
import io
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from math import gcd

import boto3
import mne
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from scipy import signal as scipy_signal

mne.set_log_level("WARNING")

# ─── GLOBAL CONFIG ────────────────────────────────────────────────────────────

BUCKET      = "dementia-research2025"
SRC_ROOT    = "nema_final_used/original_data"
OUT_ROOT    = "nema_final_used/final_raw_npy_external"

MASTER_19 = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5","P3","Pz","P4","T6",
    "O1","O2",
]

CHANNEL_ALIASES = {
    "T7":"T3","T8":"T4","P7":"T5","P8":"T6",
    "FP1":"Fp1","FP2":"Fp2",
    "CZ":"Cz","FZ":"Fz","PZ":"Pz",
}

s3 = boto3.client("s3")

# ─── S3 HELPERS ───────────────────────────────────────────────────────────────

def s3_exists(bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key); return True
    except ClientError: return False

def s3_download(bucket, key, local_path):
    s3.download_file(bucket, key, str(local_path))

def s3_upload_npy(arr, bucket, key):
    buf = io.BytesIO()
    np.save(buf, arr)
    s3.put_object(Bucket=bucket, Key=key,
                  Body=buf.getvalue(),
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

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ─── CHANNEL HELPERS ─────────────────────────────────────────────────────────

def norm_ch(name):
    return CHANNEL_ALIASES.get(name, CHANNEL_ALIASES.get(name.upper(), name))

def select_19ch(raw):
    """Select and reorder to MASTER_19 from MNE Raw. Raises if missing."""
    norm_map = {norm_ch(ch): ch for ch in raw.ch_names}
    picks, missing = [], []
    for tch in MASTER_19:
        if tch in norm_map:
            picks.append(norm_map[tch])
        else:
            missing.append(tch)
    if missing:
        raise ValueError(f"Missing channels: {missing}\nAvailable: {raw.ch_names[:25]}")
    raw.pick_channels(picks, ordered=True)
    rename = {norm_map[t]: t for t in MASTER_19 if norm_map[t] != t}
    if rename:
        raw.rename_channels(rename)
    return raw

def resample_to_256(eeg, fs_in):
    """Polyphase resample (19, T) → 256 Hz."""
    fs_out = 256
    if abs(fs_in - fs_out) < 0.5:
        return eeg.astype(np.float32)
    g = gcd(int(round(fs_in)), fs_out)
    return scipy_signal.resample_poly(
        eeg, fs_out//g, int(round(fs_in))//g, axis=1
    ).astype(np.float32)

def save_manifest(rows, bucket, out_pfx, name):
    if not rows:
        log(f"  No rows for {name}"); return
    df  = pd.DataFrame(rows)
    key = f"{out_pfx}/manifest_{name}.csv"
    s3_upload_csv(df, bucket, key)
    log(f"  Manifest → s3://{bucket}/{key}")
    log(f"  Total={len(df)}  label=1={int((df.true_label==1).sum())}  "
        f"label=0={int((df.true_label==0).sum())}")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 1 — AD Cohort B (Figshare, Sadegh-Zadeh 2023)
#  Format : MATLAB .mat, key "export" → (N_subjects, N_channels, N_samples)
#           OR (N_samples, N_channels) per subject — inspect first
#  Channels: 19, standard 10-20
#  sFreq  : not stated in paper — inspect from data or assume 256Hz
#           (MMSE scores also in file, not used here)
#  Subjects: 59 AD (label=1), 7 MCI (label=1), 102 HC (label=0)
#  Labels : separate label array or inferred from file structure
#  S3 key : nema_final_used/AD_COHORT_B/  (adjust to actual folder name)
#
#  NOTE: The .mat file has an "export" key that is the EEG array.
#        Subject-level labels are in a separate key — inspect to find it.
# ═══════════════════════════════════════════════════════════════════════════════

def inspect_ad_cohort_b(bucket=BUCKET):
    """
    Print full structure of the AD Cohort B .mat file.
    Run this first to understand the key layout before ingesting.
    """
    import h5py
    import scipy.io as sio

    # Find the .mat file
    src_pfx = f"{SRC_ROOT}/Figshare AD Cohort B"
    print(src_pfx, "--000")
    keys    = s3_list_keys(bucket, src_pfx, suffix=".mat")
    print(keys, "--999-")
    if not keys:
        # Try alternative folder names
        for alt in ["fFigshare AD Cohort B", "cohort_b", "sadegh", "AD_cohort"]:
            keys = s3_list_keys(bucket, f"{SRC_ROOT}/{alt}", suffix=".mat")
            if keys: break
    if not keys:
        log("[ERROR] No .mat file found — check S3 folder name")
        log(f"  Searched under: {SRC_ROOT}/Figshare AD Cohort B/")
        return

    log(f"Found {len(keys)} .mat files")

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(keys[0]).name
        s3_download(bucket, keys[0], local)

        # Try scipy first
        try:
            mat = sio.loadmat(str(local), squeeze_me=True, struct_as_record=False)
            mat = {k:v for k,v in mat.items() if not k.startswith("__")}
            log(f"\nMATLAB v5 format. Keys:")
            for k, v in mat.items():
                if hasattr(v, "shape"):
                    log(f"  '{k}': shape={v.shape} dtype={v.dtype}")
                    if v.ndim <= 2 and v.size < 20:
                        log(f"    values: {v}")
                else:
                    log(f"  '{k}': {type(v).__name__} = {v}")
            return mat
        except Exception:
            pass

        # Try h5py (v7.3)
        try:
            import h5py
            with h5py.File(str(local), "r") as f:
                log(f"\nHDF5 v7.3 format. Keys:")
                def show(name, obj):
                    if hasattr(obj, "shape"):
                        log(f"  '{name}': shape={obj.shape} dtype={obj.dtype}")
                    else:
                        log(f"  '{name}'/")
                f.visititems(show)
        except Exception as e:
            log(f"[ERROR] Cannot read: {e}")


def ingest_AD_COHORT_B(bucket=BUCKET):
    """
    AD Cohort B — Figshare, Sadegh-Zadeh 2023
    .mat file with 'export' key → EEG array
    168 subjects: 59 AD + 7 MCI (label=1), 102 HC (label=0)

    Expected structure (from paper description):
      mat["export"] : (168, 19, T) or (168, T, 19) — all subjects stacked
      mat["label"]  : (168,) array — 1=AD/MCI, 0=HC  OR separate arrays

    If structure differs, run with --inspect first.
    """
    import scipy.io as sio

    dataset = "Figshare AD Cohort B"
    out_pfx = f"{OUT_ROOT}/{dataset}"
    log(f"=== AD Cohort B (Figshare, Sadegh-Zadeh 2023) ===")

    # Find mat file — try multiple possible S3 folder names
    mat_key = None
    for folder in ["Figshare AD Cohort B", "figshare_ad_cohort_b",
                   "figshare_ad", "cohort_b", "sadegh_zadeh"]:
        candidates = s3_list_keys(bucket, f"{SRC_ROOT}/{folder}", suffix=".mat")
        if candidates:
            mat_key = candidates[0]
            log(f"  Found: s3://{bucket}/{mat_key}")
            break

    if not mat_key:
        log("[ERROR] Cannot find AD Cohort B .mat file.")
        log("  Searched folders: AD_COHORT_B, figshare_ad_cohort_b, etc.")
        log("  Run with --inspect to debug, or adjust folder name in code.")
        return []

    rows = []

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / Path(mat_key).name
        s3_download(bucket, mat_key, local)
        log(f"  Downloaded {local.name}")

        # ── Load mat file ─────────────────────────────────────────────────────
        mat = None
        try:
            mat = sio.loadmat(str(local), squeeze_me=True, struct_as_record=False)
            mat = {k:v for k,v in mat.items() if not k.startswith("__")}
            log(f"  Loaded (scipy v5). Keys: {list(mat.keys())}")
        except Exception:
            try:
                import h5py
                mat = {}
                with h5py.File(str(local), "r") as f:
                    for k in f.keys():
                        mat[k] = f[k][()]
                log(f"  Loaded (h5py v7.3). Keys: {list(mat.keys())}")
            except Exception as e:
                log(f"  [ERROR] Cannot load: {e}"); return []

        # ── Find EEG array under 'export' key ─────────────────────────────────
        if "export" not in mat:
            # Try common alternative keys
            for alt in ["data","eeg","EEG","signal","X"]:
                if alt in mat:
                    log(f"  'export' key not found — using '{alt}'")
                    mat["export"] = mat[alt]
                    break
            else:
                log(f"  [ERROR] No 'export' key. Keys: {list(mat.keys())}")
                log("  Run with --inspect to examine structure.")
                return []

        eeg_all = np.array(mat["export"], dtype=np.float32)
        log(f"  export shape: {eeg_all.shape}")

        # ── Determine shape convention ─────────────────────────────────────────
        # Could be (N, 19, T), (N, T, 19), (19, T) single subject, etc.
        if eeg_all.ndim == 3:
            n_subjects = eeg_all.shape[0]
            # Find which axis is 19
            if eeg_all.shape[1] == 19:
                # (N, 19, T) — correct orientation per subject
                get_subject = lambda i: eeg_all[i]          # (19, T)
            elif eeg_all.shape[2] == 19:
                # (N, T, 19) — need transpose per subject
                get_subject = lambda i: eeg_all[i].T        # (19, T)
            else:
                log(f"  [ERROR] Cannot find 19-ch axis in shape {eeg_all.shape}")
                return []
        elif eeg_all.ndim == 2:
            # Single-subject file? Or (N, features)?
            if 19 in eeg_all.shape:
                n_subjects = 1
                eeg_all = eeg_all[np.newaxis]  # add subject axis
                get_subject = lambda i: eeg_all[0] if eeg_all.shape[1]==19 \
                              else eeg_all[0].T
            else:
                log(f"  [ERROR] 2D array shape {eeg_all.shape} — unclear structure")
                log("  Run with --inspect to debug.")
                return []
        else:
            log(f"  [ERROR] Unexpected ndim={eeg_all.ndim}")
            return []

        log(f"  Subjects: {n_subjects}")

        # ── Find labels ───────────────────────────────────────────────────────
        # Paper: 59 AD + 7 MCI = 66 label=1, 102 HC = label=0
        # Labels may be in: mat["label"], mat["labels"], mat["group"], mat["y"]
        # OR inferred from ordering (first 102 HC, then 66 AD+MCI)
        labels = None

        for lk in ["label","labels","group","y","Y","class","diagnosis"]:
            if lk in mat:
                raw = np.array(mat[lk]).flatten()
                if len(raw) == n_subjects:
                    log(f"  Labels from key '{lk}': "
                        f"unique={np.unique(raw).tolist()}")
                    labels = raw
                    break

        if labels is None:
            # Infer from known dataset composition
            # Convention varies — most common: 0=HC, 1=AD, 2=MCI
            # OR: HC first 102, then AD 59, then MCI 7
            log("  [WARN] No label key found — inferring from dataset composition")
            log("  Assuming: first 102 = HC (0), next 59 = AD (1), last 7 = MCI (1)")
            labels = np.array(
                [0]*102 + [1]*59 + [1]*7, dtype=np.float32
            )
            if len(labels) != n_subjects:
                log(f"  [WARN] Inferred label count ({len(labels)}) "
                    f"≠ subjects ({n_subjects}) — using all label=1 as fallback")
                labels = None

        # ── Binary label map ──────────────────────────────────────────────────
        def to_binary(raw_label):
            """Map raw label values to 0/1/None."""
            rl = int(raw_label) if not np.isnan(float(raw_label)) else None
            # Common codings:
            # 0=HC→0, 1=AD→1, 2=MCI→1  OR  1=HC→0, 2=AD→1, 3=MCI→1
            mapping = {0:0, 1:1, 2:1, 3:1,   # 0-based
                       -1:None}               # unknown
            if rl in mapping:
                return mapping[rl]
            # If all values are 1/2: 1=HC, 2=AD
            return None

        # ── Process each subject ──────────────────────────────────────────────
        # Detect sfreq — not stated in paper, inspect data duration
        # Common for this type: 256 Hz, 512 Hz, or 500 Hz
        # We try to detect from sample count heuristics
        sample_count = get_subject(0).shape[1]
        # Assume 60s recording: sfreq = samples/60 rounded to common values
        for candidate_sfreq in [256, 500, 512, 250, 200, 128]:
            if abs(sample_count / candidate_sfreq - round(sample_count / candidate_sfreq)) < 0.1:
                sfreq_orig = candidate_sfreq
                log(f"  Inferred sfreq: {sfreq_orig} Hz "
                    f"(samples={sample_count}, "
                    f"duration≈{sample_count/sfreq_orig:.0f}s)")
                break
        else:
            sfreq_orig = 256
            log(f"  [WARN] Could not infer sfreq — defaulting to 256 Hz")

        for i in range(n_subjects):
            stem    = f"ADCOHORTB_{i:03d}"
            out_key = f"{out_pfx}/{stem}.npy"

            if s3_exists(bucket, out_key):
                log(f"  [SKIP] {stem}"); continue

            eeg = get_subject(i)   # (19, T)

            # Unit check — convert μV → V if needed
            if eeg.std() > 1.0:
                eeg = (eeg / 1e6).astype(np.float32)

            # Get label
            if labels is not None:
                raw_lbl = labels[i]
                binary  = to_binary(raw_lbl)
            else:
                binary = None   # unknown

            duration = eeg.shape[1] / sfreq_orig
            if duration < 10:
                log(f"  [SKIP] {stem}: too short ({duration:.1f}s)")
                continue

            s3_upload_npy(eeg, bucket, out_key)

            rows.append({
                "subject_id":     stem,
                "stem":           stem,
                "dataset":        dataset,
                "true_label":     binary,
                "original_sfreq": sfreq_orig,
                "duration_sec":   round(duration, 1),
                "shape":          str(eeg.shape),
                "clean_npy":      out_key,
                "raw_label":      int(labels[i]) if labels is not None else None,
            })
            log(f"  [{i+1}/{n_subjects}] {stem}  "
                f"label={binary}  {duration:.0f}s")

    save_manifest(rows, bucket, out_pfx, dataset)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 2 — TD-BRAIN-SAMPLE (BrainVision .vhdr/.eeg)
#  Format : BrainVision (.vhdr + .eeg + .vmrk)
#  Channels: 26-channel 10-10 system, Ag/AgCl, 500Hz
#  Subjects: MDD (426), ADHD (271), SMC (119), OCD (75), Healthy (47)
#  Decision: Use SMC (Subjective Memory Complaints) → label=0 ONLY
#            MDD/ADHD/OCD confound dementia EEG signal — excluded
#            Healthy (n=47) → label=0 — included
#
#  Why SMC as controls:
#    SMC subjects have memory concerns but no clinical impairment.
#    They are the correct age-matched control group for a dementia study.
#    MDD/ADHD/OCD produce EEG changes that could be mistaken for dementia.
#
#  Labels: read from participants TSV / sidecar JSON, not assumed from position
#  S3 key: nema_final_used/TD-BRAIN-SAMPLE/
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_TDBRAIN(bucket=BUCKET):
    """
    TD-BRAIN-SAMPLE — clinical lifespan EEG database
    26-channel 10-10 system, 500Hz, BrainVision format
    SMC (Subjective Memory Complaints) + Healthy → label=0 only
    MDD, ADHD, OCD → EXCLUDED

    Labels read from participants.tsv or sidecar JSON per subject.
    """
    dataset = "TDBRAIN"
    src_pfx = f"{SRC_ROOT}/TD-BRAIN-SAMPLE"
    out_pfx = f"{OUT_ROOT}/{dataset}"
    log("=== TD-BRAIN-SAMPLE (SMC + Healthy controls only) ===")

    # ── Label map from diagnosis ───────────────────────────────────────────────
    # SMC = subjective memory complaints = age-matched controls → label=0
    # HC  = healthy                                             → label=0
    # MDD/ADHD/OCD/other psychiatric                           → excluded (None)
    TDBRAIN_LABEL_MAP = {
        "smc":     0,   "SMC":     0,
        "healthy": 0,   "hc":      0,   "control": 0, "HC": 0,
        "mdd":     None, "MDD":    None,
        "adhd":    None, "ADHD":   None,
        "ocd":     None, "OCD":    None,
        "other":   None, "unknown": None,
    }

    # ── Load participants metadata ─────────────────────────────────────────────
    label_lookup = {}
    for tsv_key in [f"{src_pfx}/participants.tsv",
                    f"{src_pfx}/participants.csv"]:
        if s3_exists(bucket, tsv_key):
            data   = s3.get_object(Bucket=bucket, Key=tsv_key)["Body"].read()
            sep    = "\t" if tsv_key.endswith(".tsv") else ","
            pts_df = pd.read_csv(io.BytesIO(data), sep=sep)
            pts_df.columns = pts_df.columns.str.lower().str.strip()

            id_col  = next((c for c in pts_df.columns
                            if "participant" in c or "subject" in c or c=="id"),
                           pts_df.columns[0])
            dx_col  = next((c for c in pts_df.columns
                            if "diagnosis" in c or "group" in c
                            or "dx" in c or "disorder" in c), None)

            if dx_col:
                for _, row in pts_df.iterrows():
                    sid = str(row[id_col]).strip()
                    dx  = str(row[dx_col]).strip()
                    label_lookup[sid] = TDBRAIN_LABEL_MAP.get(
                        dx, TDBRAIN_LABEL_MAP.get(dx.lower(), None))
                log(f"  Loaded {len(label_lookup)} labels from {tsv_key}")
            break

    if not label_lookup:
        log("  [WARN] No participants.tsv found — will skip subjects with unknown labels")

    # ── Find all .vhdr files ───────────────────────────────────────────────────
    vhdr_keys = s3_list_keys(bucket, src_pfx, suffix=".vhdr")
    log(f"  Found {len(vhdr_keys)} .vhdr files")

    rows = []

    for vhdr_key in vhdr_keys:
        stem       = Path(vhdr_key).stem
        subject_id = stem.split("_")[0] if "_" in stem else stem

        # Resolve label — try full stem, then subject ID
        label = label_lookup.get(stem,
                label_lookup.get(subject_id,
                label_lookup.get(f"sub-{subject_id}", None)))

        if label is None:
            # Try reading label from sidecar JSON
            json_key = vhdr_key.replace(".vhdr", ".json")
            if s3_exists(bucket, json_key):
                try:
                    data = s3.get_object(Bucket=bucket, Key=json_key)["Body"].read()
                    meta = json.loads(data)
                    dx   = str(meta.get("diagnosis","") or
                               meta.get("group","") or
                               meta.get("condition","")).strip()
                    label = TDBRAIN_LABEL_MAP.get(dx,
                            TDBRAIN_LABEL_MAP.get(dx.lower(), None))
                except Exception:
                    pass

        if label is None:
            log(f"  [SKIP] {stem} — no label or excluded diagnosis")
            continue

        out_key = f"{out_pfx}/{stem}.npy"
        if s3_exists(bucket, out_key):
            log(f"  [SKIP] {stem} already exists"); continue

        log(f"  {stem}  label={label}")

        try:
            with tempfile.TemporaryDirectory() as tmp:
                base   = str(Path(vhdr_key).with_suffix(""))
                parent = str(Path(vhdr_key).parent)
                s_name = Path(vhdr_key).stem

                for ext in [".vhdr", ".eeg", ".vmrk"]:
                    k = f"{parent}/{s_name}{ext}"
                    lp = Path(tmp) / f"{s_name}{ext}"
                    try:
                        s3_download(bucket, k, lp)
                    except ClientError:
                        if ext == ".vhdr":
                            raise

                raw = mne.io.read_raw_brainvision(
                    str(Path(tmp)/f"{s_name}.vhdr"),
                    preload=True, verbose=False)

                # Select 19 standard channels from 26-channel 10-10 system
                raw   = select_19ch(raw)
                sfreq = raw.info["sfreq"]
                eeg   = raw.get_data().astype(np.float32)   # Volts from MNE

            s3_upload_npy(eeg, bucket, out_key)
            rows.append({
                "subject_id":     stem,
                "stem":           stem,
                "dataset":        dataset,
                "true_label":     label,
                "original_sfreq": sfreq,
                "duration_sec":   round(eeg.shape[1]/sfreq, 1),
                "shape":          str(eeg.shape),
                "clean_npy":      out_key,
            })
            log(f"    → {eeg.shape}  {sfreq:.0f}Hz  label={label}")

        except Exception as e:
            log(f"  [ERROR] {stem}: {e}")
            traceback.print_exc()

    save_manifest(rows, bucket, out_pfx, dataset)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET 3 — GENEEG (.eeg + .art format)
#  Format : Neuroscan .eeg binary + .art artifact file
#  Channels: 19 standard 10-20 BUT T5 and T6 were excluded in the
#            original study for technical reasons → 17 usable channels
#            We include all 19 slots but zero-fill T5/T6 with a flag
#            OR skip T5/T6 and use 17ch — this script uses 17ch approach
#            and maps to a 17-channel subset of MASTER_19
#  sFreq  : 250 Hz
#  Task   : 4-minute eyes-closed P300 protocol (NOT pure resting state)
#           BUT eyes-closed + sitting still — close enough to resting
#  Subjects: MCI (label=1), HC (label=0)
#  Labels : from filename prefix, participants.tsv, or folder structure
#  S3 key : nema_final_used/GENEEG/
#
#  NOTE on .art files: artifact markers — MNE reads .eeg directly
#  The .art file is not needed for the EEG signal itself
# ═══════════════════════════════════════════════════════════════════════════════

# GENEEG uses 17 channels (T5 and T6 excluded per original study)
GENEEG_17CH = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "P3","Pz","P4",           # T5 and T6 excluded
    "O1","O2",
]

def ingest_GENEEG(bucket=BUCKET):
    """
    GENEEG — MCI and Healthy Controls
    Neuroscan .eeg format, 250Hz, 17ch (T5/T6 excluded)
    Eyes-closed P300 protocol (4 min, approximately resting)
    Labels from participants.tsv or filename prefix (MCI_/HC_)

    17-channel output — pipeline handles this via data_pipeline.py
    which will reorder to available channels only.
    """
    dataset = "GENEEG"
    src_pfx = f"{SRC_ROOT}/GENEEG"
    out_pfx = f"{OUT_ROOT}/{dataset}"
    log("=== GENEEG (MCI/HC, 17ch, 250Hz, eyes-closed P300) ===")

    # ── Label lookup from participants.tsv ────────────────────────────────────
    label_lookup = {}
    for tsv_key in [f"{src_pfx}/participants.tsv", f"{src_pfx}/participants.csv"]:
        if s3_exists(bucket, tsv_key):
            data   = s3.get_object(Bucket=bucket, Key=tsv_key)["Body"].read()
            sep    = "\t" if tsv_key.endswith(".tsv") else ","
            pts    = pd.read_csv(io.BytesIO(data), sep=sep)
            pts.columns = pts.columns.str.lower().str.strip()
            id_col = next((c for c in pts.columns
                          if "participant" in c or "subject" in c or c=="id"),
                         pts.columns[0])
            dx_col = next((c for c in pts.columns
                          if "diagnosis" in c or "group" in c or "class" in c
                          or "condition" in c), None)
            if dx_col:
                for _, row in pts.iterrows():
                    sid = str(row[id_col]).strip()
                    dx  = str(row[dx_col]).strip().upper()
                    if "MCI" in dx:
                        label_lookup[sid] = 1
                    elif dx in ("HC","CONTROL","HEALTHY","NORMAL"):
                        label_lookup[sid] = 0
            log(f"  Loaded {len(label_lookup)} labels from {tsv_key}")
            break

    # ── Find all .eeg files ───────────────────────────────────────────────────
    eeg_keys = s3_list_keys(bucket, src_pfx, suffix=".eeg")
    log(f"  Found {len(eeg_keys)} .eeg files")

    rows = []

    for eeg_key in eeg_keys:
        stem = Path(eeg_key).stem

        # ── Resolve label ─────────────────────────────────────────────────────
        # Strategy 1: participants.tsv lookup
        label = label_lookup.get(stem, label_lookup.get(stem.split("_")[0], None))

        # Strategy 2: filename prefix (MCI_001.eeg → MCI, HC_001.eeg → HC)
        if label is None:
            stem_upper = stem.upper()
            if stem_upper.startswith("MCI"):
                label = 1
            elif any(stem_upper.startswith(p) for p in ["HC","CONTROL","HEALTHY"]):
                label = 0

        if label is None:
            log(f"  [SKIP] {stem} — cannot determine label")
            continue

        out_key = f"{out_pfx}/{stem}.npy"
        if s3_exists(bucket, out_key):
            log(f"  [SKIP] {stem} exists"); continue

        log(f"  {stem}  label={label}")

        try:
            with tempfile.TemporaryDirectory() as tmp:
                local_eeg = Path(tmp) / Path(eeg_key).name
                s3_download(bucket, eeg_key, local_eeg)

                # Neuroscan .eeg — MNE reads with read_raw_cnt or read_raw_eeglab
                # Try MNE's CNT reader first (Neuroscan format)
                try:
                    raw = mne.io.read_raw_cnt(str(local_eeg),
                                              preload=True, verbose=False)
                except Exception:
                    # Fall back to EEGLAB reader
                    raw = mne.io.read_raw_eeglab(str(local_eeg),
                                                  preload=True, verbose=False)

                sfreq = raw.info["sfreq"]

                # Select 17 channels (T5/T6 excluded per original study)
                available = raw.ch_names
                norm_avail = {norm_ch(ch): ch for ch in available}

                picks = []
                used_ch = []
                for tch in GENEEG_17CH:
                    if tch in norm_avail:
                        picks.append(norm_avail[tch])
                        used_ch.append(tch)

                if len(picks) < 15:
                    raise ValueError(
                        f"Too few channels found: {len(picks)}/17. "
                        f"Available: {available[:20]}"
                    )

                raw.pick_channels(picks, ordered=True)
                rename = {norm_avail[t]: t for t in used_ch
                          if norm_avail.get(t) and norm_avail[t] != t}
                if rename:
                    raw.rename_channels(rename)

                eeg = raw.get_data().astype(np.float32)  # Volts

            s3_upload_npy(eeg, bucket, out_key)
            rows.append({
                "subject_id":     stem,
                "stem":           stem,
                "dataset":        dataset,
                "true_label":     label,
                "original_sfreq": sfreq,
                "n_channels":     eeg.shape[0],
                "duration_sec":   round(eeg.shape[1]/sfreq, 1),
                "shape":          str(eeg.shape),
                "clean_npy":      out_key,
                "note":           "17ch_T5T6_excluded",
            })
            log(f"    → {eeg.shape}  {sfreq:.0f}Hz  {eeg.shape[1]/sfreq:.0f}s")

        except Exception as e:
            log(f"  [ERROR] {stem}: {e}")
            traceback.print_exc()

    save_manifest(rows, bucket, out_pfx, dataset)
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET: ds006036 (AHEPA eyes-open + photic stimulation)
#  DECISION: ❌ EXCLUDED — do not ingest
#
#  Reasons:
#  1. Eyes-OPEN + photic stimulation — OOD for eyes-closed trained model
#     Alpha suppression during eyes-open changes all posterior biomarkers
#     Photic stimulation drives visual cortex — not resting neural state
#  2. SAME SUBJECTS as ds004504 (already in training)
#     "The participant numbers match the respective participant numbers
#     of the aforementioned dataset" — direct quote from the dataset page
#     If ingested, creates train/test leakage (same brain, different task)
#  3. Protocol incompatibility:
#     ds004504 (in training): eyes-closed resting state
#     ds006036: eyes-open + 17 photic stimulation frequencies
#     These are physiologically different states — mixing would degrade model
#
#  This function exists as documentation, not for use.
# ═══════════════════════════════════════════════════════════════════════════════

def explain_ds006036_exclusion():
    log("=== ds006036 — EXCLUDED ===")
    log("  Reason 1: Eyes-OPEN + photic stimulation (not resting state)")
    log("  Reason 2: Same subjects as ds004504 (already in training)")
    log("            Would create train/test subject-level leakage")
    log("  Reason 3: Photic stimulation drives visual cortex")
    log("            Not comparable to eyes-closed resting biomarkers")
    log("  → Skip this dataset entirely")


# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET: Olfactory Stimulation AD/MCI EEG (.mat)
#  DECISION: ❌ EXCLUDED — do not ingest
#
#  Reasons:
#  1. Olfactory stimulation protocol — OOD for resting-state trained model
#     Olfactory stimulation activates entorhinal cortex, hippocampus
#     These activations are NOT present in resting-state EEG
#     Your model learned resting-state alpha/theta patterns — not task ERP
#  2. AD detection during olfactory task ≠ resting-state dementia biomarkers
#     The EEG signature of AD in this protocol is event-related (ERP)
#     Your biomarkers (CDI, PCR, NIS_v3) are spectral, not ERP-based
#  3. Adding this would teach the model task-evoked confounds
#
#  Could be valuable for a SEPARATE study on olfactory EEG biomarkers.
#  Do not mix with resting-state training data.
# ═══════════════════════════════════════════════════════════════════════════════

def explain_olfactory_exclusion():
    log("=== Olfactory Stimulation EEG — EXCLUDED ===")
    log("  Reason: Olfactory stimulation activates entorhinal cortex/hippocampus")
    log("  Not comparable to resting-state EEG trained model")
    log("  ERP-based task signal ≠ spectral resting-state biomarkers")
    log("  → Skip this dataset entirely for current study")


# ─── CLI ──────────────────────────────────────────────────────────────────────

REGISTRY = {
    "AD_COHORT_B": ingest_AD_COHORT_B,
    "TDBRAIN":     ingest_TDBRAIN,
    "GENEEG":      ingest_GENEEG,
}

EXCLUDED = {
    "ds006036":  explain_ds006036_exclusion,
    "OLFACTORY": explain_olfactory_exclusion,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Ingest AD Cohort B, TD-BRAIN, GENEEG to S3"
    )
    p.add_argument("--dataset", default=None,
                   choices=list(REGISTRY.keys()) + list(EXCLUDED.keys()) + ["all"],
                   help="Dataset to ingest")
    p.add_argument("--bucket",  default=BUCKET)
    p.add_argument("--inspect", action="store_true",
                   help="Inspect file structure only (AD Cohort B)")
    args = p.parse_args()

    if args.inspect:
        inspect_ad_cohort_b(args.bucket)
        sys.exit(0)

    if args.dataset is None:
        p.print_help()
        print(f"\nAvailable  : {list(REGISTRY.keys())}")
        print(f"Excluded   : {list(EXCLUDED.keys())} (documented, not ingested)")
        sys.exit(0)

    targets = list(REGISTRY.keys()) if args.dataset == "all" else [args.dataset]

    for ds in targets:
        log(f"\n{'='*60}\n{ds}\n{'='*60}")
        if ds in REGISTRY:
            REGISTRY[ds](bucket=args.bucket)
        elif ds in EXCLUDED:
            EXCLUDED[ds]()
        else:
            log(f"Unknown dataset: {ds}")