"""
data_pipeline.py — NEMA EEG Data Standardisation Pipeline
==========================================================
Single entry point for ALL EEG data — training, test, and external cohort.
Identical processing for every file regardless of source.

What it does per file:
  1. Load from S3 (auto-detects .set/.edf/.bdf/.vhdr/.mat/.npy/.fif)
  2. Select + reorder to 19 standard 10-20 channels
  3. Detect and convert units → Volts (auto-detects μV)
  4. Resample → 128 Hz (polyphase, anti-aliased)
  5. Preprocess: detrend → bandpass 0.5–45 Hz → CAR
  6. QC: score 0–100, tag TRAIN / REVIEW / REJECT
  7. Validate output contract
  8. Save (19, T) float32 Volts .npy + sidecar .json to S3

NOTE: Per-window z-score normalisation is NOT applied here.
      It is applied in build_dataset_cache.py at training time,
      which is correct — the .npy stores the real signal.

Output contract (enforced at write time):
  .npy  : (19, T) float32, Volts, 128 Hz, detrend+BP+CAR
  .json : full provenance sidecar

S3 layout:
  Input : s3://{bucket}/{raw_prefix}/{dataset}/{files}
  Output: s3://{bucket}/{clean_prefix}/{dataset}/{stem}.npy
          s3://{bucket}/{clean_prefix}/{dataset}/{stem}.json

Modes:
  --mode train    → clean_prefix = nema_final_used/clean_train/
  --mode external → clean_prefix = nema_final_used/clean_external/

Usage:
  # Training data
  python data_pipeline.py \
      --bucket dementia-research2025 \
      --dataset P-ADIC \
      --raw_prefix nema_final_used/raw/ \
      --labels_csv nema_final_used/raw/P-ADIC/labels.csv \
      --mode train

  # External cohort (identical processing, different output prefix)
  python data_pipeline.py \
      --bucket dementia-research2025 \
      --dataset ds005385_session2 \
      --raw_prefix nema_final_used/final_raw_npy_external/ \
      --labels_csv nema_final_used/final_raw_npy_external/ds005385_session2/labels.csv \
      --mode external

python data_pipeline.py \
    --bucket dementia-research2025 \
    --mode train \
    --dataset CAUEEG \
    --raw_prefix nema_final_used/final_npy_raw/CAUEEG/

  # Process all datasets at
  #  once
  python data_pipeline.py \
      --bucket dementia-research2025 \
      --raw_prefix nema_final_used/raw/ \
      --mode train --all_datasets

Requirements: pip install mne boto3 scipy numpy pandas
"""

import argparse
import io
import json
import logging
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from math import gcd
from pathlib import Path

import boto3
import mne
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError
from scipy.signal import butter, detrend, sosfiltfilt, welch
from scipy import signal as scipy_signal
from scipy.stats import entropy as scipy_entropy

mne.set_log_level("WARNING")

# ─── OUTPUT CONTRACT ──────────────────────────────────────────────────────────

CONTRACT = {
    "n_channels":    19,
    "sfreq":         256,
    "dtype":         "float32",
    "units":         "Volts",
    "preprocessing": "detrend+bandpass_0.5-45Hz+CAR",
    "normalisation": "none_in_npy_applied_per_window_at_cache_build",
    "channel_order": [
        "Fp1","Fp2","F7","F3","Fz","F4","F8",
        "T3","C3","Cz","C4","T4",
        "T5","P3","Pz","P4","T6",
        "O1","O2",
    ],
}

# Channel name aliases across different EEG systems
CHANNEL_ALIASES = {
    # Extended 10-20
    "T7":"T3",  "T8":"T4",
    "P7":"T5",  "P8":"T6",
    # Capitalisation variants
    "FP1":"Fp1","FP2":"Fp2",
    "CZ":"Cz",  "FZ":"Fz",  "PZ":"Pz",
    # Prefixed variants (some systems add 'EEG ')
    "EEG FP1":"Fp1","EEG FP2":"Fp2",
    "EEG F7":"F7",  "EEG F3":"F3",  "EEG FZ":"Fz",
    "EEG F4":"F4",  "EEG F8":"F8",
    "EEG T3":"T3",  "EEG C3":"C3",  "EEG CZ":"Cz",
    "EEG C4":"C4",  "EEG T4":"T4",
    "EEG T5":"T5",  "EEG P3":"P3",  "EEG PZ":"Pz",
    "EEG P4":"P4",  "EEG T6":"T6",
    "EEG O1":"O1",  "EEG O2":"O2",
}

# QC thresholds — calibrated for Volts
# QC_THRESHOLDS = {
#     "flatline_std":      1e-9,    # V — channel is flat below this
#     "dropout_ptp":       1e-8,    # V — channel is dead below this
#     "artifact_amp":      500e-6,  # V — window has artefact above this
#     "min_corr":          0.05,    # mean abs cross-channel correlation
#     "max_emg_ratio":     0.40,    # fraction of power above 30 Hz
#     "max_line_ratio":    0.30,    # fraction of power at 50/60 Hz
#     "min_duration_sec":  10.0,    # minimum usable duration
#     "score_train":       85,      # quality_tag = TRAIN above this
#     "score_review":      60,      # quality_tag = REVIEW above this
#                                   # below → REJECT
# }

QC_THRESHOLDS = {
    "flatline_std":      1e-9,    # V — unchanged, correct for Volts
    "dropout_ptp":       1e-8,    # V — unchanged
    "artifact_amp":      2000e-6, # V — 2mV: only gross artefacts
                                  # was 500e-6 — too tight post-CAR for clinical
    "min_corr":          0.01,    # was 0.05 — clinical EEG has lower correlation
    "max_emg_ratio":     0.50,    # was 0.40 — clinical recordings have more EMG
    "max_line_ratio":    0.40,    # was 0.30 — Indian power grid (50Hz) is variable
    "min_duration_sec":  5.0,     # was 10.0 — clinical segments can be short
    "score_train":       75,      # was 85 — clinical data is noisier
    "score_review":      50,      # was 60
}

# S3 output prefixes per mode
OUTPUT_PREFIXES = {
    "train":    "nema_final_used/clean_train/",
    "external": "nema_final_used/clean_external/",
}

# Label mapping — handles string and numeric labels
LABEL_MAP = {
    "ad":1, "mci":1, "alzheimer":1, "dementia":1, "ftd":1,
    "hc":0, "control":0, "healthy":0, "normal":0, "cn":0,
    "1":1,  "2":1,   "0":0,
    "sz":None, "dep":None, "scz":None, "pd":None,  # excluded
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

def make_logger():
    log = logging.getLogger("data_pipeline")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        log.addHandler(h)
    return log

log = make_logger()

# ─── S3 ───────────────────────────────────────────────────────────────────────

def make_s3():
    return boto3.client("s3")

def s3_exists(s3, bucket, key):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False

def s3_read_bytes(s3, bucket, key):
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

def s3_write_npy(s3, bucket, key, arr: np.ndarray):
    buf = io.BytesIO()
    np.save(buf, arr)
    s3.put_object(Bucket=bucket, Key=key,
                  Body=buf.getvalue(),
                  ContentType="application/octet-stream")

def s3_write_json(s3, bucket, key, obj: dict):
    s3.put_object(Bucket=bucket, Key=key,
                  Body=json.dumps(obj, indent=2).encode(),
                  ContentType="application/json")

def s3_list_keys(s3, bucket, prefix, extensions=None):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if extensions is None or Path(k).suffix.lower() in extensions:
                keys.append(k)
    return sorted(keys)

def s3_read_csv(s3, bucket, key):
    data = s3_read_bytes(s3, bucket, key)
    return pd.read_csv(io.BytesIO(data))

# ─── FORMAT DETECTION ─────────────────────────────────────────────────────────

FORMAT_MAP = {
    ".set":  "eeglab",
    ".edf":  "edf",
    ".bdf":  "bdf",
    ".vhdr": "brainvision",
    ".npy":  "npy",
    ".mat":  "mat",
    ".fif":  "fif",
}

SKIP_EXTENSIONS = {
    ".json",".csv",".tsv",".txt",".pdf",".png",
    ".jpg",".gz",".zip",".log",".vmrk",".eeg",".fdt",
}

def detect_format(key: str):
    ext = Path(key).suffix.lower()
    return FORMAT_MAP.get(ext, None)

# ─── LOADERS ──────────────────────────────────────────────────────────────────

def _download(s3, bucket, key, tmp_dir):
    local = Path(tmp_dir) / Path(key).name
    s3.download_file(bucket, key, str(local))
    return local

def load_eeglab(s3, bucket, key, tmp_dir):
    local = _download(s3, bucket, key, tmp_dir)
    fdt_key = str(Path(key).with_suffix(".fdt"))
    if s3_exists(s3, bucket, fdt_key):
        _download(s3, bucket, fdt_key, tmp_dir)
    raw = mne.io.read_raw_eeglab(str(local), preload=True, verbose=False)
    return raw, raw.info["sfreq"]

def load_brainvision(s3, bucket, key, tmp_dir):
    stem   = Path(key).stem
    parent = str(Path(key).parent)
    for ext in [".vhdr", ".eeg", ".vmrk"]:
        k = f"{parent}/{stem}{ext}"
        local = Path(tmp_dir) / f"{stem}{ext}"
        try:
            s3.download_file(bucket, k, str(local))
        except ClientError:
            if ext == ".vhdr":
                raise FileNotFoundError(f"Required .vhdr missing: {k}")
    raw = mne.io.read_raw_brainvision(
        str(Path(tmp_dir) / f"{stem}.vhdr"), preload=True, verbose=False)
    return raw, raw.info["sfreq"]

def load_edf(s3, bucket, key, tmp_dir):
    local = _download(s3, bucket, key, tmp_dir)
    raw   = mne.io.read_raw_edf(str(local), preload=True, verbose=False)
    return raw, raw.info["sfreq"]

def load_fif(s3, bucket, key, tmp_dir):
    local = _download(s3, bucket, key, tmp_dir)
    raw   = mne.io.read_raw_fif(str(local), preload=True, verbose=False)
    return raw, raw.info["sfreq"]

def load_mat(s3, bucket, key, tmp_dir):
    """Load .mat → (19, T) float32. Handles MATLAB v5 and v7.3 (HDF5)."""
    import scipy.io as sio
    local = _download(s3, bucket, key, tmp_dir)

    # Try scipy (v5/v6)
    try:
        mat = sio.loadmat(str(local), squeeze_me=True, struct_as_record=False)
        mat = {k: v for k, v in mat.items() if not k.startswith("__")}
    except Exception as e_scipy:
        try:
            import h5py
            mat = {}
            with h5py.File(str(local), "r") as f:
                def _ext(obj):
                    if isinstance(obj, h5py.Dataset): return obj[()]
                    elif isinstance(obj, h5py.Group):
                        return {k: _ext(v) for k, v in obj.items()}
                    return obj
                for k in f.keys():
                    mat[k] = _ext(f[k])
        except Exception as e_h5:
            raise RuntimeError(f"Cannot load .mat\n  scipy: {e_scipy}\n  h5py: {e_h5}")

    # Find 2D array with 19 channels
    candidates = [
        (k, v) for k, v in mat.items()
        if isinstance(v, np.ndarray) and v.ndim == 2 and 19 in v.shape
    ]
    if not candidates:
        raise ValueError(
            f"No 19-channel 2D array in .mat. "
            f"Keys: {[(k, getattr(v,'shape','?')) for k,v in mat.items()]}"
        )
    key_name, eeg = candidates[0]
    if eeg.shape[0] != 19:
        eeg = eeg.T
    log.info(f"    .mat key='{key_name}'  shape={eeg.shape}")
    # .mat has no sfreq — caller must infer
    return eeg.astype(np.float32), None

def load_npy(s3, bucket, key, tmp_dir):
    """Load .npy → (19, T) float32."""
    local = _download(s3, bucket, key, tmp_dir)
    eeg   = np.load(str(local), allow_pickle=True)
    if eeg.dtype == object:
        eeg = eeg.item()
    eeg = np.array(eeg, dtype=np.float32)
    if eeg.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {eeg.shape}")
    if eeg.shape[0] != 19 and eeg.shape[1] == 19:
        eeg = eeg.T
    elif eeg.shape[0] != 19:
        raise ValueError(f"Cannot find 19 channels in shape {eeg.shape}")
    return eeg, None

# ─── CHANNEL STANDARDISATION ──────────────────────────────────────────────────

def _normalise_name(ch):
    """Apply alias map to normalise a channel name."""
    return CHANNEL_ALIASES.get(ch, CHANNEL_ALIASES.get(ch.upper(), ch))

def standardise_channels_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Select and reorder MNE raw to CONTRACT 19-channel order.
    Raises ValueError if any required channel is missing.
    """
    available  = raw.ch_names
    normalised = {_normalise_name(ch): ch for ch in available}

    ch_map  = {}
    missing = []
    for tch in CONTRACT["channel_order"]:
        if tch in normalised:
            ch_map[tch] = normalised[tch]
        else:
            missing.append(tch)

    if missing:
        raise ValueError(
            f"Required channels not found: {missing}\n"
            f"Available (normalised): {list(normalised.keys())[:25]}"
        )

    raw.pick_channels([ch_map[t] for t in CONTRACT["channel_order"]],
                      ordered=True)
    rename = {ch_map[t]: t for t in CONTRACT["channel_order"]
              if ch_map[t] != t}
    if rename:
        raw.rename_channels(rename)
    return raw

def standardise_channels_npy(eeg: np.ndarray,
                              source_channels: list) -> np.ndarray:
    """
    Reorder numpy array from source_channels to CONTRACT order.
    Used for .mat and .npy files where channel names are known.
    """
    normalised = [_normalise_name(c) for c in source_channels]
    order = []
    for tch in CONTRACT["channel_order"]:
        for si, sch in enumerate(normalised):
            if sch.lower() == tch.lower():
                order.append(si)
                break
        else:
            raise ValueError(
                f"Channel '{tch}' not found in source.\n"
                f"Source (normalised): {normalised}"
            )
    return eeg[order]

# ─── UNIT DETECTION & CONVERSION ─────────────────────────────────────────────

def detect_and_convert_units(eeg: np.ndarray):
    """
    Detect signal units from amplitude and convert to Volts if needed.

    Heuristic:
      std < 1e-3  → already Volts  (clinical EEG ≈ 20–50 μV = 2–5e-5 V)
      std > 0.5   → ambiguous large values
      else        → microvolts (divide by 1e6)

    Returns (eeg_volts, original_units, conversion_applied)
    """
    std = float(eeg.std())

    if std < 1e-3:
        return eeg, "Volts", False

    elif 0.1 <= std <= 1000:
        # Microvolts — clinical EEG is typically 1–200 μV
        log.info(f"    Units: microvolts (std={std:.2f}) → converting to Volts")
        return (eeg / 1e6).astype(np.float32), "microvolts", True

    else:
        log.warning(
            f"    Units: UNKNOWN (std={std:.4f}) — "
            f"signal may be in unusual units. Saving as-is."
        )
        return eeg, "unknown", False

# ─── RESAMPLING ───────────────────────────────────────────────────────────────

def resample_to_contract(eeg: np.ndarray, fs_in: float) -> np.ndarray:
    """
    Polyphase resample (19, T_in) → (19, T_out) at CONTRACT sfreq.
    Uses scipy.signal.resample_poly — anti-aliased, minimal ringing.
    """
    fs_out = CONTRACT["sfreq"]
    if abs(fs_in - fs_out) < 0.5:
        return eeg.astype(np.float32)

    g    = gcd(int(round(fs_in)), fs_out)
    up   = fs_out // g
    down = int(round(fs_in)) // g
    log.info(f"    Resampling {fs_in:.0f}→{fs_out}Hz "
             f"(polyphase up={up} down={down})")
    return scipy_signal.resample_poly(
        eeg, up, down, axis=1
    ).astype(np.float32)

# ─── PREPROCESSING ────────────────────────────────────────────────────────────

def preprocess(eeg: np.ndarray) -> np.ndarray:
    """
    Standard preprocessing — IDENTICAL for training and external cohort.

    Steps:
      1. NaN/Inf → 0
      2. Linear detrend (removes DC offset and linear drift)
      3. Bandpass 0.5–45 Hz, 4th order Butterworth, zero-phase
      4. Common Average Reference (CAR)

    Input/output: (19, T) float32 in Volts
    """
    sfreq = CONTRACT["sfreq"]

    # 1. Clean
    eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)

    # 2. Detrend per channel
    from scipy.signal import detrend as sp_detrend
    eeg = sp_detrend(eeg, axis=1).astype(np.float32)

    # 3. Bandpass 0.5–45 Hz
    nyq = sfreq / 2.0
    sos = butter(4, [0.5 / nyq, 45.0 / nyq], btype="bandpass", output="sos")
    eeg = sosfiltfilt(sos, eeg, axis=1).astype(np.float32)

    # 4. CAR
    eeg = (eeg - eeg.mean(axis=0, keepdims=True)).astype(np.float32)

    return eeg

# ─── QC ───────────────────────────────────────────────────────────────────────

def run_qc(eeg: np.ndarray) -> dict:
    """
    Full QC suite. All thresholds in Volts.
    Returns metrics dict + quality_score (0–100) + quality_tag.
    """
    sfreq  = CONTRACT["sfreq"]
    n_ch, n_t = eeg.shape
    duration  = n_t / sfreq
    thr = QC_THRESHOLDS

    # ── Channel checks ────────────────────────────────────────────────────────
    stds             = eeg.std(axis=1)
    flat_channels    = int((stds < thr["flatline_std"]).sum())
    dropout_channels = int((np.ptp(eeg, axis=1) < thr["dropout_ptp"]).sum())

    corr_mat  = np.corrcoef(eeg)
    bad_corr  = 0
    for i in range(n_ch):
        vals = np.delete(corr_mat[i], i)
        if np.mean(np.abs(vals)) < thr["min_corr"]:
            bad_corr += 1

    # ── Window-level artefact ─────────────────────────────────────────────────
    win_size = 1024 #512
    total_w  = bad_w = 0
    for s in range(0, n_t - win_size, win_size):
        total_w += 1
        if np.max(np.abs(eeg[:, s:s+win_size])) > thr["artifact_amp"]:
            bad_w += 1
    bad_win_ratio = bad_w / max(total_w, 1)

    # ── Spectral checks ───────────────────────────────────────────────────────
    emg_ratios, line_ratios, entropies = [], [], []
    for ch in eeg:
        freqs, psd = welch(ch, fs=sfreq, nperseg=min(512, n_t))
        total      = np.trapezoid(psd, freqs) + 1e-30

        # EMG: power above 30 Hz
        mask_emg   = freqs > 30
        emg_ratios.append(
            np.trapezoid(psd[mask_emg], freqs[mask_emg]) / total
            if mask_emg.any() else 0.0
        )
        # Line noise: 49–51 Hz or 59–61 Hz
        mask_ln = ((freqs >= 49) & (freqs <= 51)) | \
                  ((freqs >= 59) & (freqs <= 61))
        line_ratios.append(
            np.trapezoid(psd[mask_ln], freqs[mask_ln]) / total
            if mask_ln.any() else 0.0
        )
        # Spectral entropy
        pn = psd / (psd.sum() + 1e-30)
        entropies.append(float(scipy_entropy(pn + 1e-30)))

    emg_score   = float(np.mean(emg_ratios))
    line_score  = float(np.mean(line_ratios))
    mean_ent    = float(np.mean(entropies))

    # Peak alpha frequency (occipital: O1=index 17, O2=index 18)
    paf_vals = []
    for ci in [17, 18]:
        if ci < n_ch:
            freqs_w, psd_w = welch(eeg[ci], fs=sfreq, nperseg=min(512, n_t))
            alpha_mask = (freqs_w >= 8) & (freqs_w <= 13)
            if alpha_mask.any():
                paf_vals.append(
                    float(freqs_w[alpha_mask][np.argmax(psd_w[alpha_mask])])
                )
    paf = float(np.mean(paf_vals)) if paf_vals else None

    # ── Score ─────────────────────────────────────────────────────────────────
    penalties = 0.0
    penalties += flat_channels    * 12.0
    penalties += dropout_channels * 10.0
    penalties += bad_corr         *  5.0
    penalties += bad_w            *  4.0
    penalties += min(emg_score  / thr["max_emg_ratio"],  1.0) * 15.0
    penalties += min(line_score / thr["max_line_ratio"],  1.0) * 10.0
    if duration < thr["min_duration_sec"]:
        penalties += 30.0

    score = max(0.0, 100.0 - penalties)
    tag   = ("TRAIN"  if score >= thr["score_train"]  else
             "REVIEW" if score >= thr["score_review"] else
             "REJECT")

    return {
        "quality_score":       round(score, 2),
        "quality_tag":         tag,
        "duration_sec":        round(duration, 2),
        "n_samples":           n_t,
        "flat_channels":       flat_channels,
        "dropout_channels":    dropout_channels,
        "bad_corr_channels":   bad_corr,
        "bad_windows":         bad_w,
        "total_windows":       total_w,
        "bad_window_ratio":    round(bad_win_ratio, 4),
        "emg_score":           round(emg_score, 5),
        "line_noise_score":    round(line_score, 5),
        "spectral_entropy":    round(mean_ent, 4),
        "peak_alpha_freq_hz":  round(paf, 2) if paf else None,
        "mean_amplitude_uv":   round(float(np.mean(np.abs(eeg))) * 1e6, 2),
        "max_amplitude_uv":    round(float(np.max(np.abs(eeg)))  * 1e6, 2),
    }

# ─── CONTRACT VALIDATION ──────────────────────────────────────────────────────

def validate_output(eeg: np.ndarray) -> list:
    """
    Hard checks on the output array before writing to S3.
    Returns list of violation strings — empty means pass.
    """
    v = []
    if eeg.ndim != 2:
        v.append(f"ndim={eeg.ndim} expected 2")
    if eeg.shape[0] != CONTRACT["n_channels"]:
        v.append(f"n_channels={eeg.shape[0]} expected {CONTRACT['n_channels']}")
    if eeg.dtype != np.float32:
        v.append(f"dtype={eeg.dtype} expected float32")
    if np.any(np.isnan(eeg)):
        v.append("contains_NaN")
    if np.any(np.isinf(eeg)):
        v.append("contains_Inf")
    if eeg.std() > 1.0:
        v.append(f"std={eeg.std():.4f}_likely_still_microvolts")
    if eeg.std() < 1e-10:
        v.append(f"std={eeg.std():.2e}_signal_is_flat")
    return v

# ─── SIDECAR BUILDER ─────────────────────────────────────────────────────────

def build_sidecar(eeg: np.ndarray, qc: dict, provenance: dict) -> dict:
    return {
        # Contract fields — what downstream scripts can rely on
        "contract": {
            "shape":          list(eeg.shape),
            "dtype":          str(eeg.dtype),
            "units":          CONTRACT["units"],
            "sfreq":          CONTRACT["sfreq"],
            "channel_order":  CONTRACT["channel_order"],
            "preprocessing":  CONTRACT["preprocessing"],
            "normalisation":  CONTRACT["normalisation"],
        },
        # QC
        "qc": qc,
        # Provenance
        "provenance": {
            **provenance,
            "pipeline_version": "2.0",
            "pipeline_script":  "data_pipeline.py",
            "processed_at":     datetime.now(timezone.utc).isoformat(),
        },
    }

# ─── CORE: PROCESS ONE FILE ───────────────────────────────────────────────────

def process_one(
    s3_client,
    bucket:       str,
    source_key:   str,
    clean_prefix: str,
    dataset:      str,
    label:        int  = None,
    subject_id:   str  = None,
    known_sfreq:  float = None,
    known_channels: list = None,
    skip_existing: bool = True,
) -> dict:
    """
    Full pipeline for one EEG file.
    Returns a manifest row dict regardless of success or failure.
    """
    stem     = Path(source_key).stem
    sid      = subject_id or stem
    fmt      = detect_format(source_key)

    out_npy  = f"{clean_prefix.rstrip('/')}/{dataset}/{stem}.npy"
    out_json = f"{clean_prefix.rstrip('/')}/{dataset}/{stem}.json"

    # Skip if already processed
    if skip_existing and s3_exists(s3_client, bucket, out_npy):
        log.info(f"  [SKIP] {stem} already exists")
        return _make_row(stem, sid, dataset, label, fmt,
                         source_key, out_npy, out_json,
                         0, "EXISTING", [])

    if fmt is None:
        log.info(f"  [SKIP] {stem} — unsupported format")
        return _make_row(stem, sid, dataset, label, fmt or "unknown",
                         source_key, out_npy, out_json,
                         0, "SKIP", ["unsupported_format"])

    log.info(f"  [{fmt.upper()}] {stem}  label={label}")

    errors          = []
    original_sfreq  = None
    original_units  = "unknown"
    unit_converted  = False
    ch_verified     = False

    with tempfile.TemporaryDirectory() as tmp:
        # ── LOAD ─────────────────────────────────────────────────────────────
        try:
            if fmt == "eeglab":
                raw, original_sfreq = load_eeglab(s3_client, bucket, source_key, tmp)
                raw = standardise_channels_raw(raw)
                eeg = raw.get_data().astype(np.float32)
                ch_verified = True
                del raw

            elif fmt == "brainvision":
                raw, original_sfreq = load_brainvision(s3_client, bucket, source_key, tmp)
                raw = standardise_channels_raw(raw)
                eeg = raw.get_data().astype(np.float32)
                ch_verified = True
                del raw

            elif fmt in ("edf", "bdf"):
                raw, original_sfreq = load_edf(s3_client, bucket, source_key, tmp)
                raw = standardise_channels_raw(raw)
                eeg = raw.get_data().astype(np.float32)
                ch_verified = True
                del raw

            elif fmt == "fif":
                raw, original_sfreq = load_fif(s3_client, bucket, source_key, tmp)
                raw = standardise_channels_raw(raw)
                eeg = raw.get_data().astype(np.float32)
                ch_verified = True
                del raw

            elif fmt == "mat":
                eeg, _ = load_mat(s3_client, bucket, source_key, tmp)
                original_sfreq = known_sfreq
                if known_channels:
                    eeg = standardise_channels_npy(eeg, known_channels)
                    ch_verified = True
                else:
                    log.warning(f"    No channel list for .mat — "
                                f"assuming CONTRACT order")
                    errors.append("channel_order_unverified")

            elif fmt == "npy":
                eeg, _ = load_npy(s3_client, bucket, source_key, tmp)
                original_sfreq = known_sfreq
                if known_channels:
                    eeg = standardise_channels_npy(eeg, known_channels)
                    ch_verified = True
                else:
                    errors.append("channel_order_unverified")

            else:
                raise ValueError(f"Unhandled format: {fmt}")

        except Exception as e:
            log.error(f"    LOAD FAILED: {e}")
            errors.append(f"load_failed:{e}")
            _write_failure_sidecar(s3_client, bucket, out_json,
                                   dataset, source_key, sid, label,
                                   fmt, errors)
            return _make_row(stem, sid, dataset, label, fmt,
                             source_key, out_npy, out_json,
                             0, "REJECT", errors)

        # ── UNIT CONVERSION ───────────────────────────────────────────────────
        eeg, original_units, unit_converted = detect_and_convert_units(eeg)

        # ── RESAMPLE ─────────────────────────────────────────────────────────
        if original_sfreq is None:
            original_sfreq = known_sfreq
        if original_sfreq is None:
            log.warning(f"    sfreq unknown — assuming {CONTRACT['sfreq']}Hz")
            errors.append("sfreq_assumed_256")
        else:
            eeg = resample_to_contract(eeg, original_sfreq)

        # ── PREPROCESS ───────────────────────────────────────────────────────
        eeg = preprocess(eeg)

        # ── CONTRACT VALIDATION ───────────────────────────────────────────────
        violations = validate_output(eeg)
        if violations:
            errors.extend(violations)
            log.warning(f"    Contract violations: {violations}")

        # ── QC ───────────────────────────────────────────────────────────────
        qc = run_qc(eeg)
        log.info(
            f"    QC: {qc['quality_tag']}  "
            f"score={qc['quality_score']:.1f}  "
            f"dur={qc['duration_sec']:.0f}s  "
            f"flat={qc['flat_channels']}  "
            f"bad_win={qc['bad_windows']}  "
            f"paf={qc['peak_alpha_freq_hz']}Hz"
        )

        # ── WRITE ─────────────────────────────────────────────────────────────
        s3_write_npy(s3_client, bucket, out_npy, eeg)

        provenance = {
            "dataset":              dataset,
            "source_key":           source_key,
            "clean_npy_key":        out_npy,
            "clean_json_key":       out_json,
            "label":                label,
            "subject_id":           sid,
            "format":               fmt,
            "original_sfreq":       original_sfreq,
            "original_units":       original_units,
            "unit_conversion":      unit_converted,
            "channel_order_verified": ch_verified,
            "errors":               errors,
        }
        sidecar = build_sidecar(eeg, qc, provenance)
        s3_write_json(s3_client, bucket, out_json, sidecar)

        log.info(f"    ✓ → s3://{bucket}/{out_npy}")

    return _make_row(stem, sid, dataset, label, fmt,
                     source_key, out_npy, out_json,
                     qc["quality_score"], qc["quality_tag"],
                     errors, qc)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _make_row(stem, sid, dataset, label, fmt,
              source_key, out_npy, out_json,
              score, tag, errors, qc=None):
    row = {
        "subject_id":      sid,
        "stem":            stem,
        "dataset":         dataset,
        "true_label":      label,
        "format":          fmt,
        "source_key":      source_key,
        "clean_npy":       out_npy,
        "clean_json":      out_json,
        "quality_score":   score,
        "quality_tag":     tag,
        "errors":          "; ".join(str(e) for e in errors),
        "processed_at":    datetime.now(timezone.utc).isoformat(),
    }
    if qc:
        row.update({
            "duration_sec":       qc.get("duration_sec"),
            "peak_alpha_freq_hz": qc.get("peak_alpha_freq_hz"),
            "flat_channels":      qc.get("flat_channels"),
            "bad_windows":        qc.get("bad_windows"),
            "emg_score":          qc.get("emg_score"),
        })
    return row

def _write_failure_sidecar(s3_client, bucket, out_json,
                            dataset, source_key, sid, label,
                            fmt, errors):
    sidecar = {
        "contract": {"shape": None, "units": None},
        "qc": {"quality_score": 0, "quality_tag": "REJECT"},
        "provenance": {
            "dataset": dataset, "source_key": source_key,
            "subject_id": sid, "label": label, "format": fmt,
            "errors": errors,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
    }
    try:
        s3_write_json(s3_client, bucket, out_json, sidecar)
    except Exception:
        pass

# ─── DATASET RUNNER ───────────────────────────────────────────────────────────

def run_dataset(
    s3_client,
    bucket:        str,
    raw_prefix:    str,
    clean_prefix:  str,
    dataset:       str,
    labels_csv_key: str = None,
    manifest_prefix: str = "nema_final_used/manifests/",
    known_sfreq:   float = None,
    known_channels: list = None,
    skip_existing: bool  = True,
) -> pd.DataFrame:
    """
    Process all EEG files in one dataset.
    Writes per-file .npy + .json and a run manifest CSV.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Dataset : {dataset}")
    # log.info(f"Source  : s3://{bucket}/{raw_prefix}")
    log.info(f"Output  : s3://{bucket}/{clean_prefix}{dataset}/")
    log.info(f"{'='*60}")

    dataset_prefix = raw_prefix.rstrip("/")

    if dataset_prefix.split("/")[-1] != dataset:
        dataset_prefix = f"{dataset_prefix}/{dataset}"

    log.info(f"Source  : s3://{bucket}/{dataset_prefix}")

    # GUL # Load label lookup
    # label_lookup = {}
    # if labels_csv_key:
    #     try:
    #         ldf = s3_read_csv(s3_client, bucket, labels_csv_key)
    #         ldf.columns = ldf.columns.str.strip().str.lower()
    #         id_col = next((c for c in ldf.columns
    #                        if "id" in c or "subject" in c), ldf.columns[0])
    #         dx_col = next((c for c in ldf.columns
    #                        if "label" in c or "diag" in c
    #                        or "dx" in c or "group" in c or "class" in c
    #                        ), None)
    #         if dx_col:
    #             for _, row in ldf.iterrows():
    #                 sid = str(row[id_col]).strip()
    #                 dx  = str(row[dx_col]).strip().lower()
    #                 label_lookup[sid] = LABEL_MAP.get(dx, None)
    #         log.info(f"Labels: {len(label_lookup)} entries loaded")
    #     except Exception as e:
    #         log.warning(f"Labels CSV load failed: {e}")

    # Load labels from ingestion manifest
    label_lookup = {}

    # manifest_key = f"{raw_prefix.rstrip('/')}/manifest_{dataset}.csv"
    manifest_name = dataset.replace("-", "_")

    manifest_key = (
        f"{dataset_prefix}/"
        f"manifest_{manifest_name}.csv"
    )

    log.info(f"Reading manifest: {manifest_key}")

    try:

        mdf = s3_read_csv(
            s3_client,
            bucket,
            manifest_key,
        )

        mdf.columns = mdf.columns.str.strip()

        # label_lookup = dict(
        #     zip(
        #         mdf["subject_id"].astype(str),
        #         mdf["true_label"],
        #     )
        # )

        lookup_col = "stem" if "stem" in mdf.columns else "subject_id"

        label_lookup = dict(
            zip(
                mdf[lookup_col].astype(str),
                mdf["true_label"],
            )
        )

        log.info(
            f"Manifest loaded: {len(label_lookup)} labels"
        )

    except Exception as e:

        log.warning(
            f"Manifest load failed: {e}"
        )

    # Discover EEG files
    # all_keys = s3_list_keys(s3_client, bucket, raw_prefix)
    all_keys = s3_list_keys(
        s3_client,
        bucket,
        dataset_prefix,
    )
    eeg_keys = [
        k for k in all_keys
        if Path(k).suffix.lower() not in SKIP_EXTENSIONS
        and detect_format(k) is not None
    ]
    log.info(f"Found {len(eeg_keys)} EEG files")

    rows    = []
    n_ok    = n_err = n_skip = 0

    for i, key in enumerate(eeg_keys):
        stem = Path(key).stem

        # GUL # Resolve label
        # label = None
        # for candidate in [stem,
        #                    stem.split("_")[0],
        #                    stem.split("-")[0],
        #                    stem.split(".")[0]]:
        #     if candidate in label_lookup:
        #         label = label_lookup[candidate]
        #         break
        # if label is None and len(label_lookup) > 0:
        #     log.warning(f"  No label for {stem}")

        label = label_lookup.get(stem)

        if label is None:
            log.warning(f"  No label for {stem}")

        log.info(f"\n[{i+1}/{len(eeg_keys)}]")
        dataset_json = {"ADFSU":128, "DS004504":500, "BrainLat":512 , "Isfahan": 200, "P-ADIC":500, "DS003800":250, "BACA_train":1000, "BACA_longitudinal":1000,"PEARL-Neuro":1000, "CAUEEG":200,"nicolet_test": 256, "APAVA":256, "FIGSHARE":256, "ADSZ-AD":256, "FIGSHARE-128Hz":128, "FIGSHARE-256Hz":256, "DS005048": 250,}
        # if dataset =="ADFSU":
        #     known_sfreq = 
        

        try:
            row = process_one(
                s3_client    = s3_client,
                bucket       = bucket,
                source_key   = key,
                clean_prefix = clean_prefix,
                dataset      = dataset,
                label        = label,
                subject_id   = stem,
                known_sfreq  = dataset_json[dataset],
                known_channels = known_channels,
                skip_existing= skip_existing,
            )
            rows.append(row)
            if row["quality_tag"] == "EXISTING":
                n_skip += 1
            elif row["quality_tag"] != "REJECT":
                n_ok += 1
            else:
                n_err += 1
        except Exception as e:
            log.error(f"  UNHANDLED: {e}")
            traceback.print_exc()
            n_err += 1

    # Write manifest
    mdf = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not mdf.empty:
        ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        mkey = f"{manifest_prefix}{dataset}_{ts}.csv"
        buf  = io.BytesIO(mdf.to_csv(index=False).encode())
        s3_client.put_object(Bucket=bucket, Key=mkey,
                             Body=buf.getvalue(), ContentType="text/csv")
        log.info(f"\nManifest → s3://{bucket}/{mkey}")
        log.info(f"  Total    : {len(mdf)}")
        log.info(f"  TRAIN    : {(mdf.quality_tag=='TRAIN').sum()}")
        log.info(f"  REVIEW   : {(mdf.quality_tag=='REVIEW').sum()}")
        log.info(f"  REJECT   : {(mdf.quality_tag=='REJECT').sum()}")
        log.info(f"  Existing : {n_skip}")
        log.info(f"  Label=1  : {(mdf.true_label==1).sum()}")
        log.info(f"  Label=0  : {(mdf.true_label==0).sum()}")

    return mdf

# ─── CLI ──────────────────────────────────────────────────────────────────────
BUCKET = "dementia-research2025"
S3_SRC_ROOT = "nema_final_used/final_npy_raw//"                  # where raw source files live

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="NEMA EEG Data Pipeline — Load, Standardise, QC, Save to S3"
    )
    p.add_argument("--bucket",   default =  BUCKET,   required=False)
    p.add_argument("--mode",    default = "train"  ,   #required=True,
                   choices=["train","external"],
                   help="train → clean_train/  |  external → clean_external/")
    p.add_argument("--dataset",      default=None,
                   help="Dataset name (subfolder under raw_prefix)")
    p.add_argument("--raw_prefix",   default=S3_SRC_ROOT,
                   help="S3 prefix for raw input files")
    p.add_argument("--labels_csv",   default=None,
                   help="S3 key of labels.csv for this dataset")
    p.add_argument("--all_datasets", action="store_true",
                   help="Process all subfolders under raw_prefix")
    p.add_argument("--known_sfreq",  type=float, default=None,
                   help="Original sampling rate (required for .mat/.npy sources)")
    p.add_argument("--clean_prefix", default=None,
                   help="Override default clean output prefix")
    p.add_argument("--manifest_prefix",
                   default="nema_final_used/manifests/")
    p.add_argument("--no_skip",      action="store_true",
                   help="Reprocess files that already exist")

    args = p.parse_args()

    s3_client    = make_s3()
    clean_prefix = args.clean_prefix or OUTPUT_PREFIXES[args.mode]
    skip         = not args.no_skip

    if args.all_datasets:
        # Discover all subfolders under raw_prefix
        assert args.raw_prefix, "--raw_prefix required with --all_datasets"
        paginator = s3_client.get_paginator("list_objects_v2")
        datasets  = set()
        for page in paginator.paginate(
            Bucket=args.bucket,
            Prefix=args.raw_prefix,
            Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                ds = cp["Prefix"].rstrip("/").split("/")[-1]
                datasets.add(ds)

        log.info(f"Discovered {len(datasets)} datasets: {sorted(datasets)}")

        for ds in sorted(datasets):
            pfx = f"{args.raw_prefix.rstrip('/')}/{ds}/"
            lbl = f"{pfx}labels.csv"
            if not s3_exists(s3_client, args.bucket, lbl):
                lbl = None
                log.warning(f"No labels.csv for {ds} — labels will be None")
            run_dataset(
                s3_client      = s3_client,
                bucket         = args.bucket,
                raw_prefix     = pfx,
                clean_prefix   = clean_prefix,
                dataset        = ds,
                labels_csv_key = lbl,
                manifest_prefix= args.manifest_prefix,
                known_sfreq    = args.known_sfreq,
                skip_existing  = skip,
            )

    else:
        assert args.dataset and args.raw_prefix, \
            "--dataset and --raw_prefix required"
        run_dataset(
            s3_client      = s3_client,
            bucket         = args.bucket,
            raw_prefix     = args.raw_prefix,
            clean_prefix   = clean_prefix,
            dataset        = args.dataset,
            labels_csv_key = args.labels_csv,
            manifest_prefix= args.manifest_prefix,
            known_sfreq    = args.known_sfreq,
            skip_existing  = skip,
        )