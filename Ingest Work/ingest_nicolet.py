"""
ingest_nicolet_txt.py — Nicolet/Xltek .txt Export Ingestion to S3
==================================================================
Converts Nicolet EEG system tab/space-delimited .txt exports to
(19, T) float32 .npy files on S3.

Supported variants (auto-detected from header):
  Variant A — 32-channel Nicolet Quantum (Headbox SN ~2)
    Columns: Date Time EB Stamp C001..C032 TRIGGER
    EEG: C001–C019 (19 standard 10-20)
    Non-EEG: C020–C032 (mastoids, ECG, ground)

  Variant B — 148-channel Nicolet Xltek (Headbox SN 31644)
    Columns: Date Time EB Stamp C001..C148 PHOTIC
    EEG: C001–C128 (128 EEG channels)
    Non-EEG: C129–C148 (DC, impedance, hardware channels)
    Note: "SHORT" in data = disconnected electrode → zero-filled

Both variants:
  - Units: mV → divide by 1000 to get Volts
  - Sampling rate: 255.9978 Hz or 256.000000 Hz → rounded to 256
  - Channel names: generic C001..CXXX → need mapping to 10-20 names
  - Last column (TRIGGER or PHOTIC): always dropped
  - Non-numeric tokens (OFF, ON, SHORT): zero-filled

Channel map (MUST verify against your headbox documentation):
  The C001..CXXX slot numbering maps to electrode positions defined
  by the physical cap/headbox used at recording time.
  Provide --channel_map_key with a CSV (columns: slot, name) like:
    C001,Fp1
    C002,Fp2
    ...
  If not provided, the script uses the default maps below but warns.

S3 layout:
  Source: nema_final_used/original_data/{DATASET}/*.txt
  Output: nema_final_used/all_npy_raw/{DATASET}/{stem}.npy

Usage:
  # Inspect — always run this first on a new file type
  python ingest_nicolet.py inspect \
      --bucket dementia-research2025 \
      --key    nema_final_used/original_data/nicolet_test/POONAM_Export.txt

  # Full ingestion with channel map (recommended)
  python ingest_nicolet_txt.py ingest \
      --bucket          dementia-research2025 \
      --src_prefix      nema_final_used/original_data/nicolet_test/ \
      --dataset         nicolet_test \
      --labels_csv      nema_final_used/original_data/nicolet_test/labels.csv \
      --channel_map_key nema_final_used/original_data/nicolet_test/channel_map.csv

  # Without channel map — uses defaults, verify output carefully
  python ingest_nicolet.py ingest \
      --bucket     dementia-research2025 \
      --src_prefix nema_final_used/original_data/nicolet_test/ \
      --dataset    nicolet_test \
      --labels_csv nema_final_used/original_data/nicolet_test/labels.csv
"""
"""
ingest_nicolet_txt.py — Nicolet/Xltek .txt Export Ingestion to S3
==================================================================
Converts Nicolet EEG system tab/space-delimited .txt exports to
(19, T) float32 .npy files on S3.

Supported variants (auto-detected from header):
  Variant A — 32-channel Nicolet Quantum (Headbox SN ~2)
    Columns: Date Time EB Stamp C001..C032 TRIGGER
    EEG: C001–C019 (19 standard 10-20)
    Non-EEG: C020–C032 (mastoids, ECG, ground)

  Variant B — 148-channel Nicolet Xltek (Headbox SN 31644)
    Columns: Date Time EB Stamp C001..C148 PHOTIC
    EEG: C001–C128 (128 EEG channels)
    Non-EEG: C129–C148 (DC, impedance, hardware channels)
    Note: "SHORT" in data = disconnected electrode → zero-filled

Both variants:
  - Units: mV → divide by 1000 to get Volts
  - Sampling rate: 255.9978 Hz or 256.000000 Hz → rounded to 256
  - Channel names: generic C001..CXXX → need mapping to 10-20 names
  - Last column (TRIGGER or PHOTIC): always dropped
  - Non-numeric tokens (OFF, ON, SHORT): zero-filled

Channel map (MUST verify against your headbox documentation):
  The C001..CXXX slot numbering maps to electrode positions defined
  by the physical cap/headbox used at recording time.
  Provide --channel_map_key with a CSV (columns: slot, name) like:
    C001,Fp1
    C002,Fp2
    ...
  If not provided, the script uses the default maps below but warns.

S3 layout:
  Source: nema_final_used/original_data/{DATASET}/*.txt
  Output: nema_final_used/all_npy_raw/{DATASET}/{stem}.npy

Usage:
  # Inspect — always run this first on a new file type
  python ingest_nicolet_txt.py inspect \
      --bucket dementia-research2025 \
      --key    nema_final_used/original_data/NIMHANS/POONAM.txt

  # Full ingestion with channel map (recommended)
  python ingest_nicolet_txt.py ingest \
      --bucket          dementia-research2025 \
      --src_prefix      nema_final_used/original_data/NIMHANS/ \
      --dataset         NIMHANS \
      --labels_csv      nema_final_used/original_data/NIMHANS/labels.csv \
      --channel_map_key nema_final_used/original_data/NIMHANS/channel_map.csv

  # Without channel map — uses defaults, verify output carefully
  python ingest_nicolet_txt.py ingest \
      --bucket     dementia-research2025 \
      --src_prefix nema_final_used/original_data/NIMHANS/ \
      --dataset    NIMHANS \
      --labels_csv nema_final_used/original_data/NIMHANS/labels.csv
"""

import argparse
import io
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from botocore.exceptions import ClientError

# ─── CONFIG ───────────────────────────────────────────────────────────────────

BUCKET   = "dementia-research2025"
OUT_ROOT = "nema_final_used/all_npy_raw"

MASTER_19 = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5","P3","Pz","P4","T6",
    "O1","O2",
]

CHANNEL_ALIASES = {
    "FP1":"Fp1","FP2":"Fp2","FZ":"Fz","CZ":"Cz","PZ":"Pz",
    "T7":"T3","T8":"T4","P7":"T5","P8":"T6",
    "FP1-REF":"Fp1","FP2-REF":"Fp2",
}

# ── Default Nicolet Quantum 32-channel slot → 10-20 electrode map ─────────────
# This is the most common Nicolet clinical 32ch ordering.
# Slots not in MASTER_19 (reference, ECG, etc.) are mapped to None → dropped.
# VERIFY THIS AGAINST YOUR HEADBOX DOCUMENTATION.
# If wrong, pass --channel_map_key with the correct mapping CSV.
DEFAULT_32CH_MAP = {
    "C001": "Fp1",  "C002": "Fp2",
    "C003": "F7",   "C004": "F3",   "C005": "Fz",   "C006": "F4",   "C007": "F8",
    "C008": "T3",   "C009": "C3",   "C010": "Cz",   "C011": "C4",   "C012": "T4",
    "C013": "T5",   "C014": "P3",   "C015": "Pz",   "C016": "P4",   "C017": "T6",
    "C018": "O1",   "C019": "O2",
    "C020": None,   # A1 (left mastoid reference)
    "C021": None,   # A2 (right mastoid reference)
    "C022": None,   # Nasion or Cz2
    "C023": None,   # ECG / non-EEG
    "C024": None,   # ECG2
    "C025": None,   # EMG or ground
    "C026": None,
    "C027": None,
    "C028": None,
    "C029": None,
    "C030": None,
    "C031": None,
    "C032": None,   # TRIGGER or ground
}

# ── Default Nicolet Xltek 148-channel map ─────────────────────────────────────
# Headbox SN 31644 — 148 channels total
# C001–C128: EEG channels (standard 10-20 subset within first ~25 slots)
# C129–C148: DC offset / impedance / hardware channels → ALL dropped
#
# The exact ordering of C001–C128 depends on your cap.
# COMMON Xltek clinical cap (verify against your documentation):
# First 19 slots = standard 10-20, remaining 109 = extended 10-10 or unused
DEFAULT_128CH_MAP = {
    # Standard 10-20 in first 19 slots
    "C001": "Fp1",  "C002": "Fp2",
    "C003": "F7",   "C004": "F3",   "C005": "Fz",   "C006": "F4",   "C007": "F8",
    "C008": "T3",   "C009": "C3",   "C010": "Cz",   "C011": "C4",   "C012": "T4",
    "C013": "T5",   "C014": "P3",   "C015": "Pz",   "C016": "P4",   "C017": "T6",
    "C018": "O1",   "C019": "O2",
    # C020–C128: extended channels / reference / unused → None (dropped)
    **{f"C{i:03d}": None for i in range(20, 129)},
    # C129–C148: hardware DC/impedance channels → None (dropped)
    **{f"C{i:03d}": None for i in range(129, 149)},
}

# ── Non-numeric token values in data lines ────────────────────────────────────
# These appear instead of float values in Nicolet exports:
#   SHORT    = electrode disconnected / short circuit → treat as 0
#   OFF      = event byte state (in EB column only)
#   ON       = event byte state (in EB column only)
#   PHOTIC   = photic column header / value (last column, always dropped)
NON_NUMERIC_TOKENS = {"SHORT", "OFF", "ON", "PHOTIC", "TRIGGER", ""}

# Amplitude threshold to detect hardware channels (not EEG)
# EEG in mV: typically ±5 mV max
# DC/impedance channels: can be ±100 mV or more (like the -55 values seen)
HARDWARE_CHANNEL_THRESHOLD_MV = 20.0

LABEL_MAP = {
    "ad":1, "mci":1, "alzheimer":1, "dementia":1, "1":1,
    "hc":0, "control":0, "healthy":0, "normal":0,  "0":0,
}

s3 = boto3.client("s3")

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def s3_exists(bucket, key):
    try: s3.head_object(Bucket=bucket, Key=key); return True
    except ClientError: return False

def s3_list_keys(bucket, prefix, suffix=None):
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if suffix is None or k.lower().endswith(suffix.lower()):
                keys.append(k)
    return sorted(keys)

def s3_read_bytes(bucket, key):
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()

def s3_upload_npy(arr, bucket, key):
    buf = io.BytesIO(); np.save(buf, arr)
    s3.put_object(Bucket=bucket, Key=key,
                  Body=buf.getvalue(),
                  ContentType="application/octet-stream")

def s3_upload_csv(df, bucket, key):
    buf = io.BytesIO(df.to_csv(index=False).encode())
    s3.put_object(Bucket=bucket, Key=key,
                  Body=buf.getvalue(), ContentType="text/csv")

# ─── CHANNEL MAP LOADER ───────────────────────────────────────────────────────

def load_channel_map(bucket, map_key=None, n_channels=32):
    """
    Load channel map: {slot_name → 10-20_name or None}.
    Priority:
      1. S3 CSV if map_key provided (columns: slot, name)
      2. Default 32ch or 128ch map based on n_channels
    """
    if map_key and s3_exists(bucket, map_key):
        data = s3_read_bytes(bucket, map_key)
        df   = pd.read_csv(io.BytesIO(data))
        df.columns = df.columns.str.strip().str.lower()
        slot_col = next((c for c in df.columns if "slot" in c or "channel" in c
                         or c in ("c","col","id")), df.columns[0])
        name_col = next((c for c in df.columns if "name" in c or "label" in c
                         or "electrode" in c), df.columns[1])
        ch_map = {}
        for _, row in df.iterrows():
            slot = str(row[slot_col]).strip().upper()
            name = str(row[name_col]).strip()
            if not slot.startswith("C"):
                slot = f"C{int(slot):03d}"
            ch_map[slot] = None if name.upper() in ("","NONE","NA","N/A","-") else name
        log(f"  Channel map loaded from S3: {map_key}")
        log(f"  {sum(1 for v in ch_map.values() if v)} EEG channels mapped")
        return ch_map

    if n_channels >= 128:
        log(f"  Using default 148ch Xltek map (no map_key provided)")
        log(f"  ⚠ C001–C019 = standard 10-20, C020–C148 dropped")
        log(f"  ⚠ VERIFY against your cap documentation before using results")
        return DEFAULT_128CH_MAP
    else:
        log(f"  Using default 32ch Quantum map (no map_key provided)")
        log(f"  ⚠ VERIFY against your headbox documentation before using results")
        return DEFAULT_32CH_MAP

# ─── NICOLET TXT PARSER ───────────────────────────────────────────────────────

def parse_nicolet_txt(raw_bytes):
    """
    Parse Nicolet/Xltek .txt export from raw bytes.

    Handles all confirmed Nicolet NeuroWorks export variants:
      - Encoding : UTF-16 LE (BOM \\xff\\xfe), UTF-8+BOM, or plain UTF-8
      - Delimiter: TAB (\\t) — NOT spaces
      - Line ends : Old Mac \\r, Windows \\r\\n, or Unix \\n
      - Date+Time : ONE tab-delimited field e.g. "05/02/2026 12:36:20"
      - Columns   : [DateTime | EB | Stamp | C001 ... CXXX | TRIGGER/PHOTIC]
        → data starts at tab-index 3 (not 4 — Date+Time is a single field)
      - SHORT     : disconnected electrode token → zero-filled

    Returns:
        eeg_raw   : (n_ch, T) float32 in ORIGINAL units (mV)
        slot_names: list of slot names ['C001','C002',...]
        sfreq     : float (actual sampling rate from header, rounded to int)
        units     : str ('mV' or 'uV' or 'V')
        metadata  : dict of header info
    """
    # ── Encoding detection ────────────────────────────────────────────────────
    if raw_bytes[:2] in (b'\xff\xfe', b'\xfe\xff'):
        text = raw_bytes.decode("utf-16", errors="replace")
    elif raw_bytes[:3] == b'\xef\xbb\xbf':
        text = raw_bytes.decode("utf-8-sig", errors="replace")
    else:
        text = raw_bytes.decode("utf-8", errors="replace")

    # ── Line ending normalisation ─────────────────────────────────────────────
    # Old Mac \r, Windows \r\n, Unix \n → all to \n
    text  = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.splitlines()

    # ── Parse header ──────────────────────────────────────────────────────────
    sfreq      = None
    units      = "mV"      # Nicolet default
    n_ch       = None
    slot_names = None
    metadata   = {}

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("%"):
            break   # header block is done

        # Use TAB split for header values (tabs separate key from value)
        tab_parts = [p.strip() for p in stripped.lstrip("% ").split("\t")]
        content   = " ".join(tab_parts)   # for simple string searches

        # Sampling rate — "255.997750 Hz" or "256.000000 Hz"
        if "Sampling Rate" in content:
            for token in content.split():
                try:
                    val = float(token)
                    if 50 < val < 10000:   # sanity: must be a plausible Hz
                        sfreq = val; break
                except ValueError:
                    continue

        # Units — "mV" "uV" "μV" "V"
        if "Units" in content:
            for u in ["mV","uV","μV","V"]:
                if u in content:
                    units = u; break

        # Channel count — "32" or "148"
        if tab_parts[0].startswith("Channels") and "Format" not in content:
            for token in content.split():
                try:
                    n_ch = int(token); break
                except ValueError:
                    continue

        # Column header line:
        # "% Date.Time\t\tEB\tStamp\t  C001\t  C002\t ... TRIGGER"
        # Tab-split gives ['% Date.Time', '', 'EB', 'Stamp', '  C001', ...]
        if "Date.Time" in stripped:
            tab_cols   = [c.strip() for c in stripped.split("\t")]
            slot_names = [c for c in tab_cols
                          if c.startswith("C") and c[1:].isdigit()]
            # Record the column index where C001 starts
            data_col_start = None
            for ci, col in enumerate(tab_cols):
                if col.strip() == "C001":
                    data_col_start = ci
                    break
            metadata["col_header"]      = tab_cols
            metadata["data_col_start"]  = data_col_start

    if sfreq is None:
        sfreq = 256.0
        log("  [WARN] Sampling rate not found — defaulting to 256 Hz")

    # Round 255.9978 → 256 etc.
    sfreq_int = round(sfreq)
    if abs(sfreq - sfreq_int) < 1.0:
        sfreq = float(sfreq_int)

    # Column offset for data:
    # Tab columns: [DateTime(0) | EB(1) | Stamp(2) | C001(3) ... CXXX | TRIGGER]
    # DateTime is ONE field ("05/02/2026 12:36:20") — space inside, tab between cols
    data_col_start = metadata.get("data_col_start", 3)

    # ── Parse data lines ──────────────────────────────────────────────────────
    data_rows  = []
    n_expected = n_ch or 32   # expected number of channel columns

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%") or not stripped:
            continue

        # TAB split — gives exact column alignment
        tab_parts = stripped.split("\t")

        # Data line: first field contains a date (has "/" and a space for time)
        if not tab_parts:
            continue
        first = tab_parts[0].strip()
        if not ("/" in first and " " in first):
            continue

        # Extract channel values starting at data_col_start
        values = []
        for i in range(data_col_start, len(tab_parts)):
            raw_tok = tab_parts[i].strip()
            tok_up  = raw_tok.upper()

            # Stop at last-column markers
            if tok_up in ("TRIGGER","PHOTIC"):
                break

            # SHORT = disconnected electrode → zero
            if tok_up == "SHORT":
                values.append(0.0)
                continue

            # Event byte states in data — skip
            if tok_up in ("OFF","ON","") and i < data_col_start + 5:
                continue

            # Try parse as float
            try:
                values.append(float(raw_tok))
            except ValueError:
                if len(values) >= 10:
                    break   # hit trailing non-numeric → stop

        if len(values) >= 10:
            data_rows.append(values)

    if not data_rows:
        raise ValueError(
            f"Could not parse labels CSV.\n"
            f"Header: {header}\n"
            f"First non-header line: {first_line}"
        )
        # raise ValueError(
        #     "No data rows parsed.\n"
        #     f"  Encoding detected: {'utf-16' if raw_bytes[:2]==b'\\xff\\xfe' else 'utf-8'}\n"
        #     f"  Total lines: {len(lines)}\n"
        #     f"  Expected ~{n_expected} channels per row\n"
        #     f"  data_col_start={data_col_start}\n"
        #     f"  First non-header line: {next((l for l in lines if l.strip() and not l.strip().startswith('%')),'<none>')[:120]}")

    # Pad rows to same length
    n_cols = max(len(r) for r in data_rows)
    data   = np.zeros((len(data_rows), n_cols), dtype=np.float32)
    for i, row in enumerate(data_rows):
        data[i, :len(row)] = row

    eeg_raw = data.T    # (n_ch, T)

    if slot_names is None:
        n_actual   = eeg_raw.shape[0]
        slot_names = [f"C{i+1:03d}" for i in range(n_actual)]
        log(f"  [WARN] Slot names not found — inferred C001..C{n_actual:03d}")

    metadata.update({
        "sfreq":     sfreq,
        "units":     units,
        "n_ch_raw":  eeg_raw.shape[0],
        "n_samples": eeg_raw.shape[1],
    })

    return eeg_raw, slot_names, sfreq, units, metadata

# ─── UNIT CONVERSION ──────────────────────────────────────────────────────────

def to_volts(eeg, units):
    """Convert from file units to Volts (pipeline standard)."""
    if units == "mV":
        return (eeg / 1000.0).astype(np.float32), True
    elif units in ("uV", "μV", "microvolts"):
        return (eeg / 1e6).astype(np.float32), True
    elif units == "V":
        return eeg.astype(np.float32), False
    else:
        # Unknown — use amplitude heuristic
        std = eeg.std()
        if std > 100:
            log(f"  [WARN] Unknown units '{units}', std={std:.2f} — assuming μV")
            return (eeg / 1e6).astype(np.float32), True
        elif std > 0.1:
            log(f"  [WARN] Unknown units '{units}', std={std:.4f} — assuming mV")
            return (eeg / 1000.0).astype(np.float32), True
        else:
            log(f"  [WARN] Unknown units '{units}', std={std:.6f} — assuming V")
            return eeg.astype(np.float32), False

# ─── CHANNEL SELECTION ────────────────────────────────────────────────────────

def select_19ch(eeg_raw, slot_names, ch_map):
    """
    Map generic slot names (C001..C032) → 10-20 names via ch_map,
    then select and reorder to MASTER_19.

    Returns (19, T) float32.
    """
    # Build name → signal index
    name_to_idx = {}
    for i, slot in enumerate(slot_names):
        if i >= eeg_raw.shape[0]:
            break
        std_name = ch_map.get(slot.upper())   # e.g. C001 → Fp1
        if std_name is None:
            continue
        # Apply additional aliases
        std_name = CHANNEL_ALIASES.get(std_name, std_name)
        name_to_idx[std_name] = i

    missing = [ch for ch in MASTER_19 if ch not in name_to_idx]
    if len(missing) > 5:
        raise ValueError(
            f"Too many channels missing from map: {missing}\n"
            f"Mapped channels: {list(name_to_idx.keys())}\n"
            f"Check --channel_map_key or update DEFAULT_32CH_MAP"
        )

    rows = []
    for ch in MASTER_19:
        if ch in name_to_idx:
            rows.append(eeg_raw[name_to_idx[ch]])
        else:
            # Zero-fill truly missing channel with warning
            log(f"    [WARN] {ch} not in channel map — zero-filled")
            rows.append(np.zeros(eeg_raw.shape[1], dtype=np.float32))

    return np.vstack(rows).astype(np.float32)

# ─── INSPECT MODE ─────────────────────────────────────────────────────────────

def inspect_one(bucket, key):
    """Print full parsing info for one file without converting."""
    log(f"\nInspecting: s3://{bucket}/{key}")
    raw = s3_read_bytes(bucket, key)
    eeg_raw, slots, sfreq, units, meta = parse_nicolet_txt(raw)

    print(f"\n{'='*60}")
    print(f"File     : {Path(key).name}")
    print(f"sFreq    : {sfreq} Hz")
    print(f"Units    : {units}")
    print(f"Shape    : {eeg_raw.shape}  (channels × samples)")
    print(f"Duration : {eeg_raw.shape[1]/sfreq:.1f}s")
    print(f"Slots    : {slots[:8]}{'...' if len(slots)>8 else ''}")
    print(f"Amplitude: min={eeg_raw.min():.4f}  max={eeg_raw.max():.4f}  "
          f"std={eeg_raw.std():.4f} {units}")
    eeg_v, _ = to_volts(eeg_raw, units)
    print(f"In Volts : std={eeg_v.std():.6f} V "
          f"= {eeg_v.std()*1e6:.1f} μV "
          f"({'✓ normal' if 5e-6 < eeg_v.std() < 500e-6 else '✗ check units'})")
    print(f"\nSlot → default 10-20 map ({len(eeg_raw)} channels):")
    ch_map = load_channel_map(bucket, None, len(eeg_raw))
    for slot in slots[:20]:
        name = ch_map.get(slot.upper(), "NOT MAPPED")
        print(f"  {slot} → {name}")
    if len(slots) > 20:
        print(f"  ... and {len(slots)-20} more")
    print(f"\nChannels that WOULD be selected:")
    mapped = {ch_map.get(s.upper()): i for i,s in enumerate(slots)
              if ch_map.get(s.upper()) in MASTER_19}
    found   = [ch for ch in MASTER_19 if ch in mapped]
    missing = [ch for ch in MASTER_19 if ch not in mapped]
    print(f"  Found  ({len(found)}): {found}")
    print(f"  Missing({len(missing)}): {missing}")

# ─── SINGLE FILE CONVERSION ───────────────────────────────────────────────────

def convert_one(bucket, src_key, out_key, label,
                ch_map, dataset, dry_run=False):
    """Download, parse, select 19ch, upload .npy. Returns manifest row."""
    stem = Path(src_key).stem

    if s3_exists(bucket, out_key):
        log(f"  [SKIP] {stem} exists")
        return {"subject_id": stem, "stem": stem, "dataset": dataset,
                "true_label": label, "clean_npy": out_key, "status": "existing"}

    if dry_run:
        log(f"  [DRY] {stem}  label={label}")
        return None

    raw = s3_read_bytes(bucket, src_key)
    eeg_raw, slots, sfreq, units, meta = parse_nicolet_txt(raw)

    log(f"  {stem}  {eeg_raw.shape[0]}ch  {eeg_raw.shape[1]/sfreq:.0f}s  "
        f"{sfreq:.0f}Hz  {units}  label={label}")

    # Unit conversion
    eeg_v, converted = to_volts(eeg_raw, units)

    # Sanity check
    std_uv = eeg_v.std() * 1e6
    if not (2 < std_uv < 1000):
        log(f"    [WARN] Amplitude {std_uv:.1f}μV — check units/conversion")

    # Channel selection
    eeg_19 = select_19ch(eeg_v, slots, ch_map)

    duration = eeg_19.shape[1] / sfreq
    if duration < 10:
        raise ValueError(f"Too short: {duration:.1f}s")

    s3_upload_npy(eeg_19, bucket, out_key)
    log(f"    ✓ → {eeg_19.shape}  {duration:.0f}s  "
        f"{'mV→V' if converted else 'V'}  "
        f"std={eeg_19.std()*1e6:.1f}μV")

    return {
        "subject_id":     stem,
        "stem":           stem,
        "dataset":        dataset,
        "true_label":     label,
        "original_sfreq": sfreq,
        "original_units": units,
        "duration_sec":   round(duration, 1),
        "n_channels_raw": eeg_raw.shape[0],
        "shape":          str(eeg_19.shape),
        "unit_converted": converted,
        "clean_npy":      out_key,
        "status":         "converted",
    }

# ─── BATCH INGESTION ──────────────────────────────────────────────────────────

def run_ingest(bucket, src_prefix, dataset, labels_csv_key,
               channel_map_key=None, dry_run=False):
    """Convert all .txt files in src_prefix."""
    out_pfx = f"{OUT_ROOT}/{dataset}"
    log(f"{'='*60}")
    log(f"Nicolet TXT Ingestion: {dataset}")
    log(f"Source  : s3://{bucket}/{src_prefix}")
    log(f"Output  : s3://{bucket}/{out_pfx}/")

    # Load labels
    label_lookup = {}
    if labels_csv_key and s3_exists(bucket, labels_csv_key):
        ldf = pd.read_csv(io.BytesIO(s3_read_bytes(bucket, labels_csv_key)))
        ldf.columns = ldf.columns.str.strip().str.lower()
        id_col  = next((c for c in ldf.columns
                        if "id" in c or "subject" in c or "file" in c),
                       ldf.columns[0])
        dx_col  = next((c for c in ldf.columns
                        if "label" in c or "diag" in c or "dx" in c
                        or "class" in c or "group" in c), None)
        if dx_col:
            for _, row in ldf.iterrows():
                sid = Path(str(row[id_col])).stem.strip()
                dx  = str(row[dx_col]).strip().lower()
                label_lookup[sid] = LABEL_MAP.get(dx, None)
        log(f"Labels  : {len(label_lookup)} entries")
    else:
        log("[WARN] No labels CSV — all files will have true_label=None")

    # Discover txt files
    txt_keys = s3_list_keys(bucket, src_prefix, suffix=".txt")
    log(f"Files   : {len(txt_keys)} .txt files found")

    if not txt_keys:
        log("[ERROR] No .txt files found"); return []

    # Detect channel count from first file to pick right default map
    try:
        sample = s3_read_bytes(bucket, txt_keys[0])
        _, _, sfreq, _, meta = parse_nicolet_txt(sample)
        n_ch_detected = meta.get("n_ch_raw", 32)
    except Exception:
        n_ch_detected = 32

    # Load channel map
    ch_map = load_channel_map(bucket, channel_map_key, n_ch_detected)

    rows = []
    n_ok = n_skip = n_err = 0

    for i, key in enumerate(txt_keys):
        stem    = Path(key).stem
        label   = label_lookup.get(stem, label_lookup.get(stem.lower(), None))
        out_key = f"{out_pfx}/{stem}.npy"

        log(f"\n[{i+1}/{len(txt_keys)}] {Path(key).name}  label={label}")

        try:
            row = convert_one(bucket, key, out_key, label,
                              ch_map, dataset, dry_run)
            if row:
                rows.append(row)
                if row.get("status") == "existing": n_skip += 1
                else: n_ok += 1
        except Exception as e:
            log(f"  [ERROR] {stem}: {e}")
            traceback.print_exc()
            n_err += 1

    # Save manifest
    if rows and not dry_run:
        df   = pd.DataFrame(rows)
        mkey = f"{out_pfx}/manifest_{dataset}.csv"
        s3_upload_csv(df, bucket, mkey)
        log(f"\n{'='*60}")
        log(f"Manifest → s3://{bucket}/{mkey}")
        log(f"  Converted : {n_ok}")
        log(f"  Existing  : {n_skip}")
        log(f"  Errors    : {n_err}")
        if "true_label" in df.columns:
            log(f"  label=0   : {int((df.true_label==0).sum())}")
            log(f"  label=1   : {int((df.true_label==1).sum())}")
        log(f"\nNext: python data_pipeline.py --dataset {dataset} "
            f"--raw_prefix {out_pfx}/ --mode train")

    elif dry_run:
        log(f"\nDry run: {len(txt_keys)} files would be processed")

    return rows

# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Ingest Nicolet/Xltek .txt EEG exports to S3 .npy"
    )
    sub = p.add_subparsers(dest="mode")

    # Inspect
    pi = sub.add_parser("inspect", help="Print file structure for one .txt file")
    pi.add_argument("--bucket", default=BUCKET)
    pi.add_argument("--key", required=True, help="S3 key of one .txt file")

    # Ingest
    pb = sub.add_parser("ingest", help="Convert all .txt files in a prefix")
    pb.add_argument("--bucket",          default=BUCKET)
    pb.add_argument("--src_prefix",      required=True,
                    help="S3 prefix containing .txt files")
    pb.add_argument("--dataset",         required=True,
                    help="Dataset name for output folder")
    pb.add_argument("--labels_csv",      default=None,
                    help="S3 key of labels CSV (columns: subject_id, diagnosis)")
    pb.add_argument("--channel_map_key", default=None,
                    help="S3 key of channel map CSV (columns: slot, name) "
                         "e.g. C001,Fp1. If omitted, uses default 32ch map")
    pb.add_argument("--dry_run",         action="store_true")

    args = p.parse_args()

    if args.mode == "inspect":
        inspect_one(args.bucket, args.key)

    elif args.mode == "ingest":
        run_ingest(
            bucket          = args.bucket,
            src_prefix      = args.src_prefix,
            dataset         = args.dataset,
            labels_csv_key  = args.labels_csv,
            channel_map_key = args.channel_map_key,
            dry_run         = args.dry_run,
        )
    else:
        p.print_help()