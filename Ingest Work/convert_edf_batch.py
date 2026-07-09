"""
Batch EDF Converter — Bipolar / Cz-referenced → Standard 19-ch Referential
Handles two montage types found in clinical EDF files:
  Type A (Bipolar):    Fp1-F3, F3-C3, ... (Girish_babu style)
  Type B (Cz-ref):     Fp1-Cz, F3-Cz, ... (Jaykant style)

Output: 19-channel EDF resampled to TARGET_SFREQ
Usage:  python convert_edf_batch.py --input_dir /path/to/edfs
                                     --output_dir /path/to/output
                                     --sfreq 256
"""

import os
import argparse
import numpy as np
import mne
from pathlib import Path

# ── Standard 19 channels (10-20 system, your pipeline expects these) ─────────
STANDARD_19 = ['Fp1','Fp2','F7','F3','Fz','F4','F8',
                'T3','C3','Cz','C4','T4',
                'T5','P3','Pz','P4','T6',
                'O1','O2']

TARGET_SFREQ = 256   # change to match your model's sfreq

# ── Bipolar chains → reconstruct referential vs O1/O2/Pz as reference ────────
# Each tuple: (bipolar_key, anode, cathode)
BIPOLAR_CHAINS = [
    # Left parasagittal
    ('EEG Fp1-F3', 'Fp1', 'F3'),
    ('EEG F3-C3',  'F3',  'C3'),
    ('EEG C3-P3',  'C3',  'P3'),
    ('EEG P3-O1',  'P3',  'O1'),
    # Right parasagittal
    ('EEG Fp2-F4', 'Fp2', 'F4'),
    ('EEG F4-C4',  'F4',  'C4'),
    ('EEG C4-P4',  'C4',  'P4'),
    ('EEG P4-O2',  'P4',  'O2'),
    # Left temporal
    ('EEG Fp1-F7', 'Fp1', 'F7'),
    ('EEG F7-T3',  'F7',  'T3'),
    ('EEG T3-T5',  'T3',  'T5'),
    ('EEG T5-O1',  'T5',  'O1'),
    # Right temporal
    ('EEG Fp2-F8', 'Fp2', 'F8'),
    ('EEG F8-T4',  'F8',  'T4'),
    ('EEG T4-T6',  'T4',  'T6'),
    ('EEG T6-O2',  'T6',  'O2'),
    # Midline
    ('EEG Fz-Cz',  'Fz',  'Cz'),
    ('EEG Cz-Pz',  'Cz',  'Pz'),
]


def detect_montage(ch_names):
    """Returns 'bipolar', 'czref', or 'unknown'"""
    eeg_chs = [c for c in ch_names if c.startswith('EEG')]
    if not eeg_chs:
        return 'unknown'
    # Cz-ref: all EEG channels end in -Cz
    if all('-Cz' in c for c in eeg_chs):
        return 'czref'
    # Bipolar: channels like Fp1-F3, F3-C3 (second part is NOT Cz)
    if any(c for c in eeg_chs if '-' in c and not c.endswith('-Cz')):
        return 'bipolar'
    return 'unknown'


def bipolar_to_referential(raw):
    """
    Reconstruct referential (avg/ear-ref) from bipolar chains.
    Strategy: integrate each chain from the last electrode (O1/O2/Pz)
    assumed = 0 (linked-ear approximation). This is an approximation.
    Returns dict: {ch_name: signal_array}
    """
    data_dict = {ch: raw.get_data(picks=ch)[0] for ch in raw.ch_names}
    n = raw.get_data().shape[1]
    ref = np.zeros(n)  # O1, O2, Pz assumed ~0 (reference anchor)

    signals = {}

    # ── Left parasagittal chain (anchor: O1=0) ────────────────────────────
    signals['O1']  = ref.copy()
    signals['P3']  = signals['O1']  + data_dict.get('EEG P3-O1',  ref)
    signals['C3']  = signals['P3']  + data_dict.get('EEG C3-P3',  ref)
    signals['F3']  = signals['C3']  + data_dict.get('EEG F3-C3',  ref)
    signals['Fp1'] = signals['F3']  + data_dict.get('EEG Fp1-F3', ref)

    # ── Right parasagittal chain (anchor: O2=0) ───────────────────────────
    signals['O2']  = ref.copy()
    signals['P4']  = signals['O2']  + data_dict.get('EEG P4-O2',  ref)
    signals['C4']  = signals['P4']  + data_dict.get('EEG C4-P4',  ref)
    signals['F4']  = signals['C4']  + data_dict.get('EEG F4-C4',  ref)
    signals['Fp2'] = signals['F4']  + data_dict.get('EEG Fp2-F4', ref)

    # ── Left temporal chain (anchor: O1 already set) ──────────────────────
    signals['T5']  = signals['O1']  + data_dict.get('EEG T5-O1',  ref)
    signals['T3']  = signals['T5']  + data_dict.get('EEG T3-T5',  ref)
    signals['F7']  = signals['T3']  + data_dict.get('EEG F7-T3',  ref)
    # Fp1 already set; consistency check possible here

    # ── Right temporal chain (anchor: O2 already set) ─────────────────────
    signals['T6']  = signals['O2']  + data_dict.get('EEG T6-O2',  ref)
    signals['T4']  = signals['T6']  + data_dict.get('EEG T4-T6',  ref)
    signals['F8']  = signals['T4']  + data_dict.get('EEG F8-T4',  ref)

    # ── Midline chain (anchor: Pz=0) ─────────────────────────────────────
    signals['Pz']  = ref.copy()
    signals['Cz']  = signals['Pz']  + data_dict.get('EEG Cz-Pz',  ref)
    signals['Fz']  = signals['Cz']  + data_dict.get('EEG Fz-Cz',  ref)

    return signals


def czref_to_referential(raw):
    """
    Convert Cz-referenced montage to standard referential.
    Channels are already X-Cz, so just rename and keep.
    Cz itself = 0 (it's the reference). Fz, Pz = interpolate or zero.
    Returns dict: {ch_name: signal_array}
    """
    signals = {}
    for ch in raw.ch_names:
        if not ch.startswith('EEG'):
            continue
        # Skip mastoids and ECG
        label = ch.replace('EEG ', '').replace('-Cz', '').strip()
        if label in ('A1', 'A2'):
            continue
        signals[label] = raw.get_data(picks=ch)[0]

    n = raw.get_data().shape[1]

    # Cz is the reference → signal is 0 relative to itself
    signals['Cz'] = np.zeros(n)

    # Fz and Pz not recorded → interpolate from neighbours or zero
    # Simple average interpolation:
    if 'Fz' not in signals:
        candidates = [signals[c] for c in ['F3', 'F4', 'Cz'] if c in signals]
        signals['Fz'] = np.mean(candidates, axis=0) if candidates else np.zeros(n)

    if 'Pz' not in signals:
        candidates = [signals[c] for c in ['P3', 'P4', 'Cz'] if c in signals]
        signals['Pz'] = np.mean(candidates, axis=0) if candidates else np.zeros(n)

    return signals


def build_mne_raw(signals, sfreq, target_sfreq):
    """Build MNE RawArray with standard 19 channels, resampled."""
    n_samples = next(iter(signals.values())).shape[0]
    data = np.zeros((19, n_samples))

    for i, ch in enumerate(STANDARD_19):
        if ch in signals:
            data[i] = signals[ch]
        else:
            print(f"  ⚠️  Missing channel {ch} — filling with zeros")

    info = mne.create_info(
        ch_names=STANDARD_19,
        sfreq=sfreq,
        ch_types='eeg'
    )
    raw_out = mne.io.RawArray(data, info, verbose=False)

    if sfreq != target_sfreq:
        raw_out.resample(target_sfreq, verbose=False)

    return raw_out


#======GUL=========================================================
def convert_raw_edf(
    edf_path,
    target_sfreq=TARGET_SFREQ,
):
    """
    Read an EDF and return a standardized
    19-channel MNE Raw object.
    """

    raw = mne.io.read_raw_edf(
        edf_path,
        preload=True,
        verbose=False,
    )

    sfreq = raw.info["sfreq"]

    montage = detect_montage(raw.ch_names)

    if montage == "bipolar":
        signals = bipolar_to_referential(raw)

    elif montage == "czref":
        signals = czref_to_referential(raw)

    else:
        raise ValueError(
            f"Unknown montage: {montage}"
        )

    raw_out = build_mne_raw(
        signals,
        sfreq,
        target_sfreq,
    )

    return raw_out
#===============================================================


def convert_file(edf_path, output_dir, target_sfreq=TARGET_SFREQ):
    fname = Path(edf_path).stem
    out_path = Path(output_dir) / f"{fname}_19ch.edf"

    print(f"\n📂 Processing: {Path(edf_path).name}")

    try:
        raw_out = convert_raw_edf(
            edf_path,
            target_sfreq,
        )
    except Exception as e:
        print(f"Conversion failed: {e}")
        return False

    try:
        raw_out.export(
            str(out_path),
            fmt="edf",
            overwrite=True,
            verbose=False,
        )
        print(f"  ✅ Saved → {out_path.name}  [{len(STANDARD_19)} ch @ {target_sfreq} Hz]")
        return True

    except Exception as e:
        print(f"  ❌ Export failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch EDF montage converter')
    parser.add_argument('--input_dir',  required=True, help='Folder containing .edf files')
    parser.add_argument('--output_dir', required=True, help='Folder to save converted files')
    parser.add_argument('--sfreq', type=float, default=TARGET_SFREQ,
                        help=f'Target sampling rate (default: {TARGET_SFREQ} Hz)')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    edf_files = sorted(Path(args.input_dir).glob('*.edf'))

    if not edf_files:
        print("No .edf files found in input directory.")
        return

    print(f"\n{'='*60}")
    print(f"Found {len(edf_files)} EDF files | Target: {int(args.sfreq)} Hz, 19-ch")
    print(f"{'='*60}")

    ok, fail = 0, 0
    for f in edf_files:
        if convert_file(str(f), args.output_dir, args.sfreq):
            ok += 1
        else:
            fail += 1

    print(f"\n{'='*60}")
    print(f"Done — ✅ {ok} converted | ❌ {fail} failed")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()