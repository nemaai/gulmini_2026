import re

import numpy as np
from scipy import signal
from scipy.signal import welch

from core.sleep_marker_detection import normalize_edf_metadata

# =========================
# TARGET EEG CHANNELS (MODEL CONSISTENT)
# =========================
STANDARD_1020_CHANNELS = [

    "FP1","FP2",

    "F7","F3","FZ","F4","F8",

    "T3","T4","T5","T6",

    "T7","T8","P7","P8",

    "C3","CZ","C4",

    "P3","PZ","P4",

    "O1","O2"
]

HEADSET_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7",
    "P7", "O1", "O2", "P8", "T8",
    "FC6", "F4", "F8", "AF4"
]

TARGET_CHANNELS = list(dict.fromkeys(STANDARD_1020_CHANNELS + HEADSET_CHANNELS))


# =========================
# CLEAN CHANNEL NAME
# =========================
def clean_channel_name(ch):
    ch = str(ch).upper()

    ch = ch.replace("EEG ", "")
    ch = ch.replace("-REF", "")
    ch = ch.replace("-LE", "")
    ch = ch.replace("-A1", "")
    ch = ch.replace("-A2", "")

    ch = re.sub(r"[^A-Z0-9]", "", ch)

    return ch


# =========================
# FILTER EEG CHANNELS
# =========================
def filter_eeg_channels(eeg, ch_names=None):
    """
    eeg: (channels, samples)
    ch_names: optional list of channel names
    """

    # If no names passed, assume already filtered
    if ch_names is None:
        return eeg, list(range(eeg.shape[0]))

    cleaned = [clean_channel_name(c) for c in ch_names]

    selected_idx = []
    for i, ch in enumerate(cleaned):
        if i < eeg.shape[0] and ch in TARGET_CHANNELS:
            selected_idx.append(i)

    if len(selected_idx) == 0:
        return None, []

    eeg_filtered = eeg[selected_idx]

    return eeg_filtered, selected_idx


# =========================
# ARTIFACT HELPERS
# =========================
def _safe_quality_result():
    return {
        "overall_quality": 0,
        "bad_channels": [],
        "bad_channel_names": [],
        "muscle_artifacts": "High",
        "eye_blinks": "High",
        "line_noise": "High",
        "movement_artifacts": "High",
        "quality_details": {
            "reason": "invalid_or_insufficient_data",
            "selected_channels": [],
            "penalties": {
                "bad_channels": 0.0,
                "missing_channels": 0.0,
                "muscle_artifacts": 0.0,
                "eye_blinks": 0.0,
                "line_noise": 0.0,
                "movement_artifacts": 0.0,
            },
            "scores": {},
        },
    }


def _band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(psd[mask], freqs[mask]))
    return float(np.trapz(psd[mask], freqs[mask]))


def _mean_band_power(freqs, psd, low, high):
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return 0.0
    return float(np.mean(psd[mask]))


def _severity_from_score(score, moderate_at=0.20, high_at=0.50):
    if score >= high_at:
        return "High"
    if score >= moderate_at:
        return "Moderate"
    return "Low"


def _clip01(value):
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _scale01(value, low, high):
    if high <= low:
        return 1.0 if value >= high else 0.0
    return _clip01((float(value) - low) / (high - low))


def _robust_zscore(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=float)

    median = float(np.nanmedian(values[finite]))
    mad = float(np.nanmedian(np.abs(values[finite] - median)))
    if mad < 1e-12:
        return np.zeros_like(values, dtype=float)

    return 0.6745 * (values - median) / mad


def _round_metric(value, digits=4):
    if isinstance(value, (np.floating, float)):
        return round(float(value), digits)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _channel_names_from_indices(channel_names, indices):
    resolved = []
    for index in indices:
        try:
            idx = int(index)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(channel_names):
            resolved.append(str(channel_names[idx]))
    return resolved


def _clean_signal(epoch):
    epoch = np.asarray(epoch, dtype=float)
    if epoch.ndim != 1:
        epoch = np.ravel(epoch)
    return np.nan_to_num(epoch, nan=0.0, posinf=0.0, neginf=0.0)


def _welch(epoch, Fs):
    epoch = _clean_signal(epoch)
    if len(epoch) < 4:
        return np.array([]), np.array([])

    nperseg = min(len(epoch), max(4, int(Fs * 2)))
    return welch(epoch, Fs, nperseg=nperseg)


def _to_microvolts(eeg):
    eeg = np.asarray(eeg, dtype=float)
    scale = np.nanpercentile(np.abs(eeg), 95) if eeg.size else 0.0
    if np.isfinite(scale) and scale < 1.0:
        return eeg * 1e6
    return eeg


def _quality_bandpass(eeg_uv, sfreq, low=1.0, high=45.0):
    eeg_uv = np.asarray(eeg_uv, dtype=float)
    if eeg_uv.ndim != 2 or eeg_uv.shape[1] < 8:
        return signal.detrend(eeg_uv, axis=1, type="constant")

    nyq = sfreq / 2.0
    high = min(high, nyq * 0.95)
    if low <= 0 or high <= low:
        return signal.detrend(eeg_uv, axis=1, type="constant")

    try:
        sos = signal.butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
        return signal.sosfiltfilt(sos, eeg_uv, axis=1)
    except ValueError:
        return signal.detrend(eeg_uv, axis=1, type="constant")


def _windowed_segments(eeg, sfreq, window_sec=2.0):
    win = max(4, int(window_sec * sfreq))
    n_windows = eeg.shape[1] // win
    if n_windows == 0:
        return eeg[:, np.newaxis, :]

    trimmed = eeg[:, :n_windows * win]
    return trimmed.reshape(eeg.shape[0], n_windows, win)


def _channel_correlation_quality(segments, correlation_threshold=0.4):
    n_channels, n_windows, _ = segments.shape
    if n_channels < 3 or n_windows == 0:
        return {
            "median_max_correlation": np.ones(n_channels, dtype=float),
            "low_correlation_fraction": np.zeros(n_channels, dtype=float),
            "dropout_fraction": np.zeros(n_channels, dtype=float),
        }

    max_corr_windows = np.zeros((n_channels, n_windows), dtype=float)
    active_windows = np.ones((n_channels, n_windows), dtype=bool)

    for window_idx in range(n_windows):
        window = np.asarray(segments[:, window_idx, :], dtype=float)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        centered = window - np.mean(window, axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        active = norms > 1e-6
        active_windows[:, window_idx] = active

        if int(np.sum(active)) < 2:
            continue

        normalized = centered[active] / norms[active, np.newaxis]
        corr = np.abs(normalized @ normalized.T)
        np.fill_diagonal(corr, 0.0)

        active_idx = np.where(active)[0]
        max_corr_windows[active_idx, window_idx] = np.max(corr, axis=1)

    return {
        "median_max_correlation": np.nanmedian(max_corr_windows, axis=1),
        "low_correlation_fraction": np.mean(max_corr_windows < correlation_threshold, axis=1),
        "dropout_fraction": np.mean(~active_windows, axis=1),
    }


def _assess_channel_quality(eeg, sfreq):
    eeg_uv = _to_microvolts(eeg)
    eeg_band = _quality_bandpass(eeg_uv, sfreq, low=1.0, high=45.0)
    segments = _windowed_segments(eeg_band, sfreq, window_sec=2.0)

    epoch_std = np.nanstd(segments, axis=2)
    epoch_ptp = np.ptp(segments, axis=2)
    median_std = np.nanmedian(epoch_std, axis=1)
    median_ptp = np.nanmedian(epoch_ptp, axis=1)
    corr = _channel_correlation_quality(segments, correlation_threshold=0.4)

    positive_std = median_std[np.isfinite(median_std) & (median_std > 0)]
    if len(positive_std) == 0:
        all_channels = list(range(eeg.shape[0]))
        return {
            "bad_channels": all_channels,
            "flat_channels": all_channels,
            "noisy_channels": [],
            "low_correlation_channels": [],
            "dropout_channels": all_channels,
            "channel_scores": [1.0] * eeg.shape[0],
            "bad_channel_score": 1.0,
            "median_std_uv": [0.0] * eeg.shape[0],
            "median_ptp_uv": [0.0] * eeg.shape[0],
            "amplitude_z": [0.0] * eeg.shape[0],
            "low_correlation_fraction": [1.0] * eeg.shape[0],
            "dropout_fraction": [1.0] * eeg.shape[0],
            "median_max_correlation": [0.0] * eeg.shape[0],
        }

    global_median_std = float(np.median(positive_std))
    global_median_ptp = float(np.median(median_ptp[np.isfinite(median_ptp)]))
    flat_threshold = max(1.0, 0.05 * global_median_std)

    log_std = np.log10(np.maximum(median_std, 1e-9))
    robust_z = _robust_zscore(log_std)
    std_ratio = median_std / (global_median_std + 1e-9)
    ptp_ratio = median_ptp / (global_median_ptp + 1e-9)

    flat_channels = [
        i for i, std_uv in enumerate(median_std)
        if std_uv < flat_threshold
    ]
    dropout_channels = [
        i for i, fraction in enumerate(corr["dropout_fraction"])
        if fraction > 0.01
    ]
    noisy_channels = [
        i for i, (std_uv, ptp_uv, z) in enumerate(zip(median_std, median_ptp, robust_z))
        if (
            z > 4.5
            and std_uv > max(150.0, 4.0 * global_median_std)
            and ptp_uv > max(800.0, 4.0 * global_median_ptp)
        )
    ]
    low_correlation_channels = [
        i for i, fraction in enumerate(corr["low_correlation_fraction"])
        if fraction >= 0.35
    ]

    channel_scores = []
    for i in range(eeg.shape[0]):
        flat_score = _scale01(flat_threshold - median_std[i], 0.0, flat_threshold)
        noisy_score = max(
            _scale01(robust_z[i], 2.5, 5.0),
            _scale01(std_ratio[i], 2.0, 5.0) * _scale01(ptp_ratio[i], 2.0, 5.0),
        )
        dropout_score = _scale01(corr["dropout_fraction"][i], 0.005, 0.05)
        correlation_score = _scale01(corr["low_correlation_fraction"][i], 0.10, 0.40)
        channel_scores.append(max(flat_score, noisy_score, dropout_score, correlation_score))

    bad_channels = sorted(
        set(flat_channels + dropout_channels + noisy_channels + low_correlation_channels)
    )

    return {
        "bad_channels": bad_channels,
        "flat_channels": flat_channels,
        "noisy_channels": noisy_channels,
        "low_correlation_channels": low_correlation_channels,
        "dropout_channels": dropout_channels,
        "channel_scores": [float(_clip01(score)) for score in channel_scores],
        "bad_channel_score": float(np.mean(channel_scores)) if channel_scores else 0.0,
        "median_std_uv": [float(v) for v in median_std],
        "median_ptp_uv": [float(v) for v in median_ptp],
        "amplitude_z": [float(v) for v in robust_z],
        "low_correlation_fraction": [float(v) for v in corr["low_correlation_fraction"]],
        "dropout_fraction": [float(v) for v in corr["dropout_fraction"]],
        "median_max_correlation": [float(v) for v in corr["median_max_correlation"]],
    }


def _detect_bad_channels_robust(eeg, sfreq):
    channel_quality = _assess_channel_quality(eeg, sfreq)
    return (
        channel_quality["bad_channels"],
        channel_quality["flat_channels"],
        channel_quality["noisy_channels"],
    )


def _detect_movement_windows(eeg, sfreq):
    eeg_uv = _to_microvolts(eeg)
    eeg_band = _quality_bandpass(eeg_uv, sfreq, low=0.5, high=45.0)
    segments = _windowed_segments(eeg_band, sfreq, window_sec=2.0)
    if segments.shape[1] < 3:
        return False, 0.0, 0.0

    window_ptp = np.nanmedian(np.ptp(segments, axis=2), axis=0)
    baseline = float(np.nanmedian(window_ptp))
    mad = float(np.nanmedian(np.abs(window_ptp - baseline))) + 1e-9
    z_scores = 0.6745 * (window_ptp - baseline) / mad
    burst_windows = (z_scores > 8.0) & (window_ptp > max(500.0, 3.0 * baseline))
    burst_fraction = float(np.mean(burst_windows))
    max_z = float(np.nanmax(z_scores)) if len(z_scores) else 0.0

    is_movement = burst_fraction >= 0.05 or (np.sum(burst_windows) >= 5 and burst_fraction >= 0.02)
    return bool(is_movement), max_z, burst_fraction


def _expected_channel_count(selected_channel_names):
    if not selected_channel_names:
        return None

    cleaned = [clean_channel_name(ch) for ch in selected_channel_names]
    standard_hits = sum(ch in STANDARD_1020_CHANNELS for ch in cleaned)
    headset_hits = sum(ch in HEADSET_CHANNELS for ch in cleaned)

    if standard_hits >= headset_hits and standard_hits >= 4:
        return len(STANDARD_1020_CHANNELS)
    if headset_hits >= 4:
        return len(HEADSET_CHANNELS)
    return len(selected_channel_names)


def detect_emg_artifact(epoch, Fs):
    """epoch: (n_samples,) single channel"""
    epoch = _clean_signal(epoch)
    freqs, psd = _welch(epoch, Fs)
    if len(freqs) == 0:
        return False, 0.0, 0.0, 0.0

    total_power = _band_power(freqs, psd, 1, 70)
    emg_power = _band_power(freqs, psd, 30, 70)
    emg_ratio = emg_power / (total_power + 1e-10)

    # Hjorth complexity adds shape information beyond high-frequency power.
    dx = np.diff(epoch)
    ddx = np.diff(dx)
    if len(dx) == 0 or len(ddx) == 0:
        return False, 0.0, float(emg_ratio), 0.0

    mob = np.sqrt(np.var(dx) / (np.var(epoch) + 1e-10))
    mob2 = np.sqrt(np.var(ddx) / (np.var(dx) + 1e-10))
    complexity = mob2 / (mob + 1e-10)

    is_emg = (emg_ratio > 0.40) and (complexity > 5.0)
    confidence = min(1.0, (emg_ratio / 0.4) * 0.5 + (complexity / 5.0) * 0.5)
    return bool(is_emg), float(confidence), float(emg_ratio), float(complexity)


def _blink_threshold(epoch_fp1, epoch_fp2, threshold_uV):
    if threshold_uV != 150e-6:
        return threshold_uV

    # The snippet assumes Volts. Many EEG exports are already in microvolts,
    # so keep the same default but adapt it to the apparent data scale.
    combined = np.concatenate([np.abs(epoch_fp1), np.abs(epoch_fp2)])
    scale = np.nanpercentile(combined, 95) if combined.size else 0.0
    return 150.0 if scale > 1.0 else 150e-6


def detect_eye_blink(epoch_fp1, epoch_fp2, Fs, threshold_uV=150e-6):
    """epoch_fp1/fp2: (n_samples,) in Volts or microvolts"""
    epoch_fp1 = _clean_signal(epoch_fp1)
    epoch_fp2 = _clean_signal(epoch_fp2)
    n = min(len(epoch_fp1), len(epoch_fp2))
    if n < max(8, int(0.25 * Fs)):
        return False, 0, 0.0

    epoch_fp1 = epoch_fp1[:n]
    epoch_fp2 = epoch_fp2[:n]

    nyq = Fs / 2
    low = 0.5 / nyq
    high = min(5.0 / nyq, 0.99)
    if low <= 0 or high <= low:
        return False, 0, 0.0

    b, a = signal.butter(4, [low, high], btype="band")
    try:
        fp1_filt = signal.filtfilt(b, a, epoch_fp1)
        fp2_filt = signal.filtfilt(b, a, epoch_fp2)
    except ValueError:
        fp1_filt = signal.detrend(epoch_fp1)
        fp2_filt = signal.detrend(epoch_fp2)

    win = max(2, int(0.2 * Fs))

    def pp_series(sig):
        if len(sig) <= win:
            return np.array([np.ptp(sig)])
        out = np.zeros(len(sig) - win)
        for i in range(len(out)):
            out[i] = np.max(sig[i:i + win]) - np.min(sig[i:i + win])
        return out

    pp1 = pp_series(fp1_filt)
    pp2 = pp_series(fp2_filt)
    threshold = _blink_threshold(fp1_filt, fp2_filt, threshold_uV)
    candidates = (pp1 > threshold) | (pp2 > threshold)

    if np.std(fp1_filt) < 1e-12 or np.std(fp2_filt) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(fp1_filt, fp2_filt)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0

    blink_count = int(np.sum(np.diff(candidates.astype(int)) == 1))
    if bool(candidates[0]):
        blink_count += 1

    is_blink_epoch = (blink_count > 0) and (corr > 0.70)
    return bool(is_blink_epoch), blink_count, corr


def detect_movement_artifact(epoch_data, baseline_vars):
    """
    epoch_data: (n_channels, n_samples)
    baseline_vars: median variance per channel from clean reference epochs
    """
    epoch_data = np.asarray(epoch_data, dtype=float)
    baseline_vars = np.asarray(baseline_vars, dtype=float)
    epoch_vars = np.nanvar(epoch_data, axis=1)

    if baseline_vars.ndim == 0:
        baseline_vars = np.full(epoch_vars.shape, float(baseline_vars))
    elif len(baseline_vars) != len(epoch_vars):
        baseline_vars = np.resize(baseline_vars, epoch_vars.shape)

    mad = np.median(np.abs(epoch_vars - np.median(epoch_vars)))
    z_scores = np.abs(epoch_vars - baseline_vars) / (mad + 1e-10)
    is_movement = np.any(z_scores > 5.0)
    max_z = float(np.max(z_scores)) if len(z_scores) else 0.0
    return bool(is_movement), max_z, z_scores


def remove_ecg_artifact_from_eeg(eeg_channel, ecg_channel, Fs):
    """
    eeg_channel: (n_samples,)
    ecg_channel: (n_samples,) - dedicated ECG lead
    """
    eeg_channel = _clean_signal(eeg_channel)
    ecg_channel = _clean_signal(ecg_channel)
    n = min(len(eeg_channel), len(ecg_channel))
    if n == 0:
        return eeg_channel

    eeg_channel = eeg_channel[:n]
    ecg_channel = ecg_channel[:n]

    try:
        import neurokit2 as nk

        _, info = nk.ecg_process(ecg_channel, sampling_rate=int(Fs))
        r_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    except Exception:
        ecg_std = np.std(ecg_channel)
        if ecg_std < 1e-12:
            return eeg_channel

        z_ecg = (ecg_channel - np.mean(ecg_channel)) / ecg_std
        min_distance = max(1, int(0.45 * Fs))
        r_peaks, _ = signal.find_peaks(z_ecg, height=1.5, distance=min_distance)

    hw = int(0.2 * Fs)
    if hw <= 0:
        return eeg_channel

    templates = []
    for rp in r_peaks:
        if rp - hw >= 0 and rp + hw < len(eeg_channel):
            templates.append(eeg_channel[rp - hw:rp + hw])

    if len(templates) == 0:
        return eeg_channel

    template = np.mean(templates, axis=0)

    eeg_clean = eeg_channel.copy()
    for rp in r_peaks:
        if rp - hw >= 0 and rp + hw < len(eeg_channel):
            eeg_clean[rp - hw:rp + hw] -= template

    return eeg_clean


def _find_blink_pair(eeg, ch_names):
    if ch_names is None:
        return None

    cleaned = [clean_channel_name(c) for c in ch_names]
    pairs = [
        ("FP1", "FP2"),
        ("AF3", "AF4"),
        ("F7", "F8"),
        ("F3", "F4"),
    ]

    for left, right in pairs:
        if left in cleaned and right in cleaned:
            left_idx = cleaned.index(left)
            right_idx = cleaned.index(right)
            if left_idx < eeg.shape[0] and right_idx < eeg.shape[0]:
                return eeg[left_idx], eeg[right_idx]

    return None


# =========================
# MAIN QUALITY FUNCTION
# =========================
def compute_signal_quality(eeg, sfreq=256, ch_names=None):
    """
    eeg: shape (channels, samples)
    ch_names: optional (recommended)
    """

    # -----------------------
    # SHAPE FIX
    # -----------------------
    try:
        eeg = np.asarray(eeg, dtype=float)
    except Exception:
        return _safe_quality_result()

    if eeg.ndim != 2 or eeg.shape[0] == 0 or eeg.shape[1] == 0:
        return _safe_quality_result()

    if eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T

    eeg_original = eeg.copy()

    # -----------------------
    # FILTER CHANNELS
    # -----------------------
    eeg, selected_idx = filter_eeg_channels(eeg, ch_names)

    if eeg is None or len(selected_idx) < 4:
        return _safe_quality_result()

    eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
    n_channels = eeg.shape[0]
    if ch_names is None:
        selected_channel_names = [str(i) for i in selected_idx]
    else:
        selected_channel_names = [ch_names[i] for i in selected_idx]

    # -----------------------
    # BAD CHANNEL / MOVEMENT CHECK
    # -----------------------
    channel_quality = _assess_channel_quality(eeg, sfreq)
    bad_channels = channel_quality["bad_channels"]
    flat_channels = channel_quality["flat_channels"]
    noisy_channels = channel_quality["noisy_channels"]
    movement_detected, movement_z, movement_fraction = _detect_movement_windows(eeg, sfreq)

    # -----------------------
    # FREQUENCY ANALYSIS
    # -----------------------
    alpha_power = []
    theta_power = []
    hf_power = []
    line_power = []
    line_total_ratios = []
    line_neighbor_ratios = []
    emg_flags = []
    emg_confidence = []
    emg_ratios = []

    for ch in eeg:
        freqs, psd = _welch(ch, sfreq)
        if len(freqs) == 0:
            continue

        total_band_power = _band_power(freqs, psd, 1, 70)
        line_band_power = _band_power(freqs, psd, 48, 52)
        line_neighbor_power = max(_band_power(freqs, psd, 45, 55) - line_band_power, 0.0)

        alpha_power.append(_band_power(freqs, psd, 8, 13))
        theta_power.append(_band_power(freqs, psd, 4, 8))
        hf_power.append(_band_power(freqs, psd, 30, 80))
        line_power.append(line_band_power)
        line_total_ratios.append(line_band_power / (total_band_power + 1e-20))
        line_neighbor_ratios.append(line_band_power / (line_neighbor_power + 1e-20))

        is_emg, confidence, emg_ratio, _ = detect_emg_artifact(ch, sfreq)
        emg_flags.append(is_emg)
        emg_confidence.append(confidence if is_emg else 0.0)
        emg_ratios.append(emg_ratio)

    if len(alpha_power) == 0:
        return _safe_quality_result()

    alpha_power = max(float(np.mean(alpha_power)), 1e-20)
    theta_power = float(np.mean(theta_power))
    hf_power = float(np.mean(hf_power))
    line_power = float(np.mean(line_power))

    # -----------------------
    # ARTIFACT DETECTION
    # -----------------------
    hf_alpha_ratio = hf_power / alpha_power
    emg_flag_fraction = np.mean(emg_flags) if len(emg_flags) else 0.0
    mean_emg_confidence = np.mean(emg_confidence) if len(emg_confidence) else 0.0
    mean_emg_ratio = np.mean(emg_ratios) if len(emg_ratios) else 0.0

    emg_ratio_score = _scale01(mean_emg_ratio, 0.15, 0.40)
    hf_alpha_score = 0.65 * _scale01(hf_alpha_ratio, 2.0, 8.0)
    muscle_score = max(
        emg_flag_fraction,
        mean_emg_confidence,
        emg_ratio_score,
        hf_alpha_score,
    )
    muscle = _severity_from_score(muscle_score, moderate_at=0.35, high_at=0.75)

    theta_alpha_ratio = theta_power / alpha_power
    blink_pair = _find_blink_pair(eeg_original, ch_names)
    blink_detected = False
    blink_count = 0
    blink_corr = 0.0
    if blink_pair is not None:
        blink_detected, blink_count, blink_corr = detect_eye_blink(
            blink_pair[0], blink_pair[1], sfreq
        )

    duration_min = max(eeg.shape[1] / float(sfreq) / 60.0, 1e-9)
    blink_rate_per_min = blink_count / duration_min
    blink_score = _scale01(blink_corr, 0.35, 0.75) * _scale01(
        blink_rate_per_min, 3.0, 20.0
    )
    if blink_detected:
        blink_score = max(blink_score, 0.65)

    theta_eye_score = 0.55 * _scale01(theta_alpha_ratio, 1.8, 4.0)
    eye_score = max(blink_score, theta_eye_score)
    eye = _severity_from_score(eye_score, moderate_at=0.35, high_at=0.75)

    mean_line_fraction = float(np.mean(line_total_ratios)) if len(line_total_ratios) else 0.0
    median_line_neighbor_ratio = (
        float(np.median(line_neighbor_ratios)) if len(line_neighbor_ratios) else 0.0
    )
    noisy_line_fraction = (
        float(np.mean([
            total_ratio > 0.12 and neighbor_ratio > 4.0
            for total_ratio, neighbor_ratio in zip(line_total_ratios, line_neighbor_ratios)
        ]))
        if len(line_total_ratios)
        else 0.0
    )

    line_score = max(
        _scale01(mean_line_fraction, 0.02, 0.20),
        _scale01(noisy_line_fraction, 0.05, 0.55),
        _scale01(median_line_neighbor_ratio, 2.0, 8.0)
        * _scale01(mean_line_fraction, 0.01, 0.08),
    )
    line_noise = _severity_from_score(line_score, moderate_at=0.35, high_at=0.75)

    movement_score = max(
        _scale01(movement_fraction, 0.005, 0.10),
        0.5
        * _scale01(movement_z, 8.0, 25.0)
        * _scale01(movement_fraction, 0.005, 0.03),
    )
    movement = _severity_from_score(movement_score, moderate_at=0.25, high_at=0.60)

    # -----------------------
    # FINAL SCORE
    # -----------------------
    score = 100.0

    # Bad channels are important, but one bad sensor should not dominate the
    # whole-file score when the rest of the recording is usable.
    bad_channel_fraction = len(bad_channels) / max(n_channels, 1)
    bad_channel_penalty = min(
        18.0,
        len(bad_channels) * 2.0 + 8.0 * channel_quality["bad_channel_score"],
    )

    # missing channel penalty
    expected_channels = _expected_channel_count(selected_channel_names)
    missing_channels = 0
    missing_channel_penalty = 0.0
    if expected_channels is not None:
        missing_channels = max(0, expected_channels - n_channels)
        missing_channel_penalty = min(12.0, missing_channels * 1.25)

    muscle_penalty = 18.0 * (muscle_score ** 1.1)
    eye_penalty = 14.0 * eye_score
    line_penalty = 12.0 * line_score
    movement_penalty = 12.0 * movement_score

    penalties = {
        "bad_channels": bad_channel_penalty,
        "missing_channels": missing_channel_penalty,
        "muscle_artifacts": muscle_penalty,
        "eye_blinks": eye_penalty,
        "line_noise": line_penalty,
        "movement_artifacts": movement_penalty,
    }

    score -= sum(penalties.values())

    score = max(min(score, 100), 0)

    return {
        "overall_quality": int(round(score)),
        "bad_channels": bad_channels,
        "bad_channel_names": _channel_names_from_indices(
            selected_channel_names,
            bad_channels,
        ),
        "muscle_artifacts": muscle,
        "eye_blinks": eye,
        "line_noise": line_noise,
        "movement_artifacts": movement,
        "quality_details": {
            "selected_channels": [str(name) for name in selected_channel_names],
            "selected_channel_count": int(n_channels),
            "expected_channel_count": (
                int(expected_channels) if expected_channels is not None else None
            ),
            "missing_channel_count": int(missing_channels),
            "penalties": {
                name: _round_metric(value, digits=2)
                for name, value in penalties.items()
            },
            "scores": {
                "bad_channel_fraction": _round_metric(bad_channel_fraction),
                "bad_channel_burden": _round_metric(channel_quality["bad_channel_score"]),
                "muscle": _round_metric(muscle_score),
                "eye": _round_metric(eye_score),
                "line_noise": _round_metric(line_score),
                "movement": _round_metric(movement_score),
            },
            "metrics": {
                "hf_alpha_ratio": _round_metric(hf_alpha_ratio),
                "mean_emg_ratio": _round_metric(mean_emg_ratio),
                "emg_flag_fraction": _round_metric(emg_flag_fraction),
                "theta_alpha_ratio": _round_metric(theta_alpha_ratio),
                "blink_count": int(blink_count),
                "blink_rate_per_min": _round_metric(blink_rate_per_min),
                "blink_correlation": _round_metric(blink_corr),
                "mean_line_fraction": _round_metric(mean_line_fraction),
                "median_line_neighbor_ratio": _round_metric(median_line_neighbor_ratio),
                "noisy_line_fraction": _round_metric(noisy_line_fraction),
                "movement_detected": bool(movement_detected),
                "movement_z": _round_metric(movement_z),
                "movement_fraction": _round_metric(movement_fraction),
            },
            "channel_quality": {
                "flat_channels": flat_channels,
                "flat_channel_names": _channel_names_from_indices(
                    selected_channel_names,
                    flat_channels,
                ),
                "noisy_channels": noisy_channels,
                "noisy_channel_names": _channel_names_from_indices(
                    selected_channel_names,
                    noisy_channels,
                ),
                "dropout_channels": channel_quality["dropout_channels"],
                "dropout_channel_names": _channel_names_from_indices(
                    selected_channel_names,
                    channel_quality["dropout_channels"],
                ),
                "low_correlation_channels": channel_quality["low_correlation_channels"],
                "low_correlation_channel_names": _channel_names_from_indices(
                    selected_channel_names,
                    channel_quality["low_correlation_channels"],
                ),
                "channel_scores": [
                    _round_metric(value) for value in channel_quality["channel_scores"]
                ],
                "median_std_uv": [
                    _round_metric(value) for value in channel_quality["median_std_uv"]
                ],
                "amplitude_z": [
                    _round_metric(value) for value in channel_quality["amplitude_z"]
                ],
                "low_correlation_fraction": [
                    _round_metric(value)
                    for value in channel_quality["low_correlation_fraction"]
                ],
                "dropout_fraction": [
                    _round_metric(value) for value in channel_quality["dropout_fraction"]
                ],
                "median_max_correlation": [
                    _round_metric(value)
                    for value in channel_quality["median_max_correlation"]
                ],
            },
        },
    }


def analyze_quality_metadata(edf_metadata):
    normalized = normalize_edf_metadata(edf_metadata)
    quality_result = compute_signal_quality(
        normalized.data,
        sfreq=normalized.fs,
        ch_names=normalized.original_ch_names,
    )

    quality_result.setdefault("quality_details", {})
    quality_result["quality_details"]["normalized_channel_names"] = normalized.ch_names
    quality_result["quality_details"]["metadata_warnings"] = normalized.warnings

    return {
        "status": "success",
        "quality_result": quality_result,
    }
