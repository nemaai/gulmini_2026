"""
Combined EDF sleep-marker quality analysis.

The backend can pass one EDF metadata dictionary into `analyze_edf_metadata`.
This module normalizes the metadata once, then feeds the same EEG matrix,
channel names, and sampling rate into:

- sleep spindle detection
- vertex wave detection
- K-complex detection

Expected signal shape after normalization:
    data: numpy array with shape (n_channels, n_samples), internally in volts
    ch_names: canonical uppercase channel names such as CZ, C3, F3
    Fs: one sampling rate in Hz
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import signal
except ImportError:  # pragma: no cover - handled at runtime for clearer API errors
    signal = None


TARGET_CHANNELS = ("FZ", "CZ", "C3", "C4", "PZ", "P3", "P4", "F3", "F4", "O1", "O2")
SIGNAL_KEYS = ("data", "raw_data", "signals", "samples", "signal_data", "eeg_data")
CHANNEL_NAME_KEYS = ("ch_names", "channel_names", "channels", "labels", "signal_labels")
SAMPLE_RATE_KEYS = (
    "Fs",
    "fs",
    "sfreq",
    "sample_rate",
    "sample_rate_hz",
    "sample_frequency",
    "sample_frequency_hz",
    "sampling_rate",
    "sampling_frequency",
    "sampling_frequency_hz",
    "frequency",
)
SAMPLES_KEYS = ("data", "samples", "signal", "values", "raw", "raw_data")
LABEL_KEYS = ("label", "name", "channel", "ch_name", "channel_name")
UNIT_KEYS = ("unit", "units", "dimension", "physical_dimension")


@dataclass
class NormalizedEDF:
    """Single normalized representation used by every detector."""

    data: np.ndarray
    ch_names: List[str]
    original_ch_names: List[str]
    fs: float
    duration_s: float
    units: List[str]
    warnings: List[str]


def _require_scipy() -> None:
    if signal is None:
        raise RuntimeError(
            "scipy is required for sleep marker detection. Install it with `pip install scipy`."
        )


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _canonical_channel_name(name: Any) -> str:
    text = str(name).strip().upper()
    text = text.replace("EEG", " ")
    text = text.replace(".", "")
    tokens = [token for token in re.split(r"[^A-Z0-9]+", text) if token]

    for token in tokens:
        if token in TARGET_CHANNELS:
            return token

    for channel in TARGET_CHANNELS:
        if re.search(rf"(^|[^A-Z0-9]){re.escape(channel)}([^A-Z0-9]|$)", text):
            return channel

    return text or "UNKNOWN"


def _channel_name_from_item(item: Any, index: int) -> str:
    if isinstance(item, Mapping):
        value = _first_present(item, LABEL_KEYS)
        if value is not None:
            return str(value)
    return str(item) if item is not None else f"CH{index + 1}"


def _unit_from_item(item: Any) -> str:
    if isinstance(item, Mapping):
        value = _first_present(item, UNIT_KEYS)
        if value is not None:
            return str(value)
    return ""


def _unit_scale_to_volts(unit: str) -> Optional[float]:
    normalized = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    if not normalized:
        return None
    if normalized in {"uv", "uvolt", "uvolts", "microvolt", "microvolts"}:
        return 1e-6
    if normalized in {"mv", "millivolt", "millivolts"}:
        return 1e-3
    if normalized in {"v", "volt", "volts"}:
        return 1.0
    return None


def _infer_scale_to_volts(values: np.ndarray) -> Tuple[float, str]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0, "unknown"

    p95 = float(np.percentile(np.abs(finite), 95))
    if p95 > 1.0:
        return 1e-6, "auto_microvolts"
    if p95 > 0.01:
        return 1e-3, "auto_millivolts"
    return 1.0, "auto_volts"


def _extract_channel_dict_signals(channels: Sequence[Any]) -> Tuple[Optional[np.ndarray], List[str], List[str], List[Optional[float]]]:
    rows: List[np.ndarray] = []
    names: List[str] = []
    units: List[str] = []
    rates: List[Optional[float]] = []

    for idx, item in enumerate(channels):
        if not isinstance(item, Mapping):
            return None, [], [], []

        samples = _first_present(item, SAMPLES_KEYS)
        if samples is None:
            return None, [], [], []

        row = np.asarray(samples, dtype=float)
        if row.ndim != 1:
            row = np.ravel(row)

        rows.append(row)
        names.append(_channel_name_from_item(item, idx))
        units.append(_unit_from_item(item))
        rates.append(_as_float(_first_present(item, SAMPLE_RATE_KEYS)))

    if not rows:
        return None, [], [], []

    min_len = min(len(row) for row in rows)
    if min_len == 0:
        return np.empty((len(rows), 0), dtype=float), names, units, rates

    trimmed = np.vstack([row[:min_len] for row in rows])
    return trimmed, names, units, rates


def _extract_signal_headers(metadata: Mapping[str, Any]) -> Tuple[List[str], List[str], List[Optional[float]]]:
    headers = metadata.get("signal_headers") or metadata.get("headers") or []
    if not isinstance(headers, Sequence) or isinstance(headers, (str, bytes)):
        return [], [], []

    names: List[str] = []
    units: List[str] = []
    rates: List[Optional[float]] = []
    for idx, header in enumerate(headers):
        if not isinstance(header, Mapping):
            continue
        names.append(_channel_name_from_item(header, idx))
        units.append(_unit_from_item(header))
        rates.append(_as_float(_first_present(header, SAMPLE_RATE_KEYS)))
    return names, units, rates


def _expand_units(raw_units: Any, n_channels: int) -> Optional[List[str]]:
    if raw_units is None:
        return None
    if isinstance(raw_units, str):
        return [raw_units] * n_channels
    if isinstance(raw_units, Sequence):
        expanded = [str(unit) for unit in raw_units]
        return (expanded + [""] * n_channels)[:n_channels]
    return [str(raw_units)] * n_channels


def _normalize_2d_array(data: Any, channel_names: Optional[Sequence[Any]]) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError("EDF signal data must be a 1D or 2D array-like object.")

    if channel_names:
        n_names = len(channel_names)
        if array.shape[0] == n_names:
            return array
        if array.shape[1] == n_names:
            return array.T

    if array.shape[0] > array.shape[1]:
        # Most EEG inputs are channels x samples; this handles common samples x channels uploads.
        return array.T

    return array


def _resample_rows_if_needed(
    data: np.ndarray,
    fs: float,
    per_channel_rates: Sequence[Optional[float]],
    warnings: List[str],
) -> Tuple[np.ndarray, float]:
    known_rates = [rate for rate in per_channel_rates if rate is not None]
    if not known_rates:
        return data, fs

    rounded = {round(rate, 6) for rate in known_rates}
    if len(rounded) <= 1:
        return data, known_rates[0]

    _require_scipy()
    target_fs = max(known_rates)
    resampled_rows: List[np.ndarray] = []

    for idx, row in enumerate(data):
        source_fs = per_channel_rates[idx] if idx < len(per_channel_rates) else fs
        if source_fs is None or abs(source_fs - target_fs) < 1e-6:
            resampled_rows.append(np.asarray(row, dtype=float))
            continue

        ratio = Fraction(target_fs / source_fs).limit_denominator(1000)
        resampled_rows.append(signal.resample_poly(row, ratio.numerator, ratio.denominator))

    min_len = min(len(row) for row in resampled_rows)
    warnings.append(
        f"EDF channels had mixed sample rates; all channels were resampled to {target_fs:.3f} Hz."
    )
    return np.vstack([row[:min_len] for row in resampled_rows]), target_fs


def normalize_edf_metadata(metadata: Any) -> NormalizedEDF:
    """
    Normalize raw EDF metadata into one signal matrix for all detectors.

    Supported input examples:
        {"data": [[...], [...]], "ch_names": ["CZ", "C3"], "Fs": 256}
        {"signals": [[...], [...]], "channels": ["EEG Cz-REF", "EEG C3-REF"], "sample_rate": 256}
        {"channels": [{"label": "CZ", "samples": [...], "sample_rate": 256, "unit": "uV"}]}
        (data, ch_names, Fs)
    """

    warnings: List[str] = []

    if isinstance(metadata, tuple) and len(metadata) >= 3:
        raw_data, raw_names, raw_fs = metadata[:3]
        data = _normalize_2d_array(raw_data, raw_names)
        original_names = [str(name) for name in raw_names]
        units = [""] * len(original_names)
        fs = _as_float(raw_fs)
        per_channel_rates: List[Optional[float]] = []
    elif isinstance(metadata, Mapping):
        wrapped = metadata
        if "edf_metadata" in wrapped and isinstance(wrapped["edf_metadata"], Mapping):
            wrapped = wrapped["edf_metadata"]
        elif "metadata" in wrapped and isinstance(wrapped["metadata"], Mapping) and not any(
            key in wrapped for key in SIGNAL_KEYS
        ):
            wrapped = wrapped["metadata"]

        header_names, header_units, header_rates = _extract_signal_headers(wrapped)
        raw_channels = _first_present(wrapped, CHANNEL_NAME_KEYS)
        raw_data = _first_present(wrapped, SIGNAL_KEYS)
        global_units = _first_present(wrapped, UNIT_KEYS)

        per_channel_rates = []
        units = []

        if raw_data is None and isinstance(raw_channels, Sequence) and not isinstance(raw_channels, (str, bytes)):
            data, original_names, units, per_channel_rates = _extract_channel_dict_signals(raw_channels)
            if data is None:
                raise ValueError(
                    "EDF metadata has channels but no signal samples. Include sample arrays under "
                    "`data`, `signals`, or each channel's `samples` field."
                )
        else:
            if raw_data is None:
                raise ValueError(
                    "EDF metadata does not contain signal samples. Header-only EDF metadata is not enough "
                    "for spindle, vertex wave, or K-complex detection."
                )

            channel_name_items: Optional[Sequence[Any]]
            if isinstance(raw_channels, Sequence) and not isinstance(raw_channels, (str, bytes)):
                channel_name_items = raw_channels
            elif header_names:
                channel_name_items = header_names
            else:
                channel_name_items = None

            data = _normalize_2d_array(raw_data, channel_name_items)
            if channel_name_items:
                original_names = [_channel_name_from_item(item, idx) for idx, item in enumerate(channel_name_items)]
                units = [_unit_from_item(item) for item in channel_name_items]
            else:
                original_names = [f"CH{idx + 1}" for idx in range(data.shape[0])]
                units = [""] * data.shape[0]

            if header_units and not any(units):
                units = header_units
            expanded_units = _expand_units(global_units, data.shape[0])
            if expanded_units and not any(units):
                units = expanded_units
            if header_rates:
                per_channel_rates = header_rates

        expanded_units = _expand_units(global_units, data.shape[0])
        if expanded_units and not any(units):
            units = expanded_units

        fs = _as_float(_first_present(wrapped, SAMPLE_RATE_KEYS))
        if fs is None and per_channel_rates:
            known_rates = [rate for rate in per_channel_rates if rate is not None]
            if known_rates:
                fs = known_rates[0]
    else:
        raise TypeError("EDF metadata must be a dictionary or a (data, ch_names, Fs) tuple.")

    if fs is None:
        raise ValueError("EDF metadata must include a positive sampling rate such as `Fs` or `sample_rate`.")

    if data.size == 0 or data.shape[1] == 0:
        raise ValueError("EDF signal data is empty.")

    if len(original_names) != data.shape[0]:
        if len(original_names) > data.shape[0]:
            original_names = original_names[: data.shape[0]]
        else:
            original_names = original_names + [
                f"CH{idx + 1}" for idx in range(len(original_names), data.shape[0])
            ]

    if len(units) != data.shape[0]:
        units = (units + [""] * data.shape[0])[: data.shape[0]]

    data = np.asarray(data, dtype=float)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data, fs = _resample_rows_if_needed(data, fs, per_channel_rates, warnings)

    for idx in range(data.shape[0]):
        scale = _unit_scale_to_volts(units[idx])
        scale_source = units[idx].strip() if units[idx] else ""
        if scale is None:
            scale, scale_source = _infer_scale_to_volts(data[idx])
        data[idx] = data[idx] * scale
        if scale_source.startswith("auto_"):
            warnings.append(
                f"Unit missing for channel {original_names[idx]}; interpreted values as {scale_source[5:]}."
            )

    canonical_names = [_canonical_channel_name(name) for name in original_names]
    duration_s = data.shape[1] / fs

    return NormalizedEDF(
        data=data,
        ch_names=canonical_names,
        original_ch_names=original_names,
        fs=fs,
        duration_s=duration_s,
        units=units,
        warnings=warnings,
    )


def _bandpass_filter(sig: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    _require_scipy()
    nyq = fs / 2.0
    if low <= 0 or high >= nyq:
        raise ValueError(f"Cannot bandpass {low}-{high} Hz with sampling rate {fs:.3f} Hz.")

    sos = signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    try:
        return signal.sosfiltfilt(sos, sig)
    except ValueError:
        if len(sig) < max(16, int(fs)):
            raise
        return signal.sosfilt(sos, sig)


def _welch_peak_frequency(seg: np.ndarray, fs: float, low: float, high: float) -> Optional[float]:
    _require_scipy()
    if len(seg) < max(8, int(0.25 * fs)):
        return None

    nperseg = min(len(seg), max(8, int(fs * 0.5)))
    freqs, power = signal.welch(seg, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return None
    return float(freqs[mask][np.argmax(power[mask])])


def detect_sleep_spindles(data: np.ndarray, ch_names: Sequence[str], Fs: float) -> Dict[str, Any]:
    """Detect 11-16 Hz sleep spindles from the shared EDF signal matrix."""

    if Fs <= 32:
        return {
            "spindles_detected": False,
            "n_spindles": 0,
            "spindle_density_per_min": 0.0,
            "spindle_events": [],
            "warning": "Sampling rate must be greater than 32 Hz for 11-16 Hz spindle detection.",
        }

    spindle_channels = [channel for channel in ("CZ", "C3", "C4", "PZ", "P3", "P4") if channel in ch_names]
    if not spindle_channels:
        spindle_channels = [ch_names[0]]

    all_spindles: List[Dict[str, Any]] = []

    for channel in spindle_channels:
        idx = ch_names.index(channel)
        raw_sig = data[idx]
        try:
            sig_filtered = _bandpass_filter(raw_sig, Fs, 11.0, 16.0)
        except ValueError as exc:
            return {
                "spindles_detected": False,
                "n_spindles": 0,
                "spindle_density_per_min": 0.0,
                "spindle_events": [],
                "warning": str(exc),
            }

        win = max(1, int(round(0.125 * Fs)))
        kernel = np.ones(win) / win
        envelope = np.sqrt(np.convolve(sig_filtered**2, kernel, mode="same"))

        threshold = float(np.mean(envelope) + 1.5 * np.std(envelope))
        above = (envelope > threshold).astype(int)
        transitions = np.diff(above, prepend=0, append=0)
        starts = np.where(transitions == 1)[0]
        ends = np.where(transitions == -1)[0]

        for start, end in zip(starts, ends):
            duration_s = (end - start) / Fs
            if not (0.5 <= duration_s <= 3.0):
                continue

            seg = sig_filtered[start:end]
            env_seg = envelope[start:end]
            if len(seg) < int(0.5 * Fs) or len(env_seg) < 3:
                continue

            peak_f = _welch_peak_frequency(seg, Fs, 11.0, 16.0)
            if peak_f is None or not (11.0 <= peak_f <= 16.0):
                continue

            peak_idx = int(np.argmax(env_seg))
            peak_position = peak_idx / max(1, len(env_seg) - 1)
            cv_env = float(np.std(env_seg) / (np.mean(env_seg) + 1e-10))
            wax_wane_ok = bool(0.15 <= peak_position <= 0.85 and cv_env > 0.20)
            amplitude_uV = float((np.max(raw_sig[start:end]) - np.min(raw_sig[start:end])) * 1e6)

            all_spindles.append(
                {
                    "channel": channel,
                    "start_s": round(start / Fs, 3),
                    "end_s": round(end / Fs, 3),
                    "duration_s": round(duration_s, 3),
                    "peak_freq_Hz": round(peak_f, 1),
                    "amplitude_uV": round(amplitude_uV, 1),
                    "spindle_type": "slow" if peak_f <= 13 else "fast",
                    "wax_wane_ok": wax_wane_ok,
                    "CV_envelope": round(cv_env, 3),
                }
            )

    recording_min = data.shape[1] / Fs / 60.0
    spindle_density = len(all_spindles) / (recording_min + 1e-10)

    return {
        "spindles_detected": len(all_spindles) > 0,
        "n_spindles": len(all_spindles),
        "spindle_density_per_min": round(spindle_density, 2),
        "spindle_events": all_spindles,
    }


def detect_vertex_waves(data: np.ndarray, ch_names: Sequence[str], Fs: float) -> Dict[str, Any]:
    """Detect vertex waves from the same normalized EDF signal matrix."""

    if "CZ" not in ch_names:
        return {
            "vertex_detected": False,
            "n_vertex": 0,
            "events": [],
            "warning": "CZ channel is required for vertex wave detection.",
        }
    if Fs <= 20:
        return {
            "vertex_detected": False,
            "n_vertex": 0,
            "events": [],
            "warning": "Sampling rate must be greater than 20 Hz for 1-10 Hz vertex wave detection.",
        }

    _require_scipy()
    cz_idx = ch_names.index("CZ")
    sig_cz = data[cz_idx]

    try:
        sig_filt = _bandpass_filter(sig_cz, Fs, 1.0, 10.0)
    except ValueError as exc:
        return {"vertex_detected": False, "n_vertex": 0, "events": [], "warning": str(exc)}

    neg_thresh = -2.5 * float(np.std(sig_filt))
    neg_peaks, _ = signal.find_peaks(
        -sig_filt,
        height=-neg_thresh,
        distance=max(1, int(0.5 * Fs)),
        prominence=abs(neg_thresh) * 0.5,
    )

    vertex_events: List[Dict[str, Any]] = []
    half_window = int(0.3 * Fs)

    for peak in neg_peaks:
        if peak - half_window < 0 or peak + half_window >= len(sig_filt):
            continue

        peak_amp = float(sig_filt[peak])
        half_amp = peak_amp / 2.0

        left_range = np.arange(max(0, peak - half_window), peak)
        right_range = np.arange(peak, min(len(sig_filt), peak + half_window))
        left_cross = left_range[sig_filt[left_range] > half_amp]
        right_cross = right_range[sig_filt[right_range] > half_amp]
        if len(left_cross) == 0 or len(right_cross) == 0:
            continue

        width_s = float((right_cross[0] - left_cross[-1]) / Fs)
        if not (0.08 <= width_s <= 0.30):
            continue

        post_seg = sig_filt[peak : peak + half_window]
        max_pos = float(np.max(post_seg)) if len(post_seg) else 0.0
        biphasic_ratio = max_pos / (abs(peak_amp) + 1e-10)
        if biphasic_ratio < 0.25:
            continue

        surround = [channel for channel in ("F3", "F4", "O1", "O2") if channel in ch_names]
        if surround:
            surround_amps = [abs(data[ch_names.index(channel), peak]) for channel in surround]
            topo_ratio = float(abs(peak_amp) / (np.mean(surround_amps) + 1e-10))
        else:
            topo_ratio = 1.0

        vertex_events.append(
            {
                "time_s": round(peak / Fs, 3),
                "amplitude_uV": round(peak_amp * 1e6, 1),
                "width_ms": round(width_s * 1000, 1),
                "biphasic_ratio": round(biphasic_ratio, 2),
                "topo_ratio": round(topo_ratio, 2),
            }
        )

    return {
        "vertex_detected": len(vertex_events) > 0,
        "n_vertex": len(vertex_events),
        "events": vertex_events,
    }


def detect_k_complexes(data: np.ndarray, ch_names: Sequence[str], Fs: float) -> Dict[str, Any]:
    """Detect K-complexes from the same normalized EDF signal matrix."""

    kc_channels = [channel for channel in ("FZ", "CZ", "F3", "F4") if channel in ch_names]
    if not kc_channels:
        return {
            "kc_detected": False,
            "n_kc": 0,
            "events": [],
            "warning": "At least one FZ, CZ, F3, or F4 channel is required for K-complex detection.",
        }
    if Fs <= 6:
        return {
            "kc_detected": False,
            "n_kc": 0,
            "events": [],
            "warning": "Sampling rate must be greater than 6 Hz for 0.5-3 Hz K-complex detection.",
        }

    _require_scipy()
    all_kc: List[Dict[str, Any]] = []

    for channel in kc_channels:
        idx = ch_names.index(channel)
        try:
            sig = _bandpass_filter(data[idx], Fs, 0.5, 3.0)
        except ValueError as exc:
            return {"kc_detected": False, "n_kc": 0, "events": [], "warning": str(exc)}

        neg_thresh = -3.0 * float(np.std(sig))
        neg_peaks, _ = signal.find_peaks(
            -sig,
            height=-neg_thresh,
            distance=max(1, int(1.0 * Fs)),
            prominence=abs(neg_thresh) * 0.5,
        )

        for peak in neg_peaks:
            if peak + int(0.8 * Fs) >= len(sig) or peak - int(0.1 * Fs) < 0:
                continue

            slope = (sig[peak] - sig[max(0, peak - int(0.05 * Fs))]) / (0.05 + 1e-10)
            if abs(slope) < 25e-6:
                continue

            post_start = peak + int(0.05 * Fs)
            post_seg = sig[post_start : post_start + int(0.6 * Fs)]
            if len(post_seg) == 0:
                continue

            max_pos = float(np.max(post_seg))
            pos_time_ms = float(np.argmax(post_seg) / Fs * 1000.0)
            peak_neg_uV = float(sig[peak] * 1e6)
            max_pos_uV = float(max_pos * 1e6)

            if abs(peak_neg_uV) > 75 and max_pos_uV > 50 and pos_time_ms < 600:
                duration_s = (int(0.05 * Fs) + int(0.6 * Fs)) / Fs
                all_kc.append(
                    {
                        "channel": channel,
                        "time_s": round(peak / Fs, 3),
                        "neg_amp_uV": round(peak_neg_uV, 1),
                        "pos_amp_uV": round(max_pos_uV, 1),
                        "duration_s": round(duration_s, 3),
                        "score_amp_uV": round(abs(peak_neg_uV) + max_pos_uV, 1),
                    }
                )

    deduped = _deduplicate_k_complexes(all_kc)
    for event in deduped:
        event.pop("score_amp_uV", None)

    return {
        "kc_detected": len(deduped) > 0,
        "n_kc": len(deduped),
        "events": deduped,
    }


def _deduplicate_k_complexes(events: List[Dict[str, Any]], window_s: float = 0.5) -> List[Dict[str, Any]]:
    if not events:
        return []

    events = sorted(events, key=lambda item: item["time_s"])
    clusters: List[List[Dict[str, Any]]] = [[events[0]]]

    for event in events[1:]:
        if event["time_s"] - clusters[-1][-1]["time_s"] <= window_s:
            clusters[-1].append(event)
        else:
            clusters.append([event])

    deduped: List[Dict[str, Any]] = []
    for cluster in clusters:
        deduped.append(max(cluster, key=lambda item: item.get("score_amp_uV", 0)))
    return sorted(deduped, key=lambda item: item["time_s"])


def build_combined_analysis(
    normalized: NormalizedEDF,
    spindles: Mapping[str, Any],
    vertex: Mapping[str, Any],
    k_complexes: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create one human-readable marker quality interpretation."""

    findings: List[str] = []
    recommendations: List[str] = []

    spindle_density = float(spindles.get("spindle_density_per_min", 0.0) or 0.0)
    has_spindles = bool(spindles.get("spindles_detected"))
    has_vertex = bool(vertex.get("vertex_detected"))
    has_kc = bool(k_complexes.get("kc_detected"))

    score = 0
    if has_spindles:
        score += 35
        if 2.0 <= spindle_density <= 8.0:
            score += 15
            findings.append("Sleep spindles detected with density inside the usual N2 reference range.")
        else:
            findings.append("Sleep spindles detected, but density is outside the usual N2 reference range.")
    else:
        recommendations.append("No sleep spindles detected; verify central/parietal channels and signal quality.")

    if has_kc:
        score += 30
        findings.append("K-complexes detected, supporting N2 sleep-marker presence.")
    else:
        recommendations.append("No K-complexes detected in frontal-central channels.")

    if has_vertex:
        score += 10
        findings.append("Vertex waves detected, suggesting N1/drowsiness marker activity.")

    canonical_set = set(normalized.ch_names)
    central_channels = canonical_set.intersection({"CZ", "C3", "C4"})
    frontal_channels = canonical_set.intersection({"FZ", "F3", "F4"})

    channel_support_score = 0
    if central_channels:
        channel_support_score += 5
    if frontal_channels:
        channel_support_score += 5
    if normalized.duration_s >= 60:
        channel_support_score += 5
    score += channel_support_score

    if not central_channels:
        recommendations.append("Add CZ/C3/C4 channels if available; spindle and vertex checks are weaker without them.")
    if not frontal_channels:
        recommendations.append("Add FZ/F3/F4 channels if available; K-complex checks are weaker without them.")
    if normalized.duration_s < 60:
        recommendations.append("Recording duration is under 60 seconds; marker counts may be unstable.")

    score = int(max(0, min(100, score)))

    if has_spindles and has_kc and score >= 70:
        quality_label = "good_sleep_marker_quality"
    elif has_spindles or has_kc:
        quality_label = "moderate_sleep_marker_quality"
    elif has_vertex:
        quality_label = "limited_sleep_marker_quality"
    else:
        quality_label = "poor_or_no_sleep_marker_quality"

    if not findings:
        findings.append("No reliable spindle, vertex wave, or K-complex markers were detected.")

    sleep_stage_indicators = {
        "N1_or_drowsiness_supported": has_vertex,
        "N2_supported": has_spindles or has_kc,
        "N2_strongly_supported": has_spindles and has_kc,
    }

    return {
        "analysis_type": "combined_sleep_marker_quality",
        "quality_label": quality_label,
        "quality_score_0_100": score,
        "sleep_stage_indicators": sleep_stage_indicators,
        "findings": findings,
        "recommendations": recommendations,
        "important_note": (
            "This is marker-based EEG quality analysis. It does not replace artifact/noise scoring "
            "or clinical sleep staging."
        ),
    }


def analyze_edf_metadata(edf_metadata: Any) -> Dict[str, Any]:
    """
    Main entry point for the website backend.

    Pass the raw EDF metadata once; the function returns all three detections
    plus one combined final analysis.
    """

    try:
        normalized = normalize_edf_metadata(edf_metadata)
        spindles = detect_sleep_spindles(normalized.data, normalized.ch_names, normalized.fs)
        vertex = detect_vertex_waves(normalized.data, normalized.ch_names, normalized.fs)
        k_complexes = detect_k_complexes(normalized.data, normalized.ch_names, normalized.fs)
        combined = build_combined_analysis(normalized, spindles, vertex, k_complexes)

        detector_warnings = [
            result.get("warning")
            for result in (spindles, vertex, k_complexes)
            if isinstance(result, Mapping) and result.get("warning")
        ]

        return {
            "status": "success",
            "input_summary": {
                "n_channels": int(normalized.data.shape[0]),
                "n_samples": int(normalized.data.shape[1]),
                "sample_rate_Hz": round(float(normalized.fs), 3),
                "duration_s": round(float(normalized.duration_s), 3),
                "channels_original": normalized.original_ch_names,
                "channels_used": normalized.ch_names,
            },
            "detections": {
                "sleep_spindles": spindles,
                "vertex_waves": vertex,
                "k_complexes": k_complexes,
            },
            "combined_analysis": combined,
            "warnings": normalized.warnings + detector_warnings,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "combined_analysis": {
                "analysis_type": "combined_sleep_marker_quality",
                "quality_label": "insufficient_input_data",
                "quality_score_0_100": 0,
                "findings": [
                    "The EDF metadata could not be analyzed because usable signal samples, channel names, "
                    "or sampling rate were missing or invalid."
                ],
                "recommendations": [
                    "Send channel-wise EDF signal samples plus channel labels and sampling rate to this module."
                ],
            },
        }


def analyze_sleep_markers(edf_metadata: Any) -> Dict[str, Any]:
    """Alias kept for readable backend imports."""

    return analyze_edf_metadata(edf_metadata)


def analyze_sleep_quality(edf_metadata: Any) -> Dict[str, Any]:
    """Alias for teams calling this as a quality-analysis step."""

    return analyze_edf_metadata(edf_metadata)


def process_edf_metadata(edf_metadata: Any) -> Dict[str, Any]:
    """Alias for backend pipelines."""

    return analyze_edf_metadata(edf_metadata)


def detect_sleep_markers(edf_metadata: Any) -> Dict[str, Any]:
    """Alias for marker-detection pipelines."""

    return analyze_edf_metadata(edf_metadata)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Optional CLI: python sleep_marker_detection.py metadata.json"""

    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print("Usage: python sleep_marker_detection.py <edf_metadata.json>")
        return 2

    with open(argv[0], "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    result = analyze_edf_metadata(metadata)
    print(json.dumps(result, indent=2, default=_json_default))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
