import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
import mne

# =====================================================
# CHANNEL GROUPS
# =====================================================

FRONTAL = [0, 1, 5, 6, 10, 13, 16]
OCCIPITAL = [4, 9]
TEMPORAL = [11, 12, 14, 15]
PARIETAL = [3, 8, 18]
CENTRAL = [2, 7, 17]

ALL_CHANNELS = OCCIPITAL + FRONTAL + PARIETAL + CENTRAL + TEMPORAL

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13)
}

# =====================================================
# SIGNAL FUNCTIONS
# =====================================================

def bandpower(signal, sfreq, band):
    fmin, fmax = BANDS[band]
    freqs, psd = welch(signal, sfreq, nperseg=2 * sfreq)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.sum(psd[mask])

def spectral_entropy(signal, sfreq):
    freqs, psd = welch(signal, sfreq, nperseg=2 * sfreq)
    psd_norm = psd / (psd.sum() + 1e-12)
    return entropy(psd_norm)

# =====================================================
# EEG STABILIZATION (Reusable)
# =====================================================

def stabilize_signal(eeg, sfreq, target_sfreq=200, max_duration_sec=30):

    max_samples = int(max_duration_sec * sfreq)
    eeg = eeg[:, :max_samples]

    info = mne.create_info(
        ch_names=[f"EEG{i}" for i in range(eeg.shape[0])],
        sfreq=sfreq,
        ch_types="eeg"
    )

    raw = mne.io.RawArray(eeg, info, verbose=False)

    if sfreq != target_sfreq:
        raw.resample(target_sfreq, verbose=False)

    raw.filter(0.5, 45, verbose=False)

    return raw.get_data(), target_sfreq

# =====================================================
# BIOMARKERS (UNCHANGED)
# =====================================================

def compute_biomarkers(eeg, sfreq):

    occ_alpha = np.mean([bandpower(eeg[ch], sfreq, "alpha") for ch in OCCIPITAL])
    global_alpha = np.mean([bandpower(eeg[ch], sfreq, "alpha") for ch in ALL_CHANNELS])

    occ_entropy = np.mean([spectral_entropy(eeg[ch], sfreq) for ch in OCCIPITAL])

    frontal_theta = np.mean([bandpower(eeg[ch], sfreq, "theta") for ch in FRONTAL])
    frontal_alpha = np.mean([bandpower(eeg[ch], sfreq, "alpha") for ch in FRONTAL])

    frontal_entropy = np.mean([spectral_entropy(eeg[ch], sfreq) for ch in FRONTAL])

    return {
        "posterior_dominance_index": occ_alpha / (global_alpha + 1e-6),
        "occipital_entropy": occ_entropy,
        "alpha_peak_gradient": frontal_alpha - occ_alpha,
        "entropy_gradient": frontal_entropy - occ_entropy,
        "theta_alpha_ratio_frontal": frontal_theta / (frontal_alpha + 1e-6)
    }

# =====================================================
# PIPELINE EXECUTION (DISABLED FOR API SAFETY)
# =====================================================

if __name__ == "__main__":
    print("Standalone biomarker pipeline disabled in API mode.")
