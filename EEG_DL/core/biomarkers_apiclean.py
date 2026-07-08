import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
import mne

# =====================================================
# CHANNEL GROUPS
# =====================================================

REGION_INDEX = {
"frontal":0,
"central":1,
"parietal":2,
"occipital":3,
"temporal":4
}


BANDS = {

"delta":(1,4),
"theta":(4,8),
"alpha":(8,13),
"beta":(13,30),
"gamma":(30,45)

}

# =====================================================
# SIGNAL FUNCTIONS
# =====================================================

def bandpower(signal, sfreq, band):
    fmin, fmax = BANDS[band]
    freqs, psd = welch(signal, sfreq, nperseg=min(len(signal), sfreq*2))
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[mask], freqs[mask])

def spectral_entropy(signal, sfreq):
    freqs, psd = welch(signal, sfreq, nperseg=min(len(signal), sfreq*2))
    psd_norm = psd / (psd.sum() + 1e-12)
    return entropy(psd_norm)

# =====================================================
# EEG STABILIZATION (Reusable)
# =====================================================

def stabilize_signal(eeg, sfreq, target_sfreq=256, max_duration_sec=30):

    if eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T

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

    return raw.get_data(), target_sfreq

# =====================================================
# BIOMARKERS (UNCHANGED)
# =====================================================

def compute_biomarkers(eeg, sfreq):
    if eeg.shape[0] < 4:
        raise ValueError("Not enough regional signals detected")

    fr = REGION_INDEX["frontal"]
    occ = REGION_INDEX["occipital"]

    occ_alpha = bandpower(eeg[occ], sfreq, "alpha")
    global_alpha = np.mean([
        bandpower(ch, sfreq, "alpha") for ch in eeg
    ])

    occ_entropy = spectral_entropy(eeg[occ], sfreq)

    frontal_theta = bandpower(eeg[fr], sfreq, "theta")
    frontal_alpha = bandpower(eeg[fr], sfreq, "alpha")

    frontal_entropy = spectral_entropy(eeg[fr], sfreq)

    return {

        "posterior_dominance_index":
            occ_alpha / (global_alpha + 1e-6),

        "occipital_entropy":
            occ_entropy,

        "alpha_peak_gradient" :
            (occ_alpha-frontal_alpha)/ (occ_alpha + frontal_alpha + 1e-6),

        "entropy_gradient":
            frontal_entropy - occ_entropy,

        "theta_alpha_ratio_frontal":
            np.clip(frontal_theta / (frontal_alpha + 1e-6), 0, 10)
    }

# =====================================================
# PIPELINE EXECUTION (DISABLED FOR API SAFETY)
# =====================================================

if __name__ == "__main__":
    print("Standalone biomarker pipeline disabled in API mode.")
