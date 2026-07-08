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
    return np.trapezoid(psd[mask], freqs[mask])

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
CHANNEL_GROUPS = {

    "frontal": [
        "FP1","FP2",
        "F3","F4",
        "F7","F8",
        "FZ"
    ],

    "central": [
        "C3","C4","CZ"
    ],

    "parietal": [
        "P3","P4",
        "P7","P8",
        "PZ"
    ],

    "occipital": [
        "O1","O2","OZ"
    ],

    "temporal": [
        "T3","T4",
        "T5","T6",
        "T7","T8"
    ]
}

def get_region_channels(
    eeg,
    channel_names,
    region
):

    wanted = CHANNEL_GROUPS[region]

    signals = []

    for i, ch in enumerate(channel_names):

        ch = str(ch).upper()

        if ch in wanted:

            signals.append(
                eeg[i]
            )

    return signals


def compute_biomarkers(
    eeg,
    sfreq,
    channel_names
):

    if eeg.shape[0] > eeg.shape[1]:
        eeg = eeg.T

    channel_names = [
        str(x).upper()
        for x in channel_names
    ]

    frontal = get_region_channels(
        eeg,
        channel_names,
        "frontal"
    )

    occipital = get_region_channels(
        eeg,
        channel_names,
        "occipital"
    )

    temporal = get_region_channels(
        eeg,
        channel_names,
        "temporal"
    )

    if len(frontal) == 0:
        raise ValueError(
            "No frontal channels"
        )

    if len(occipital) == 0:
        raise ValueError(
            "No occipital channels"
        )

    if len(temporal) == 0:
        raise ValueError(
            "No temporal channels"
        )

    # -------------------------
    # REGION POWERS
    # -------------------------

    occ_alpha = np.mean([

        bandpower(
            ch,
            sfreq,
            "alpha"
        )

        for ch in occipital

    ])

    global_alpha = np.mean([

        bandpower(
            ch,
            sfreq,
            "alpha"
        )

        for ch in eeg

    ])

    occ_entropy = np.mean([

        spectral_entropy(
            ch,
            sfreq
        )

        for ch in occipital

    ])

    frontal_theta = np.mean([

        bandpower(
            ch,
            sfreq,
            "theta"
        )

        for ch in frontal

    ])

    frontal_alpha = np.mean([

        bandpower(
            ch,
            sfreq,
            "alpha"
        )

        for ch in frontal

    ])

    frontal_beta = np.mean([

        bandpower(
            ch,
            sfreq,
            "beta"
        )

        for ch in frontal

    ])

    frontal_delta = np.mean([

        bandpower(
            ch,
            sfreq,
            "delta"
        )

        for ch in frontal

    ])

    frontal_entropy = np.mean([

        spectral_entropy(
            ch,
            sfreq
        )

        for ch in frontal

    ])

    temp_theta = np.mean([

        bandpower(
            ch,
            sfreq,
            "theta"
        )

        for ch in temporal

    ])

    temp_alpha = np.mean([

        bandpower(
            ch,
            sfreq,
            "alpha"
        )

        for ch in temporal

    ])

    # -------------------------
    # CLINICAL FEATURES
    # -------------------------

    cdi = (

        frontal_delta +
        frontal_theta

    ) / (

        frontal_alpha +
        frontal_beta +
        1e-6
    )

    theta_beta_ratio = (

        frontal_theta /
        (frontal_beta + 1e-6)

    )

    memory_ratio = (

        temp_theta /
        (temp_alpha + 1e-6)

    )

    gamma_global = np.mean([

        bandpower(
            ch,
            sfreq,
            "gamma"
        )

        for ch in eeg

    ])

    total_power = np.mean([

        bandpower(
            ch,
            sfreq,
            band
        )

        for ch in eeg

        for band in
        [
            "delta",
            "theta",
            "alpha",
            "beta",
            "gamma"
        ]
    ])

    gamma_ratio = (

        gamma_global /
        (total_power + 1e-6)

    )

    return {

        "posterior_dominance_index":
            occ_alpha /
            (global_alpha + 1e-6),

        "occipital_entropy":
            occ_entropy,

        "alpha_peak_gradient":
            (
                occ_alpha -
                frontal_alpha
            ) /
            (
                occ_alpha +
                frontal_alpha +
                1e-6
            ),

        "entropy_gradient":
            frontal_entropy -
            occ_entropy,

        "theta_alpha_ratio_frontal":
            np.clip(
                frontal_theta /
                (frontal_alpha + 1e-6),
                0,
                10
            ),

        "cognitive_decline_index":
            cdi,

        "frontal_theta_beta_ratio":
            theta_beta_ratio,

        "memory_theta_alpha_ratio":
            memory_ratio,

        "gamma_activity_ratio":
            gamma_ratio
    }

# =====================================================
# PIPELINE EXECUTION (DISABLED FOR API SAFETY)
# =====================================================

if __name__ == "__main__":
    print("Standalone biomarker pipeline disabled in API mode.")
