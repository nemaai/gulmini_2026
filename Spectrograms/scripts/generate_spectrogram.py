import os
import logging
import mne
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import stft

# ==========================================
# LOGGING
# ==========================================

LOG_DIR = "Spectrogram/logs"

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(
                LOG_DIR,
                "generate_spectrogram.log"
            )
        ),
        logging.StreamHandler()
    ]
)

# ==========================================
# CONFIG
# ==========================================

EDF_PATH = "Spectrogram/edf_inputs/01.edf"

OUTPUT_DIR = "Spectrogram/generated_specs"

CHANNEL = "T4"

START_SEC = 120

WINDOW_SEC = 20

# ==========================================
# LOAD EDF
# ==========================================

try:

    raw = mne.io.read_raw_edf(
        EDF_PATH,
        preload=True,
        verbose=False
    )

    logging.info(f"Loaded EDF: {EDF_PATH}")

    # ======================================
    # FILTER
    # ======================================

    raw.filter(
        l_freq=0.5,
        h_freq=30,
        verbose=False
    )

    fs = int(raw.info["sfreq"])

    # ======================================
    # SIGNAL
    # ======================================

    full_signal = raw.get_data(
        picks=[CHANNEL]
    )[0]

    start_sample = int(START_SEC * fs)

    end_sample = int(
        (START_SEC + WINDOW_SEC) * fs
    )

    signal = full_signal[
        start_sample:end_sample
    ]

    # ======================================
    # STFT
    # ======================================

    frequencies, times, Zxx = stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=256,
        noverlap=220,
        nfft=512,
        boundary=None
    )

    # ======================================
    # EEG RANGE
    # ======================================

    freq_mask = (
        (frequencies >= 0.5)
        &
        (frequencies <= 45)
    )

    frequencies = frequencies[freq_mask]

    Zxx = Zxx[freq_mask, :]

    # ======================================
    # MAGNITUDE
    # ======================================

    magnitude = np.abs(Zxx)

    magnitude_db = 20 * np.log10(
        magnitude + 1e-6
    )

    # ======================================
    # NORMALIZATION
    # ======================================

    vmin = np.percentile(
        magnitude_db,
        10
    )

    vmax = np.percentile(
        magnitude_db,
        98
    )

    # ======================================
    # OUTPUT FOLDER
    # ======================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ======================================
    # PLOT
    # ======================================

    fig, ax = plt.subplots(
        figsize=(10, 5),
        dpi=300
    )

    ax.imshow(
        magnitude_db,
        aspect="auto",
        origin="lower",
        cmap="gray",
        extent=[
            times.min(),
            times.max(),
            frequencies.min(),
            frequencies.max()
        ],
        interpolation="bicubic",
        vmin=vmin,
        vmax=vmax
    )

    ax.axis("off")

    plt.tight_layout(pad=0)

    # ======================================
    # SAVE
    # ======================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{CHANNEL}_spectrogram.png"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    logging.info(
        f"Saved spectrogram: {output_path}"
    )

except Exception as e:

    logging.error(str(e))