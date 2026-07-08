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
                "generate_bank.log"
            )
        ),
        logging.StreamHandler()
    ]
)

# ==========================================
# CONFIG
# ==========================================

INPUT_DIR = "Spectrogram/edf_inputs/test"

OUTPUT_DIR = "Spectrogram/test_bank/T4"

CHANNEL = "T4"

WINDOW_SEC = 20

STEP_SEC = 10

# ==========================================
# OUTPUT FOLDER
# ==========================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

edf_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".edf")
]

logging.info(
    f"Found {len(edf_files)} EDF files"
)

# ==========================================
# PROCESS FILES
# ==========================================

for edf_name in edf_files:

    try:

        logging.info(
            f"Processing: {edf_name}"
        )

        edf_path = os.path.join(
            INPUT_DIR,
            edf_name
        )

        raw = mne.io.read_raw_edf(
            edf_path,
            preload=True,
            verbose=False
        )

        raw.filter(
            l_freq=0.5,
            h_freq=30,
            verbose=False
        )

        fs = int(raw.info["sfreq"])

        signal = raw.get_data(
            picks=[CHANNEL]
        )[0]

        total_sec = int(len(signal) / fs)

        for start_sec in range(
            0,
            total_sec - WINDOW_SEC,
            STEP_SEC
        ):

            end_sec = (
                start_sec + WINDOW_SEC
            )

            start_sample = (
                start_sec * fs
            )

            end_sample = (
                end_sec * fs
            )

            window_signal = signal[
                start_sample:end_sample
            ]

            # ==================================
            # STFT
            # ==================================

            frequencies, times, Zxx = stft(
                window_signal,
                fs=fs,
                window="hann",
                nperseg=256,
                noverlap=220,
                nfft=512,
                boundary=None
            )

            freq_mask = (
                (frequencies >= 0.5)
                &
                (frequencies <= 45)
            )

            frequencies = frequencies[
                freq_mask
            ]

            Zxx = Zxx[
                freq_mask,
                :
            ]

            magnitude = np.abs(Zxx)

            magnitude_db = (
                20 * np.log10(
                    magnitude + 1e-6
                )
            )

            vmin = np.percentile(
                magnitude_db,
                10
            )

            vmax = np.percentile(
                magnitude_db,
                98
            )

            fig, ax = plt.subplots(
                figsize=(10, 5),
                dpi=200
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

            base_name = os.path.splitext(
                edf_name
            )[0]

            output_name = (
                f"{base_name}"
                f"_T4_{start_sec}s.png"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_name
            )

            plt.savefig(
                output_path,
                bbox_inches="tight",
                pad_inches=0
            )

            plt.close()

            logging.info(
                f"Saved: {output_name}"
            )

    except Exception as e:

        logging.error(
            f"{edf_name} --> {str(e)}"
        )

logging.info("Bank generation complete")