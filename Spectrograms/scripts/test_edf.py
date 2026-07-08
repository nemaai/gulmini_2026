import os
import logging
import mne

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
            os.path.join(LOG_DIR, "test_edf.log")
        ),
        logging.StreamHandler()
    ]
)

# ==========================================
# CONFIG
# ==========================================

EDF_PATH = "Spectrogram/edf_inputs/01.edf"

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

    logging.info(f"Channels: {raw.ch_names}")

    logging.info(
        f"Sampling Frequency: "
        f"{raw.info['sfreq']}"
    )

except Exception as e:

    logging.error(str(e))