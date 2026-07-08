import numpy as np

from eeg_loader.eeg_record import EEGRecord


def load_adfsu(filepath):

    data = np.loadtxt(filepath)

    return EEGRecord(
        source_file=filepath,
        format="ADFSU",
        sfreq=None,
        channels=["UNKNOWN"],
        data=data
    )