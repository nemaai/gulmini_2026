import numpy as np

from eeg_loader.eeg_record import EEGRecord


def load_geneeg(filepath):

    data = np.loadtxt(filepath)

    return EEGRecord(
        source_file=filepath,
        format="GENEEG",
        sfreq=None,
        channels=[
            f"CH{i+1}"
            for i in range(data.shape[1])
        ],
        data=data
    )