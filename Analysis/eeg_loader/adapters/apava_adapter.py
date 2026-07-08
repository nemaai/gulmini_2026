import scipy.io
import numpy as np

from eeg_loader.eeg_record import EEGRecord


def load_apava(filepath):

    mat = scipy.io.loadmat(filepath)

    record = mat["data"][0, 0]

    sfreq = int(
        record["fsample"][0, 0]
    )

    labels = [
        x[0]
        for x in record["label"].flatten()
    ]

    trials = record["trial"]

    merged = []

    for i in range(trials.shape[1]):

        merged.append(
            trials[0, i]
        )

    continuous = np.concatenate(
        merged,
        axis=1
    )

    return EEGRecord(
        source_file=filepath,
        format="APAVA",
        sfreq=sfreq,
        channels=labels,
        data=continuous
    )