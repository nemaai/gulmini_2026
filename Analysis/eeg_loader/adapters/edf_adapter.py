import mne

from eeg_loader.eeg_record import EEGRecord


def load_edf(filepath):

    raw = mne.io.read_raw_edf(
        filepath,
        preload=True,
        verbose=False
    )

    return EEGRecord(
        source_file=filepath,
        format="EDF",
        sfreq=raw.info["sfreq"],
        channels=raw.ch_names,
        data=raw
    )