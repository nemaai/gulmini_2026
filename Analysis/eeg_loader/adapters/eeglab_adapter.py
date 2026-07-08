import mne

from eeg_loader.eeg_record import EEGRecord


def load_eeglab(filepath):

    raw = mne.io.read_raw_eeglab(
        filepath,
        preload=True,
        verbose=False
    )

    return EEGRecord(
        source_file=filepath,
        format="EEGLAB",
        sfreq=raw.info["sfreq"],
        channels=raw.ch_names,
        data=raw
    )