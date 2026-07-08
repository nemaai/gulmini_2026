import mne

from eeg_loader.eeg_record import EEGRecord


def load_brainvision(filepath):

    raw = mne.io.read_raw_brainvision(
        filepath,
        preload=True,
        verbose=False
    )

    return EEGRecord(
        source_file=filepath,
        format="BRAINVISION",
        sfreq=raw.info["sfreq"],
        channels=raw.ch_names,
        data=raw
    )