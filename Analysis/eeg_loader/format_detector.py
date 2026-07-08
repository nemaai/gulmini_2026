from pathlib import Path


def detect_format(filepath):

    ext = Path(filepath).suffix.lower()

    mapping = {
        ".edf": "EDF",
        ".bdf": "BDF",
        ".set": "EEGLAB",
        ".vhdr": "BRAINVISION",
        ".eeg": "EEG_BINARY",
        ".fif": "FIF",
        ".mat": "MAT",
        ".npy": "NPY"
    }

    return mapping.get(ext, "UNKNOWN")