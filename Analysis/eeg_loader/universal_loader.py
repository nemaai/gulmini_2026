from pathlib import Path

from eeg_loader.adapters.edf_adapter import load_edf
from eeg_loader.adapters.eeglab_adapter import load_eeglab
from eeg_loader.adapters.brainvision_adapter import load_brainvision
from eeg_loader.adapters.apava_adapter import load_apava
from eeg_loader.adapters.geneeg_adapter import load_geneeg
from eeg_loader.adapters.adfsu_adapter import load_adfsu


def load_eeg(filepath):

    ext = Path(filepath).suffix.lower()

    if ext == ".edf":
        return load_edf(filepath)

    elif ext == ".set":
        return load_eeglab(filepath)

    elif ext == ".vhdr":
        return load_brainvision(filepath)

    elif ext == ".mat":
        return load_apava(filepath)

    elif ext == ".eeg":
        return load_geneeg(filepath)

    elif ext == ".txt":
        return load_adfsu(filepath)

    raise ValueError(
        f"Unsupported file type: {ext}"
    )