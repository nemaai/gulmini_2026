import numpy as np

from eeg_loader.universal_loader import (
    load_eeg
)

from eeg_loader.standardize import (
    standardize_record
)

from eeg_loader.resample import (
    resample_record
)

from eeg_loader.filtering import basic_filter

def process_record(filepath):

    record = load_eeg(
        filepath
    )

    record = standardize_record(
        record
    )

    sfreq = record.sfreq

    data, sfreq = resample_record(
        record.data,
        sfreq
    )
    data = basic_filter(
        data,
        sfreq
    )

    return {

        "eeg": data,

        "sfreq": sfreq,

        "channels": record.channels,

        "format": record.format
    }