import numpy as np

CHANNEL_MAP = {

    "FP1": "Fp1",
    "FP2": "Fp2",

    "FPZ": "Fpz",

    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",

    "T3": "T7",
    "T4": "T8",

    "T5": "P7",
    "T6": "P8",

    "A1": "M1",
    "A2": "M2",

    "OZ": "Oz"
}

DROP_CHANNELS = {

    "EKG",
    "ECG",
    "EMG",
    "PHOTIC",
    "STATUS",
    "MARKER",
    "TRIGGER"
}


def standardize_record(record):

    if hasattr(record.data, "get_data"):

        data = record.data.get_data()

        channels = list(
            record.data.ch_names
        )

    else:

        data = record.data

        channels = list(
            record.channels
        )

    data = np.asarray(
        data,
        dtype=np.float32
    )

    # force channels x samples

    if data.shape[0] > data.shape[1]:

        data = data.T

    keep_idx = []
    new_channels = []

    for i, ch in enumerate(channels):

        ch = str(ch).strip()

        ch_upper = ch.upper()

        if ch_upper.startswith("EXG"):
            continue

        if ch_upper in DROP_CHANNELS:
            continue

        ch = CHANNEL_MAP.get(
            ch_upper,
            ch
        )

        keep_idx.append(i)

        new_channels.append(ch)

    data = data[keep_idx]

    # convert to microvolts

    try:

        scale = np.nanpercentile(
            np.abs(data),
            95
        )

        if scale < 1:
            data = data * 1e6

    except:
        pass

    record.data = data

    record.channels = new_channels

    return record