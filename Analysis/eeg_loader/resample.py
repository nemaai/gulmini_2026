import numpy as np
from scipy.signal import resample_poly

TARGET_SFREQ = 256


def resample_record(data, sfreq):

    if sfreq is None:
        return data, 256

    sfreq = float(sfreq)

    if sfreq == TARGET_SFREQ:
        return data.astype(np.float32), TARGET_SFREQ

    data = resample_poly(
        data,
        TARGET_SFREQ,
        int(sfreq),
        axis=1
    )

    return data.astype(np.float32), TARGET_SFREQ