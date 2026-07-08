import mne
import numpy as np


def get_info(obj):

    result = {}

    try:

        if isinstance(
            obj,
            mne.io.BaseRaw
        ):

            result["type"] = "MNE_RAW"

            result["channels"] = obj.ch_names

            result["n_channels"] = len(
                obj.ch_names
            )

            result["sampling_rate"] = (
                obj.info["sfreq"]
            )

            result["duration_sec"] = round(
                obj.n_times /
                obj.info["sfreq"],
                2
            )

            return result

    except:
        pass

    if isinstance(obj, dict):

        result["type"] = "DICT"

        result["keys"] = list(
            obj.keys()
        )

        return result

    if isinstance(obj, np.ndarray):

        result["type"] = "NUMPY"

        result["shape"] = obj.shape

        return result

    result["type"] = str(type(obj))

    return result