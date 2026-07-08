from core.eeg_quality_v2 import (
    compute_signal_quality
)


def run_quality(result):

    return compute_signal_quality(

        result["eeg"],

        sfreq=result["sfreq"],

        ch_names=result["channels"]
    )