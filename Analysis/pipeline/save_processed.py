import os
import json
import numpy as np


def save_processed(
    output_dir,
    result,
    quality
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    np.save(

        os.path.join(
            output_dir,
            "eeg.npy"
        ),

        result["eeg"]
    )

    metadata = {

        "sfreq":
            result["sfreq"],

        "channels":
            [str(c) for c in result["channels"]],

        "n_channels":
            int(
                len(
                    result["channels"]
                )
            ),

        "n_samples":
            int(
                result["eeg"].shape[1]
            ),

        "format":
            result["format"]
    }

    with open(

        os.path.join(
            output_dir,
            "metadata.json"
        ),

        "w"

    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    with open(

        os.path.join(
            output_dir,
            "quality.json"
        ),

        "w"

    ) as f:

        json.dump(
            quality,
            f,
            indent=2
        )