import boto3
import pandas as pd
import numpy as np
import io
import os

BUCKET = "dementia-research2025"

ROOT = (
    "lead_pipeline_dementia/"
    "nigar_eeg/"
    "Final dataset for the published paper/"
)

OUT_DIR = "nigar_npy"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)

s3 = boto3.client("s3")

WINDOW = 128
STEP = 64

folders = {

    "1-Healthy": 0,
    "2-Mild": 1,
    "3-Moderate": 1,
    "4-Sever": 1
}

for folder, label in folders.items():

    print("\nProcessing:", folder)

    paginator = s3.get_paginator(
        "list_objects_v2"
    )

    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=ROOT + folder + "/"
    ):

        for obj in page.get(
            "Contents",
            []
        ):

            key = obj["Key"]

            if not key.endswith(".csv"):
                continue

            filename = os.path.basename(key)

            print(filename)

            try:

                response = s3.get_object(
                    Bucket=BUCKET,
                    Key=key
                )

                df = pd.read_csv(
                    io.BytesIO(
                        response["Body"].read()
                    )
                )

                X = df.values.astype(
                    np.float32
                )

                windows = []

                for start in range(
                    0,
                    len(X) - WINDOW + 1,
                    STEP
                ):

                    win = X[
                        start:
                        start + WINDOW
                    ]

                    windows.append(
                        win
                    )

                windows = np.array(
                    windows,
                    dtype=np.float32
                )

                out_name = (
                    filename
                    .replace(".csv", ".npy")
                )

                np.save(
                    os.path.join(
                        OUT_DIR,
                        out_name
                    ),
                    windows
                )

            except Exception as e:

                print(
                    "FAILED:",
                    filename,
                    e
                )

print("\nDONE")