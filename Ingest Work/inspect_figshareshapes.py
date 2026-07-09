from pathlib import Path
import tempfile

from scipy.io import loadmat

# Import the same helpers from ingest_all_datasets.py
from ingest_all_datasets import (
    BUCKET,
    S3_SRC_ROOT,
    s3_list_keys,
    s3_download,
)

src_pfx = f"{S3_SRC_ROOT}/Figshare AD Cohort B "

counts = {}

for key in s3_list_keys(bucket=BUCKET, prefix=src_pfx):

    if not key.endswith(".mat"):
        continue

    try:
        with tempfile.TemporaryDirectory() as tmp:

            local = Path(tmp) / Path(key).name

            s3_download(
                BUCKET,
                key,
                local,
            )

            mat = loadmat(local)

            x = mat["export"]

            shape = x.shape

            counts[shape[1]] = counts.get(shape[1], 0) + 1

            print(f"{shape}   {key}")

    except Exception as e:

        print(f"ERROR: {key}")
        print(e)

print("\n========================")
print("Channel Counts")
print("========================")

for k in sorted(counts):
    print(f"{k} channels : {counts[k]} files")