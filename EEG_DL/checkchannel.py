import boto3
import numpy as np
import io

BUCKET = "dementia-research2025"

PREFIXES = [
    "New EEG Base/EEG/",
    "lead_pipeline_dementia/REEG-BACA-19/Feature/"
]

s3 = boto3.client(
    's3',
    aws_access_key_id=' ',
    aws_secret_access_key='  ',
    region_name='ap-south-1')

files = []

for prefix in PREFIXES:

    print("Scanning:", prefix)

    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=prefix
    )

    if "Contents" not in response:
        continue

    for obj in response["Contents"]:

        key = obj["Key"]

        if key.endswith(".npy"):
            files.append(key)

print("\nTotal files found:", len(files))

count_19 = 0
count_128 = 0
count_other = 0

for key in files:

    obj = s3.get_object(Bucket=BUCKET, Key=key)

    data = np.load(io.BytesIO(obj["Body"].read()))

    if data.shape[2] == 19:
        count_19 += 1

    elif data.shape[2] == 128:
        count_128 += 1

    else:
        count_other += 1

print("\nChannel distribution:")
print("19-channel files:", count_19)
print("128-channel files:", count_128)
print("Other:", count_other)