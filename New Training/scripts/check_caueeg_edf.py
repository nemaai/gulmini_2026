import boto3
import pandas as pd

s3 = boto3.client("s3")

df = pd.read_csv(
    "caueeg_manifest.csv"
)

found = 0
missing = []

for key in df["edf_key"]:

    try:

        s3.head_object(
            Bucket="dementia-research2025",
            Key=key
        )

        found += 1

    except:

        missing.append(key)

print("FOUND:", found)
print("MISSING:", len(missing))

if missing:
    print(missing[:20])