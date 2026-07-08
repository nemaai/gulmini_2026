import boto3
import json
import pandas as pd

BUCKET = "dementia-research2025"

s3 = boto3.client("s3")

obj = s3.get_object(
    Bucket=BUCKET,
    Key="caueeg-dataset/dementia.json"
)

data = json.loads(
    obj["Body"].read()
)

subjects = (
    data["train_split"]
    + data["validation_split"]
    + data["test_split"]
)

rows = []

for row in subjects:

    serial = row["serial"]

    rows.append({
        "serial": serial,
        "age": row["age"],
        "class_name": row["class_name"],
        "class_label": row["class_label"],
        "symptom": ",".join(row["symptom"]),
        "edf_key":
            f"caueeg-dataset/signal/edf/{serial}.edf"
    })

df = pd.DataFrame(rows)

print(df.shape)
print(df.head())

df.to_csv(
    "caueeg_manifest.csv",
    index=False
)

print(
    "Saved:",
    "caueeg_manifest.csv"
)