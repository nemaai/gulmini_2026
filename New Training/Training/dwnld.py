import boto3
import pandas as pd
import io

s3 = boto3.client("s3")

obj = s3.get_object(
    Bucket="dementia-research2025",
    Key="lead_pipeline_dementia/nigar_eeg/Final dataset for the published paper/2-Mild/Mild1.csv"
)

df = pd.read_csv(
    io.BytesIO(obj["Body"].read()),
    nrows=1000
)

print(df.iloc[:, :19].shape)

print(df.iloc[:, :19].describe())