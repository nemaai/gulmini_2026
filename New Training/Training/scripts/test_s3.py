import boto3

s3 = boto3.client("s3")

bucket = "dementia-research2025"

resp = s3.list_objects_v2(
    Bucket=bucket,
    Prefix="New-Training-DB/"
)

for obj in resp.get("Contents", [])[:20]:
    print(obj["Key"])