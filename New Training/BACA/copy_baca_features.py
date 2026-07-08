import boto3

bucket = "dementia-research2025"

src_prefix = "lead_pipeline_dementia/REEG-BACA-19/Feature/"
dst_prefix = "New-Training-DB/REEG-BACA-19/Feature/"

s3 = boto3.client("s3")

with open("BACA/keep_subjects.txt") as f:
    keep = [line.strip() for line in f]

total = len(keep)

for i, num in enumerate(keep, start=1):

    filename = f"feature_{num}.npy"

    print(f"[{i}/{total}] {filename}")

    s3.copy_object(
        Bucket=bucket,
        CopySource={
            "Bucket": bucket,
            "Key": src_prefix + filename
        },
        Key=dst_prefix + filename
    )

print("DONE")