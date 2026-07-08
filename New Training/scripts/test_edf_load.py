import boto3
import tempfile

from eeg_utils import load_eeg

s3 = boto3.client("s3")

bucket = "dementia-research2025"

key = "lead_pipeline_dementia/ds005385/sub-001/ses-1/eeg/sub-001_ses-1_task-EyesClosed_acq-pre_eeg.edf"

tmp = tempfile.NamedTemporaryFile(
    suffix=".edf",
    delete=False
)

tmp.close()

print("Downloading...")

s3.download_file(
    bucket,
    key,
    tmp.name
)

print("Downloaded:", tmp.name)

raw_eeg, sfreq, channel_names = load_eeg(tmp.name)

print("Shape:", raw_eeg.shape)
print("SFREQ:", sfreq)
print("Channels:", len(channel_names))