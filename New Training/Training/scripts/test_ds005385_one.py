import boto3
import tempfile
import os

BUCKET = "dementia-research2025"

KEY = (
    "lead_pipeline_dementia/ds005385/"
    "sub-001/ses-1/eeg/"
    "sub-001_ses-1_task-EyesClosed_acq-pre_eeg.edf"
)

s3 = boto3.client("s3")

tmp = tempfile.NamedTemporaryFile(
    suffix=".edf",
    delete=False
)

tmp.close()

print("Downloading...")

s3.download_file(
    BUCKET,
    KEY,
    tmp.name
)

print("Downloaded:")
print(tmp.name)

# ==========================================
# REPLACE THIS BLOCK WITH YOUR
# WORKING GAMMA SCRIPT PROCESS_FILE()
# ==========================================

from core.eeg_loader import load_eeg

raw_eeg, sfreq, channels = load_eeg(
    tmp.name
)

print()
print("SUCCESS")
print("Shape:", raw_eeg.shape)
print("SFREQ:", sfreq)
print("Channels:", len(channels))

# ==========================================

os.remove(tmp.name)

print()
print("Temp file deleted")
print("DONE")