import io
import numpy as np
import pandas as pd
import boto3
import torch
import torch.nn as nn

# =====================================================
# CONFIG
# =====================================================

BUCKET = "dementia-research2025"

MODEL_PATH = "/home/ubuntu/training/models/eeg_dl_model_v3.pt"

INPUT_CSV = "/home/ubuntu/training/outputs/fusion_final_clean_v2.csv"

OUTPUT_CSV = "/home/ubuntu/training/outputs/fusion_final_with_dl.csv"

# =====================================================
# MODEL
# =====================================================

class EEGNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(19,32,5,padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32,64,5,padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64,64,3,padding=1)

        self.dropout = nn.Dropout(0.4)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64,1)

    def forward(self,x):

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.conv3(x))

        x = self.dropout(x)

        x = self.pool(x).squeeze(-1)

        return self.fc(x)

# =====================================================
# HELPERS
# =====================================================

s3 = boto3.client("s3")

def normalize(arr):

    mean = arr.mean(axis=2, keepdims=True)
    std = arr.std(axis=2, keepdims=True)

    arr = (arr - mean) / (std + 1e-6)

    arr = np.clip(arr, -5, 5)

    return arr

def load_s3_npy(key):

    obj = s3.get_object(
        Bucket=BUCKET,
        Key=key
    )

    arr = np.load(
        io.BytesIO(
            obj["Body"].read()
        )
    )

    if arr.ndim != 3:
        return None

    if arr.shape[2] == 19:
        arr = np.transpose(
            arr,
            (0,2,1)
        )

    if arr.shape[1] != 19:
        return None

    return arr

# =====================================================
# LOAD MODEL
# =====================================================

device = torch.device("cpu")

model = EEGNet()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

# =====================================================
# LOAD CSV
# =====================================================

df = pd.read_csv(INPUT_CSV)

print("Rows:", len(df))

# =====================================================
# INFERENCE
# =====================================================

dl_probs = []

with torch.no_grad():

    for i, row in df.iterrows():

        try:

            key = row["file_source"]

            arr = load_s3_npy(key)

            if arr is None:
                dl_probs.append(np.nan)
                continue

            arr = normalize(arr)

            X = torch.tensor(
                arr,
                dtype=torch.float32
            )

            probs = torch.sigmoid(
                model(X)
            )

            file_prob = float(
                probs.mean()
            )

            dl_probs.append(file_prob)

            if i % 25 == 0:
                print(
                    f"{i}/{len(df)}"
                )

        except Exception as e:

            print(
                "FAILED:",
                row["file_source"]
            )

            print(str(e))

            dl_probs.append(np.nan)

# =====================================================
# SAVE
# =====================================================

df["dl_probability"] = dl_probs

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print()
print("Saved:")
print(OUTPUT_CSV)

print(df.shape)