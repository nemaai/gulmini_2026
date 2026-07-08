import numpy as np
import pandas as pd
import io

import boto3

import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix
)

# =====================================================
# CONFIG
# =====================================================

BUCKET = "dementia-research2025"

MODEL_PATH = "/home/ubuntu/training/models/eeg_dl_model_v3.pt"

TEST_CSV = "/home/ubuntu/training/outputs/dl_test_manifest.csv"

# =====================================================
# MODEL
# =====================================================

class EEGNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(19, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 64, 3, padding=1)

        self.dropout = nn.Dropout(0.4)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64, 1)

    def forward(self, x):

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

def normalize(arr):

    mean = arr.mean(
        axis=2,
        keepdims=True
    )

    std = arr.std(
        axis=2,
        keepdims=True
    )

    arr = (
        arr - mean
    ) / (
        std + 1e-6
    )

    arr = np.clip(
        arr,
        -5,
        5
    )

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
# LOAD TEST
# =====================================================

df = pd.read_csv(TEST_CSV)

y_true = []
y_prob = []

# =====================================================
# PREDICT
# =====================================================

with torch.no_grad():

    for _, row in df.iterrows():

        try:

            X = load_s3_npy(
                row["file_source"]
            )

            if X is None:
                continue

            X = normalize(X)

            X = torch.tensor(
                X,
                dtype=torch.float32
            )

            logits = model(X)

            probs = torch.sigmoid(
                logits
            )

            file_prob = float(
                probs.mean()
            )

            y_true.append(
                int(row["binary_label"])
            )

            y_prob.append(
                file_prob
            )

        except Exception as e:
            print("FAILED:", row["file_source"])
            print(str(e))

# =====================================================
# BASIC STATS
# =====================================================

print("\n========================")
print("PROBABILITY STATS")
print("========================")

print("Min :", np.min(y_prob))
print("Max :", np.max(y_prob))
print("Mean:", np.mean(y_prob))

# =====================================================
# AUC
# =====================================================

auc = roc_auc_score(
    y_true,
    y_prob
)

print("\nAUC =", round(auc,4))

# =====================================================
# BEST THRESHOLD
# =====================================================

fpr, tpr, thresholds = roc_curve(
    y_true,
    y_prob
)

best_idx = np.argmax(
    tpr - fpr
)

best_threshold = thresholds[
    best_idx
]

print(
    "\nBest Threshold =",
    round(float(best_threshold),4)
)

# =====================================================
# EVALUATE BEST THRESHOLD
# =====================================================

preds = (
    np.array(y_prob) >= best_threshold
).astype(int)

acc = accuracy_score(
    y_true,
    preds
)

cm = confusion_matrix(
    y_true,
    preds
)

tn, fp, fn, tp = cm.ravel()

sen = tp / (tp + fn)
spec = tn / (tn + fp)

print("\n========================")
print("OPTIMAL THRESHOLD RESULTS")
print("========================")

print("ACC :", round(acc,4))
print("SEN :", round(sen,4))
print("SPEC:", round(spec,4))

print("\nCM")
print(cm)