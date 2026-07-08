import boto3
import numpy as np
import pandas as pd
import io
import os
import gc

import torch
import torch.nn as nn
import torch.optim as optim

import logging
import sys

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)


# =====================================================
# CONFIG
# =====================================================

BUCKET = "dementia-research2025"

TRAIN_CSV = "/home/ubuntu/training/outputs/dl_train_manifest.csv"
TEST_CSV = "/home/ubuntu/training/outputs/dl_test_manifest.csv"

EPOCHS = 8
LR = 0.001

MODEL_PATH = "/home/ubuntu/training/models/eeg_dl_model_v3.pt"

# =====================================================
# LOGGING
# =====================================================

os.makedirs(
    "/home/ubuntu/training/logs",
    exist_ok=True
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "/home/ubuntu/training/logs/eegnet_v3.log"
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.info

# =====================================================
# S3
# =====================================================

s3 = boto3.client("s3")

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

    # expected:
    # (windows,128,19)

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
# DATA
# =====================================================

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

log(f"Train files: {len(train_df)}")
log(f"Test files : {len(test_df)}")

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

log(f"Device: {device}")

# =====================================================
# CLASS WEIGHT
# =====================================================

pos = (train_df["binary_label"] == 1).sum()
neg = (train_df["binary_label"] == 0).sum()

pos_weight = torch.tensor(
    [neg / pos],
    dtype=torch.float32
).to(device)

# =====================================================
# MODEL
# =====================================================

model = EEGNet().to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)

loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

# =====================================================
# TRAIN
# =====================================================

for epoch in range(EPOCHS):

    model.train()

    losses = []

    for idx, (_, row) in enumerate(train_df.iterrows()):

        if idx % 25 == 0:
            log(
                f"Epoch {epoch+1} | "
                f"{idx}/{len(train_df)} files"
            )

        label = int(row["binary_label"])

        key = row["file_source"]

        try:

            X = load_s3_npy(key)

            if X is None:
                continue

            X = normalize(X)

            y = np.full(
                (X.shape[0],1),
                label
            )

            X = torch.tensor(
                X,
                dtype=torch.float32
            ).to(device)

            y = torch.tensor(
                y,
                dtype=torch.float32
            ).to(device)

            pred = model(X)

            loss = loss_fn(
                pred,
                y
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(
                loss.item()
            )

            del X,y,pred
            gc.collect()

        except Exception:
            continue

    log(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss={np.mean(losses):.4f}"
    )

# =====================================================
# SAVE
# =====================================================

torch.save(
    model.state_dict(),
    MODEL_PATH
)

log(f"Model saved: {MODEL_PATH}")

# =====================================================
# FILE LEVEL EVAL
# =====================================================

model.eval()

y_true = []
y_prob = []

with torch.no_grad():

    for _, row in test_df.iterrows():

        label = int(
            row["binary_label"]
        )

        key = row["file_source"]

        try:

            X = load_s3_npy(key)

            if X is None:
                continue

            X = normalize(X)

            X = torch.tensor(
                X,
                dtype=torch.float32
            ).to(device)

            logits = model(X)

            probs = torch.sigmoid(
                logits
            )

            file_prob = float(
                probs.mean()
            )

            y_true.append(label)
            y_prob.append(file_prob)

        except Exception:
            continue

# =====================================================
# METRICS
# =====================================================

auc = roc_auc_score(
    y_true,
    y_prob
)

preds = (
    np.array(y_prob) >= 0.5
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

sensitivity = tp/(tp+fn)
specificity = tn/(tn+fp)

log("====================")
log("DL RESULTS")
log("====================")

log(f"AUC : {auc:.4f}")
log(f"ACC : {acc:.4f}")
log(f"SEN : {sensitivity:.4f}")
log(f"SPEC: {specificity:.4f}")

log(f"Confusion Matrix:\n{cm}")