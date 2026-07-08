import os
import gc
import io
import logging
import sys

import boto3
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)

# =====================================================
# CONFIG
# =====================================================

BASE_MODEL = "../models/base/eeg_dl_model_v2.pt"

MANIFEST = "../manifests/nigar_manifest.csv"

OUTPUT_MODEL = (
    "../models/finetuned/eeg_dl_nigar_v1.pt"
)

EPOCHS = 5
LR = 1e-4

BUCKET = "dementia-research2025"

# =====================================================
# LOGGING
# =====================================================

os.makedirs("../logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(
            "../logs/finetune_eegnet.log"
        ),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.info

# =====================================================
# MODEL
# =====================================================

class EEGNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            19,32,5,padding=2
        )
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(
            32,64,5,padding=2
        )
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(
            64,64,3,padding=1
        )

        self.dropout = nn.Dropout(0.4)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64,1)

    def forward(self,x):

        x = torch.relu(
            self.bn1(self.conv1(x))
        )

        x = torch.relu(
            self.bn2(self.conv2(x))
        )

        x = torch.relu(
            self.conv3(x)
        )

        x = self.dropout(x)

        x = self.pool(x).squeeze(-1)

        return self.fc(x)

# =====================================================
# HELPERS
# =====================================================

def normalize(x):

    mean = x.mean(
        axis=2,
        keepdims=True
    )

    std = x.std(
        axis=2,
        keepdims=True
    )

    x = (
        x - mean
    ) / (
        std + 1e-6
    )

    x = np.clip(x,-5,5)

    return x

def load_npy(key):

    if key.startswith("s3://"):
        raise ValueError(
            "Add S3 parsing if needed"
        )

    arr = np.load(key)

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
# LOAD DATA
# =====================================================

df = pd.read_csv(MANIFEST)

df = df[df["binary_label"] != -1]

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["binary_label"],
    random_state=42
)

log(f"Train={len(train_df)}")
log(f"Test={len(test_df)}")

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

log(f"Device={device}")

# =====================================================
# LOAD MODEL
# =====================================================

model = EEGNet()

model.load_state_dict(
    torch.load(
        BASE_MODEL,
        map_location=device
    )
)

# =====================================================
# FREEZE EARLY LAYERS
# =====================================================

for p in model.conv1.parameters():
    p.requires_grad = False

for p in model.conv2.parameters():
    p.requires_grad = False

model = model.to(device)

# =====================================================
# LOSS
# =====================================================

pos = (
    train_df["binary_label"] == 1
).sum()

neg = (
    train_df["binary_label"] == 0
).sum()

pos_weight = torch.tensor(
    [neg/(pos+1e-6)],
    dtype=torch.float32
).to(device)

loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LR
)

# =====================================================
# TRAIN
# =====================================================

for epoch in range(EPOCHS):

    model.train()

    losses = []

    for idx, (_, row) in enumerate(
        train_df.iterrows()
    ):

        try:

            X = load_npy(
                row["file_source"]
            )

            if X is None:
                continue

            X = normalize(X)

            y = np.full(
                (X.shape[0],1),
                int(row["binary_label"])
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

            if idx % 25 == 0:
                log(
                    f"Epoch {epoch+1} "
                    f"{idx}/{len(train_df)}"
                )

            del X,y,pred

            gc.collect()

        except Exception:
            continue

    log(
        f"Epoch {epoch+1} "
        f"Loss={np.mean(losses):.4f}"
    )

# =====================================================
# SAVE
# =====================================================

torch.save(
    model.state_dict(),
    OUTPUT_MODEL
)

log(
    f"Saved model: "
    f"{OUTPUT_MODEL}"
)

# =====================================================
# EVALUATE
# =====================================================

model.eval()

y_true = []
y_prob = []

with torch.no_grad():

    for _, row in test_df.iterrows():

        try:

            X = load_npy(
                row["file_source"]
            )

            if X is None:
                continue

            X = normalize(X)

            X = torch.tensor(
                X,
                dtype=torch.float32
            ).to(device)

            probs = torch.sigmoid(
                model(X)
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

        except Exception:
            continue

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

sen = tp/(tp+fn)
spec = tn/(tn+fp)

log("================================")
log(f"AUC  = {auc:.4f}")
log(f"ACC  = {acc:.4f}")
log(f"SEN  = {sen:.4f}")
log(f"SPEC = {spec:.4f}")
log(f"\nCM=\n{cm}")
log("================================")