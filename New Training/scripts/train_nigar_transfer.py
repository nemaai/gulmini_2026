import os
import gc
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)

# ==========================================
# CONFIG
# ==========================================

TRAIN_CSV = "nigar_train.csv"
TEST_CSV = "nigar_test.csv"

BASE_MODEL = (
    "/home/ubuntu/training/models/"
    "eeg_dl_model_v3.pt"
)

OUT_MODEL = (
    "/home/ubuntu/training/models/"
    "eeg_dl_nigar_transfer.pt"
)

EPOCHS = 10
LR = 1e-4

# ==========================================
# MODEL
# ==========================================

class EEGNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(
            19, 32, 5, padding=2
        )

        self.bn1 = nn.BatchNorm1d(
            32
        )

        self.conv2 = nn.Conv1d(
            32, 64, 5, padding=2
        )

        self.bn2 = nn.BatchNorm1d(
            64
        )

        self.conv3 = nn.Conv1d(
            64, 64, 3, padding=1
        )

        self.dropout = nn.Dropout(
            0.4
        )

        self.pool = nn.AdaptiveAvgPool1d(
            1
        )

        self.fc = nn.Linear(
            64, 1
        )

    def forward(self, x):

        x = torch.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        x = torch.relu(
            self.bn2(
                self.conv2(x)
            )
        )

        x = torch.relu(
            self.conv3(x)
        )

        x = self.dropout(x)

        x = self.pool(x)

        x = x.squeeze(-1)

        return self.fc(x)

# ==========================================
# NORMALIZE
# ==========================================

def normalize(arr):

    arr = np.transpose(
        arr,
        (0,2,1)
    )

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

# ==========================================
# DATA
# ==========================================

train_df = pd.read_csv(
    TRAIN_CSV
)

test_df = pd.read_csv(
    TEST_CSV
)

print(
    "Train:",
    len(train_df)
)

print(
    "Test:",
    len(test_df)
)

# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "DEVICE:",
    device
)

# ==========================================
# MODEL
# ==========================================

model = EEGNet()

state = torch.load(
    BASE_MODEL,
    map_location=device
)

model.load_state_dict(
    state
)

# ==========================================
# FREEZE EARLY LAYERS
# ==========================================

for p in model.conv1.parameters():
    p.requires_grad = False

for p in model.bn1.parameters():
    p.requires_grad = False

for p in model.conv2.parameters():
    p.requires_grad = False

for p in model.bn2.parameters():
    p.requires_grad = False

model = model.to(device)

# ==========================================
# LOSS
# ==========================================

pos = (
    train_df["binary_label"] == 1
).sum()

neg = (
    train_df["binary_label"] == 0
).sum()

pos_weight = torch.tensor(
    [neg / pos],
    dtype=torch.float32
).to(device)

loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = optim.Adam(

    filter(
        lambda p:
        p.requires_grad,

        model.parameters()
    ),

    lr=LR
)

# ==========================================
# TRAIN
# ==========================================

for epoch in range(EPOCHS):

    model.train()

    losses = []

    for _, row in train_df.iterrows():

        arr = np.load(
            row["file_source"]
        )

        arr = normalize(
            arr
        )

        X = torch.tensor(
            arr,
            dtype=torch.float32
        ).to(device)

        y = torch.full(

            (
                arr.shape[0],
                1
            ),

            float(
                row[
                    "binary_label"
                ]
            ),

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

        del X
        del y
        del pred

        gc.collect()

    print(
        f"Epoch {epoch+1}"
        f" Loss="
        f"{np.mean(losses):.4f}"
    )

# ==========================================
# SAVE
# ==========================================

torch.save(
    model.state_dict(),
    OUT_MODEL
)

print(
    "\nSaved:",
    OUT_MODEL
)

# ==========================================
# EVAL
# ==========================================

model.eval()

y_true = []
y_prob = []

with torch.no_grad():

    for _, row in test_df.iterrows():

        arr = np.load(
            row["file_source"]
        )

        arr = normalize(
            arr
        )

        X = torch.tensor(
            arr,
            dtype=torch.float32
        ).to(device)

        logits = model(X)

        probs = torch.sigmoid(
            logits
        )

        file_prob = float(
            probs.mean()
        )

        y_true.append(
            row[
                "binary_label"
            ]
        )

        y_prob.append(
            file_prob
        )

auc = roc_auc_score(
    y_true,
    y_prob
)

preds = (
    np.array(y_prob)
    >= 0.5
).astype(int)

acc = accuracy_score(
    y_true,
    preds
)

cm = confusion_matrix(
    y_true,
    preds
)

print("\n================")
print("TRANSFER RESULTS")
print("================")

print(
    "AUC:",
    round(auc,4)
)

print(
    "ACC:",
    round(acc,4)
)

print(
    "\nCM:"
)

print(cm)