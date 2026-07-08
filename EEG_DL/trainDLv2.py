import boto3
import numpy as np
import io
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import logging
import sys
import gc
import random

#############################################
# LOGGING
#############################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("training_v2.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.info

#############################################
# CONFIG (OPTIMIZED FOR ~2000 FILES)
#############################################

BUCKET = "dementia-research2025"

PREFIXES = [
    "New EEG Base/EEG/",
    "lead_pipeline_dementia/REEG-BACA-19/Feature/"
]

CSV_PATH = "FINAL_XGB_UPDATED.csv"

EPOCHS = 8                      
BATCH_SIZE = 64
LR = 0.001

MAX_FILES_PER_EPOCH = 200      # sample per epoch
MAX_WINDOWS_PER_FILE = 40      # limit per file

#############################################
# S3
#############################################

s3 = boto3.client(
    's3',
    aws_access_key_id=' ',
    aws_secret_access_key='  ',
    region_name='ap-south-1'
)

#############################################
# LOAD LABELS
#############################################

log("Loading labels...")

df = pd.read_csv(CSV_PATH)

label_map = {}
for _, row in df.iterrows():
    fname = os.path.basename(row["file"])
    label_map[fname] = int(row["Labels"])

pos_count = sum(label_map.values())
neg_count = len(label_map) - pos_count

pos_weight = torch.tensor([neg_count / (pos_count + 1e-6)])

log(f"Labels loaded: {len(label_map)}")
log(f"Class balance → Pos: {pos_count}, Neg: {neg_count}")

#############################################
# MODEL
#############################################

class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(19, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 64, 3, padding=1)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.4)

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.conv3(x))

        x = self.dropout(x)

        x = self.pool(x).squeeze(-1)

        return self.fc(x)

#############################################
# DEVICE
#############################################

device = torch.device("cpu")

model = EEGNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

#############################################
# HELPERS
#############################################

def list_files():
    files = []
    for prefix in PREFIXES:
        log(f"Scanning {prefix}")
        res = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        if "Contents" in res:
            for obj in res["Contents"]:
                if obj["Key"].endswith(".npy"):
                    files.append(obj["Key"])
    return files


def load_eeg(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = np.load(io.BytesIO(obj["Body"].read()))

    # shape fix
    if data.shape[2] in [19, 128]:
        data = np.transpose(data, (0, 2, 1))

    if data.shape[1] != 19:
        return None

    return data


def normalize(X):
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)

    X = (X - mean) / (std + 1e-6)
    X = np.clip(X, -5, 5)

    return X

#############################################
# TRAIN
#############################################

log("Listing files from S3...")

files = list_files()

log(f"Total files found: {len(files)}")

for epoch in range(EPOCHS):

    log(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

    # random subset per epoch
    sampled_files = random.sample(files, min(MAX_FILES_PER_EPOCH, len(files)))

    for i, key in enumerate(sampled_files):

        fname = os.path.basename(key)

        if fname not in label_map:
            continue

        label = label_map[fname]

        data = load_eeg(key)

        if data is None:
            continue

        # limit windows
        if data.shape[0] > MAX_WINDOWS_PER_FILE:
            idx = np.random.choice(data.shape[0], MAX_WINDOWS_PER_FILE, replace=False)
            data = data[idx]

        X = normalize(data)

        y = np.full((X.shape[0], 1), label)

        X = torch.tensor(X).float().to(device)
        y = torch.tensor(y).float().to(device)

        pred = model(X)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            log(f"Processed {i}/{len(sampled_files)} files | Loss: {loss.item():.4f}")

        del X, y, data
        gc.collect()

    log(f"Epoch {epoch+1} completed")

#############################################
# SAVE MODEL
#############################################

torch.save(model.state_dict(), "eeg_dl_model_v2.pt")

log("Model saved: eeg_dl_model_v3.pt")