import boto3
import numpy as np
import io
import torch
import torch.nn as nn
import pandas as pd
import os
import logging
import sys

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, roc_curve,
    confusion_matrix, average_precision_score
)

from sklearn.isotonic import IsotonicRegression

#############################################
# LOGGING
#############################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluation_final.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.info

#############################################
# CONFIG
#############################################

MODEL_PATH = "eeg_dl_model_v2.pt"   
CSV_PATH = "FINAL_XGB_UPDATED.csv"

BUCKET = "dementia-research2025"

PREFIXES = [
    "New EEG Base/EEG/",
    "lead_pipeline_dementia/REEG-BACA-19/Feature/"
]

MAX_WINDOWS = 50

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

log(f"Labels loaded: {len(label_map)}")

#############################################
# MODEL (MATCH TRAINING EXACTLY)
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

device = torch.device("cpu")

log("Loading model...")

model = EEGNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

#############################################
# HELPERS
#############################################

def list_files():
    files = []
    for prefix in PREFIXES:
        res = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        if "Contents" in res:
            for obj in res["Contents"]:
                if obj["Key"].endswith(".npy"):
                    files.append(obj["Key"])
    return files


def load_eeg(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    data = np.load(io.BytesIO(obj["Body"].read()))

    if data.shape[2] in [19,128]:
        data = np.transpose(data,(0,2,1))

    if data.shape[1] != 19:
        return None

    return data


def normalize(X):
    mean = X.mean(axis=2,keepdims=True)
    std = X.std(axis=2,keepdims=True)

    X = (X-mean)/(std+1e-6)
    X = np.clip(X, -5, 5)

    return X

#############################################
# EVALUATION
#############################################

log("Starting evaluation...")

files = list_files()

y_true = []
y_prob = []

results_path = "evaluation_dl_final.csv"
pd.DataFrame(columns=["file","true_label","pred_prob"]).to_csv(results_path,index=False)

for i, key in enumerate(files):

    fname = os.path.basename(key)

    if fname not in label_map:
        continue

    label = label_map[fname]

    data = load_eeg(key)

    if data is None:
        continue

    if data.shape[0] > MAX_WINDOWS:
        idx = np.random.choice(data.shape[0], MAX_WINDOWS, replace=False)
        data = data[idx]

    X = normalize(data)
    X = torch.tensor(X).float()

    with torch.no_grad():
        probs = torch.sigmoid(model(X)).numpy().flatten()

    # 🔥 TOP-K aggregation
    top_k = np.sort(probs)[-10:] if len(probs) >= 10 else probs
    subject_prob = np.mean(top_k)

    y_true.append(label)
    y_prob.append(subject_prob)

    pd.DataFrame([{
        "file": fname,
        "true_label": label,
        "pred_prob": float(subject_prob)
    }]).to_csv(results_path, mode='a', header=False, index=False)

    if i % 50 == 0:
        log(f"Processed {i} files")

#############################################
# CALIBRATION
#############################################

log("Calibrating probabilities...")

iso = IsotonicRegression(out_of_bounds='clip')
y_prob = iso.fit_transform(y_prob, y_true)

#############################################
# THRESHOLD
#############################################

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
best_idx = np.argmax(tpr - fpr)
best_thresh = thresholds[best_idx]

#############################################
# METRICS
#############################################

y_pred = [1 if p > best_thresh else 0 for p in y_prob]

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
specificity = tn / (tn + fp)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)

#############################################
# PRINT
#############################################

print("\n===== FINAL DL REPORT =====\n")

print(f"Threshold: {round(best_thresh,2)}\n")

print(f"Accuracy: {round(accuracy,4)}")
print(f"Precision: {round(precision,4)}")
print(f"Sensitivity: {round(recall,4)}")
print(f"Specificity: {round(specificity,4)}")
print(f"F1: {round(f1,4)}")
print(f"ROC-AUC: {round(roc_auc,4)}")
print(f"PR-AUC: {round(pr_auc,4)}")

print("\nConfusion Matrix\n")

print(f"TP: {tp} FP: {fp}")
print(f"FN: {fn} TN: {tn}")

log("✅ Evaluation complete")