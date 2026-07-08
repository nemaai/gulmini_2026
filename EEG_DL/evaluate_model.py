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

#############################################
# LOGGING
#############################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.info

#############################################
# CONFIG
#############################################

MODEL_PATH = "eeg_dl_model.pt"
CSV_PATH = "FINAL_XGB_UPDATED.csv"

BUCKET = "dementia-research2025"

PREFIXES = [
    "New EEG Base/EEG/",
    "lead_pipeline_dementia/REEG-BACA-19/Feature/"
]

MAX_WINDOWS = 50  # fast mode

#############################################
# S3
#############################################

s3 = boto3.client(
    's3',
    aws_access_key_id=' ',
    aws_secret_access_key='  ',
    region_name='ap-south-1')

#############################################
# LOAD LABELS
#############################################

log("Loading labels")

df = pd.read_csv(CSV_PATH)

label_map = {}

for _, row in df.iterrows():
    fname = os.path.basename(row["file"])
    label_map[fname] = int(row["Labels"])

log(f"Labels loaded: {len(label_map)}")

#############################################
# MODEL
#############################################

class EEGNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(19,32,5,padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32,64,5,padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64,1)

    def forward(self,x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

log("Loading model")

device = torch.device("cpu")

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
    return (X - X.mean(axis=2,keepdims=True)) / (X.std(axis=2,keepdims=True)+1e-6)

#############################################
# EVALUATION
#############################################

log("Starting evaluation")

files = list_files()

y_true = []
y_prob = []

results_path = "evaluation_results.csv"

# initialize CSV
pd.DataFrame(columns=["file","true_label","pred_prob"]).to_csv(results_path,index=False)

for i, key in enumerate(files):

    fname = os.path.basename(key)

    if fname not in label_map:
        continue

    label = label_map[fname]

    data = load_eeg(key)

    if data is None:
        continue

    # FAST MODE
    if data.shape[0] > MAX_WINDOWS:
        idx = np.random.choice(data.shape[0],MAX_WINDOWS,replace=False)
        data = data[idx]

    X = normalize(data)
    X = torch.tensor(X).float()

    with torch.no_grad():
        probs = torch.sigmoid(model(X)).numpy()

    subject_prob = np.median(probs)

    y_true.append(label)
    y_prob.append(subject_prob)

    # write row immediately
    pd.DataFrame([{
        "file": fname,
        "true_label": label,
        "pred_prob": float(subject_prob)
    }]).to_csv(results_path, mode='a', header=False, index=False)

    if i % 50 == 0:
        log(f"Processed {i} files")

#############################################
# THRESHOLD
#############################################

log("Finding optimal threshold")

fpr, tpr, thresholds = roc_curve(y_true, y_prob)

best_idx = np.argmax(tpr - fpr)
best_thresh = thresholds[best_idx]

#############################################
# FINAL METRICS
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
# PRINT REPORT
#############################################

print("\n===== FINAL REPORT =====\n")

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

#############################################
# SAVE FINAL CSV
#############################################

df_final = pd.read_csv(results_path)
df_final["pred_label"] = (df_final["pred_prob"] > best_thresh).astype(int)

df_final.to_csv("evaluation_results_final.csv", index=False)

log("Saved final results")