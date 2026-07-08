import boto3
import numpy as np
import io
import torch
import torch.nn as nn
import pandas as pd
import os
import logging
import sys
import joblib

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

from biomarkers_apiclean import compute_biomarkers, stabilize_signal
from scores_apiclean import compute_scores_from_row

#############################################
# LOGGING
#############################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("fusion_evaluation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.info

#############################################
# CONFIG
#############################################

MODEL_DL = "eeg_dl_model_v2.pt"
MODEL_EMOTIV = "emotiv_xgb_model.pkl"
CSV_PATH = "FINAL_XGB_UPDATED.csv"
OUTPUT_CSV = "fusion_results.csv"

BUCKET = "dementia-research2025"

PREFIXES = [
    "New EEG Base/EEG/",
    "lead_pipeline_dementia/REEG-BACA-19/Feature/"
]

MAX_WINDOWS = 50

#############################################
# LOAD MODELS
#############################################

log("Loading models...")

emotiv_model = joblib.load(MODEL_EMOTIV)

class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(19,32,5,padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32,64,5,padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64,64,3,padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(64,1)

    def forward(self,x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

device = torch.device("cpu")

dl_model = EEGNet()
dl_model.load_state_dict(torch.load(MODEL_DL, map_location=device))
dl_model.eval()

#############################################
# LABELS
#############################################

log("Loading labels...")

df = pd.read_csv(CSV_PATH)

label_map = {}
for _, row in df.iterrows():
    fname = os.path.basename(row["file"])
    label_map[fname] = int(row["Labels"])

#############################################
# S3
#############################################

s3 = boto3.client(
    's3',
    aws_access_key_id=' ',
    aws_secret_access_key='  ',
    region_name='ap-south-1')

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
    return np.clip((X-mean)/(std+1e-6), -5, 5)

#############################################
# EMOTIV FEATURE PIPELINE (FIXED)
#############################################

def compute_emotiv_features(eeg, sfreq=200):

    def convert_window(window):
        FP1,FP2,F7,F3,FZ,F4,F8,T3,C3,CZ,C4,T4,T5,P3,PZ,P4,T6,O1,O2 = range(19)

        AF3 = (window[:,FP1] + window[:,F3]) / 2
        AF4 = (window[:,FP2] + window[:,F4]) / 2
        FC5 = (window[:,F3] + window[:,C3]) / 2
        FC6 = (window[:,F4] + window[:,C4]) / 2

        return np.stack([
            AF3, window[:,F7], window[:,F3], FC5,
            window[:,T3], window[:,T5],
            window[:,O1], window[:,O2],
            window[:,T6], window[:,T4],
            FC6, window[:,F4], window[:,F8], AF4
        ], axis=1)

    def to_regions(window):
        AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4 = range(14)

        frontal = np.mean(window[:,[AF3,F7,F3,FC5,FC6,F4,F8,AF4]], axis=1)
        parietal = np.mean(window[:,[P7,P8]], axis=1)
        occipital = np.mean(window[:,[O1,O2]], axis=1)
        temporal = np.mean(window[:,[T7,T8]], axis=1)
        central = (frontal + parietal)/2

        return np.stack([frontal, central, parietal, occipital, temporal], axis=1)

    rows = []

    for window in eeg:

        # FIX SHAPE
        if window.shape[0] > window.shape[1]:
            window = window.T

        # APPLY STABILIZATION PER WINDOW
        window, sfreq = stabilize_signal(window, sfreq)

        emotiv = convert_window(window.T)
        regions = to_regions(emotiv)

        biomarkers = compute_biomarkers(regions.T, sfreq)
        scores = compute_scores_from_row(biomarkers)

        rows.append({**biomarkers, **scores})

    df = pd.DataFrame(rows)
    final = df.mean().to_dict()

    return [
        final["posterior_dominance_index"],
        final["occipital_entropy"],
        final["alpha_peak_gradient"],
        final["entropy_gradient"],
        final["theta_alpha_ratio_frontal"],
        final["PCR"],
        final["CLI"],
        final["NIS"],
        final["internal_brain_health_score"]
    ]

#############################################
# FUSION
#############################################

def fuse(dl_prob, emotiv_prob):

    w_emotiv = 0.7
    w_dl = 0.3

    if dl_prob > 0.85 or dl_prob < 0.15:
        w_dl = 0.4
        w_emotiv = 0.6

    if emotiv_prob > 0.85 or emotiv_prob < 0.15:
        w_emotiv = 0.8
        w_dl = 0.2

    return w_emotiv * emotiv_prob + w_dl * dl_prob

#############################################
# INIT CSV
#############################################

pd.DataFrame(columns=[
    "file","true_label",
    "dl_prob","dl_percent",
    "emotiv_prob","emotiv_percent",
    "fusion_prob","fusion_percent"
]).to_csv(OUTPUT_CSV,index=False)

#############################################
# RUN
#############################################

log("Starting evaluation...")

files = list_files()

y_true, y_dl, y_emotiv, y_fusion = [], [], [], []

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

    # DL
    X = torch.tensor(normalize(data)).float()

    with torch.no_grad():
        probs = torch.sigmoid(dl_model(X)).numpy().flatten()

    dl_prob = np.mean(np.sort(probs)[-10:])

    # EMOTIV
    features = compute_emotiv_features(data)
    emotiv_prob = emotiv_model.predict_proba([features])[0][1]

    # FUSION
    fusion_prob = fuse(dl_prob, emotiv_prob)

    y_true.append(label)
    y_dl.append(dl_prob)
    y_emotiv.append(emotiv_prob)
    y_fusion.append(fusion_prob)

    pd.DataFrame([{
        "file": fname,
        "true_label": label,

        "dl_prob": dl_prob,
        "dl_percent": round(dl_prob*100,2),

        "emotiv_prob": emotiv_prob,
        "emotiv_percent": round(emotiv_prob*100,2),

        "fusion_prob": fusion_prob,
        "fusion_percent": round(fusion_prob*100,2)

    }]).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

    if i % 50 == 0:
        log(f"Processed {i}")

#############################################
# METRICS
#############################################

def evaluate(y_true, y_prob, name):

    y_pred = [1 if p > 0.5 else 0 for p in y_prob]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    log(f"\n===== {name} =====")
    log(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    log(f"Precision: {precision_score(y_true, y_pred)}")
    log(f"Sensitivity: {recall_score(y_true, y_pred)}")
    log(f"Specificity: {tn/(tn+fp)}")
    log(f"F1: {f1_score(y_true, y_pred)}")
    log(f"ROC-AUC: {roc_auc_score(y_true, y_prob)}")

evaluate(y_true, y_dl, "DL")
evaluate(y_true, y_emotiv, "EMOTIV")
evaluate(y_true, y_fusion, "FUSION")

log(" DONE")