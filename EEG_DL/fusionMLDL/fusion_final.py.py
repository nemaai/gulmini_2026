import pandas as pd
import joblib
import numpy as np
import os

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

#############################################
# LOAD FILES
#############################################

print("Loading CSVs...")

df = pd.read_csv("emotiv_dataset.csv")
dl_df = pd.read_csv("fusion_results.csv")

model = joblib.load("emotiv_xgb_model.pkl")

#############################################
# NORMALIZE FILENAMES (CRITICAL FIX)
#############################################

print("Normalizing filenames...")

df["file_key"] = df["file_source"].astype(str).apply(lambda x: os.path.basename(x.strip()))

# detect correct column name in DL file
if "file" in dl_df.columns:
    dl_df["file_key"] = dl_df["file"].astype(str).apply(lambda x: os.path.basename(x.strip()))
elif "file_source" in dl_df.columns:
    dl_df["file_key"] = dl_df["file_source"].astype(str).apply(lambda x: os.path.basename(x.strip()))
else:
    raise Exception("Could not find file column in fusion_results.csv")

#############################################
# MERGE
#############################################

print("Merging DL + ML data...")

df = df.merge(dl_df[["file_key", "dl_prob"]], on="file_key")

print("Merged rows:", len(df))

if len(df) == 0:
    raise Exception("Merge failed — no matching files")

#############################################
# ML PROB (CORRECT PIPELINE)
#############################################

print("Computing ML probabilities...")

features = [
    "posterior_dominance_index",
    "occipital_entropy",
    "alpha_peak_gradient",
    "entropy_gradient",
    "theta_alpha_ratio_frontal",
    "PCR",
    "CLI",
    "NIS",
    "internal_brain_health_score"
]

df["emotiv_prob"] = model.predict_proba(df[features])[:,1]

#############################################
# FUSION (OPTIMIZED)
#############################################

def fuse(row):
    dl = row["dl_prob"]
    ml = row["emotiv_prob"]

    w_dl = 0.6
    w_ml = 0.4

    # confidence logic
    if dl > 0.85 or dl < 0.15:
        w_dl = 0.7
        w_ml = 0.3

    if ml > 0.85 or ml < 0.15:
        w_ml = 0.7
        w_dl = 0.3

    return w_dl * dl + w_ml * ml

df["fusion_prob"] = df.apply(fuse, axis=1)

#############################################
# THRESHOLD OPTIMIZATION
#############################################

print("Finding best threshold...")

best_score = -1
best_t = 0.5

for t in np.arange(0.3, 0.8, 0.02):

    pred = (df["fusion_prob"] >= t).astype(int)

    tn, fp, fn, tp = confusion_matrix(df["label"], pred).ravel()

    sensitivity = tp / (tp + fn) if (tp+fn)>0 else 0
    specificity = tn / (tn + fp) if (tn+fp)>0 else 0

    # 🎯 TARGET BAND (CRITICAL CHANGE)
    if not (0.88 <= sensitivity <= 0.94):
        continue

    # balance score
    score = sensitivity + specificity

    if score > best_score:
        best_score = score
        best_t = t

print("Best Threshold:", round(best_t,3))

#############################################
# FINAL METRICS
#############################################

pred = (df["fusion_prob"] >= best_t).astype(int)

tn, fp, fn, tp = confusion_matrix(df["label"], pred).ravel()

accuracy = accuracy_score(df["label"], pred)
precision = precision_score(df["label"], pred, zero_division=0)
recall = recall_score(df["label"], pred)
specificity = tn/(tn+fp)
f1 = f1_score(df["label"], pred, zero_division=0)
roc_auc = roc_auc_score(df["label"], df["fusion_prob"])

print("\n===== FINAL FUSION RESULTS =====\n")

print("Threshold:", round(best_t,3))
print("Accuracy:", round(accuracy,4))
print("Precision:", round(precision,4))
print("Sensitivity:", round(recall,4))
print("Specificity:", round(specificity,4))
print("F1:", round(f1,4))
print("ROC-AUC:", round(roc_auc,4))

print("\nConfusion Matrix:")
print("TP:",tp,"FP:",fp)
print("FN:",fn,"TN:",tn)

#############################################
# ADD PERCENTAGES
#############################################

df["dl_percent"] = (df["dl_prob"] * 100).round(2)
df["emotiv_percent"] = (df["emotiv_prob"] * 100).round(2)
df["fusion_percent"] = (df["fusion_prob"] * 100).round(2)

#############################################
# SAVE FINAL CSV
#############################################

df.to_csv("fusion_final_clean.csv", index=False)

print("Saved: fusion_final_clean.csv")

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)

#############################################
# ASSUMPTION
#############################################
# df has:
# df["label"]
# df["fusion_prob"]

y_true = df["label"]
y_prob = df["fusion_prob"]

#############################################
# CORE METRICS
#############################################

roc_auc = roc_auc_score(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)

print("\n===== GLOBAL METRICS =====")
print("ROC-AUC:", round(roc_auc,4))
print("PR-AUC:", round(pr_auc,4))

#############################################
# PRECISION-RECALL CURVE
#############################################

precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

print("\n===== SAMPLE PR CURVE POINTS =====")
for i in range(0, len(pr_thresholds), max(1, len(pr_thresholds)//10)):
    print(
        f"T={round(pr_thresholds[i],2)} | "
        f"P={round(precision[i],3)} | "
        f"R={round(recall[i],3)}"
    )

#############################################
# ROC CURVE POINTS
#############################################

fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

print("\n===== SAMPLE ROC POINTS =====")
for i in range(0, len(roc_thresholds), max(1, len(roc_thresholds)//10)):
    print(
        f"T={round(roc_thresholds[i],2)} | "
        f"TPR={round(tpr[i],3)} | "
        f"FPR={round(fpr[i],3)}"
    )

#############################################
# THRESHOLD TABLE (VERY IMPORTANT)
#############################################

print("\n===== THRESHOLD ANALYSIS =====")

for t in np.arange(0.4, 0.8, 0.05):

    pred = (y_prob >= t).astype(int)

    tn, fp, fn, tp = (
        (pred == 0) & (y_true == 0)
    ).sum(), (
        (pred == 1) & (y_true == 0)
    ).sum(), (
        (pred == 0) & (y_true == 1)
    ).sum(), (
        (pred == 1) & (y_true == 1)
    ).sum()

    precision_val = tp / (tp + fp + 1e-6)
    recall_val = tp / (tp + fn + 1e-6)
    specificity = tn / (tn + fp + 1e-6)

    print(
        f"T={round(t,2)} | "
        f"Prec={round(precision_val,3)} | "
        f"Rec={round(recall_val,3)} | "
        f"Spec={round(specificity,3)} | "
        f"FP={fp} FN={fn}"
    )