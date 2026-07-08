import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score,
    brier_score_loss
)

# =====================================================
# LOAD DATA
# =====================================================

CSV_PATH = "/home/ubuntu/training/outputs/fusion_final_with_dl.csv"

df = pd.read_csv(CSV_PATH)

df = df[df["binary_label"] != -1].copy()

FEATURES = [
    "posterior_dominance_index",
    "occipital_entropy",
    "alpha_peak_gradient",
    "entropy_gradient",
    "theta_alpha_ratio_frontal",
    "PCR",
    "CLI",
    "NIS",
    "internal_brain_health_score",
    "dl_probability"
]

X = df[FEATURES]

y = df["binary_label"]

# =====================================================
# SAME SPLIT AS TRAINING
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(
    "/home/ubuntu/training/models/fusion_v4.pkl"
)

# =====================================================
# PREDICTIONS
# =====================================================

preds = model.predict(X_test)

probs = model.predict_proba(X_test)[:,1]

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    preds
)

tn, fp, fn, tp = cm.ravel()

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(y_test, preds)

auc = roc_auc_score(y_test, probs)

pr_auc = average_precision_score(
    y_test,
    probs
)

precision = precision_score(
    y_test,
    preds
)

recall = recall_score(
    y_test,
    preds
)

f1 = f1_score(
    y_test,
    preds
)

specificity = tn / (tn + fp)

npv = tn / (tn + fn)

balanced_acc = balanced_accuracy_score(
    y_test,
    preds
)

mcc = matthews_corrcoef(
    y_test,
    preds
)

kappa = cohen_kappa_score(
    y_test,
    preds
)

brier = brier_score_loss(
    y_test,
    probs
)

# =====================================================
# PRINT
# =====================================================

print("\n====================================")
print("FUSION V4 FULL EVALUATION")
print("====================================")

print("\nDataset")
print("Train:", len(X_train))
print("Test :", len(X_test))

print("\nConfusion Matrix")
print(cm)

print("\nCore Metrics")
print(f"Accuracy           : {accuracy:.4f}")
print(f"AUC ROC            : {auc:.4f}")
print(f"PR AUC             : {pr_auc:.4f}")

print("\nClassification Metrics")
print(f"Precision          : {precision:.4f}")
print(f"Recall/Sensitivity : {recall:.4f}")
print(f"Specificity        : {specificity:.4f}")
print(f"F1 Score           : {f1:.4f}")
print(f"NPV                : {npv:.4f}")

print("\nAgreement Metrics")
print(f"Balanced Accuracy  : {balanced_acc:.4f}")
print(f"MCC                : {mcc:.4f}")
print(f"Cohen Kappa        : {kappa:.4f}")

print("\nCalibration")
print(f"Brier Score        : {brier:.4f}")

print("\nClassification Report")
print(classification_report(
    y_test,
    preds,
    digits=4
))