import pandas as pd
import numpy as np
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

# =====================================================
# LOAD
# =====================================================

CSV_PATH = "/home/ubuntu/training/outputs/fusion_final_with_dl.csv"

df = pd.read_csv(CSV_PATH)

df = df[df["binary_label"] != -1].copy()

print("Dataset:", df.shape)

# =====================================================
# FEATURES
# =====================================================

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
# SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Train:", len(X_train))
print("Test :", len(X_test))

# =====================================================
# MODEL
# =====================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

print("\nTraining Fusion Model...")

model.fit(X_train, y_train)

# =====================================================
# PREDICT
# =====================================================

preds = model.predict(X_test)

probs = model.predict_proba(X_test)[:, 1]

# =====================================================
# METRICS
# =====================================================

acc = accuracy_score(y_test, preds)

auc = roc_auc_score(y_test, probs)

cm = confusion_matrix(y_test, preds)

tn, fp, fn, tp = cm.ravel()

sen = tp / (tp + fn)

spec = tn / (tn + fp)

print("\n==============================")
print("FUSION RESULTS")
print("==============================")

print("Accuracy   :", round(acc,4))
print("AUC        :", round(auc,4))
print("Sensitivity:", round(sen,4))
print("Specificity:", round(spec,4))

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, preds))

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

imp = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

imp = imp.sort_values(
    "importance",
    ascending=False
)

print("\nFeature Importance")
print(imp)

# =====================================================
# SAVE
# =====================================================

MODEL_PATH = "/home/ubuntu/training/models/fusion_v4.pkl"

joblib.dump(
    model,
    MODEL_PATH
)

imp.to_csv(
    "/home/ubuntu/training/metrics/fusion_v4_feature_importance.csv",
    index=False
)

metrics = {
    "accuracy": float(acc),
    "auc": float(auc),
    "sensitivity": float(sen),
    "specificity": float(spec)
}

with open(
    "/home/ubuntu/training/metrics/fusion_v4_metrics.json",
    "w"
) as f:
    json.dump(metrics, f, indent=4)

print("\n==============================")
print("MODEL SAVED")
print(MODEL_PATH)

print("\nMETRICS SAVED")
print("/home/ubuntu/training/metrics/fusion_v4_metrics.json")

print("\nFEATURE IMPORTANCE SAVED")
print("/home/ubuntu/training/metrics/fusion_v4_feature_importance.csv")