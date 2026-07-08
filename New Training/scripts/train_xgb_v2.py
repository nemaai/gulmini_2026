import json
import joblib
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# LOAD DATA
# =====================================================

CSV_PATH = (
    "/home/ubuntu/training/outputs/"
    "fusion_final_clean_v2.csv"
)

df = pd.read_csv(CSV_PATH)

print("\nOriginal Shape:")
print(df.shape)

# =====================================================
# REMOVE UNKNOWN LABELS
# =====================================================

train_df = df[
    df["binary_label"] != -1
].copy()

print("\nTraining Shape:")
print(train_df.shape)

print("\nBinary Label Distribution:")
print(
    train_df["binary_label"]
    .value_counts()
)

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
    "internal_brain_health_score"
]

TARGET = "binary_label"

X = train_df[FEATURES]
y = train_df[TARGET]

# =====================================================
# SPLIT
# =====================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )
)

print("\nTrain:", X_train.shape)
print("Test :", X_test.shape)

# =====================================================
# MODEL
# =====================================================

model = XGBClassifier(

    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    eval_metric="logloss"
)

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

# =====================================================
# PREDICTIONS
# =====================================================

probabilities = (
    model.predict_proba(X_test)
    [:, 1]
)

predictions = (
    probabilities >= 0.5
).astype(int)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)

tn, fp, fn, tp = cm.ravel()

sensitivity = (
    tp / (tp + fn)
)

specificity = (
    tn / (tn + fp)
)

print("\n==============================")
print("RESULTS")
print("==============================")

print(
    "Accuracy   :",
    round(accuracy, 4)
)

print(
    "AUC        :",
    round(auc, 4)
)

print(
    "Sensitivity:",
    round(sensitivity, 4)
)

print(
    "Specificity:",
    round(specificity, 4)
)

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        predictions
    )
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({

    "feature": FEATURES,

    "importance":
    model.feature_importances_

})

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
)

print("\nFeature Importance")
print(importance)

# =====================================================
# SAVE
# =====================================================

MODEL_PATH = (
    "/home/ubuntu/training/models/"
    "xgb_v2.pkl"
)

METRICS_PATH = (
    "/home/ubuntu/training/metrics/"
    "xgb_v2_metrics.json"
)

IMPORTANCE_PATH = (
    "/home/ubuntu/training/metrics/"
    "xgb_v2_feature_importance.csv"
)

joblib.dump(
    model,
    MODEL_PATH
)

importance.to_csv(
    IMPORTANCE_PATH,
    index=False
)

metrics = {

    "rows":
    int(len(train_df)),

    "accuracy":
    float(accuracy),

    "auc":
    float(auc),

    "sensitivity":
    float(sensitivity),

    "specificity":
    float(specificity),

    "features":
    FEATURES
}

with open(
    METRICS_PATH,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )

print("\n==============================")
print("MODEL SAVED")
print(MODEL_PATH)

print("\nMETRICS SAVED")
print(METRICS_PATH)

print("\nFEATURE IMPORTANCE SAVED")
print(IMPORTANCE_PATH)
print("==============================")