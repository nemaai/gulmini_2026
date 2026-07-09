"""
ml_model_train.py
=================
Trains, evaluates, calibrates and saves the XGBoost biomarker model.

Key steps:
  1. Load CSV, filter labels, binarise
  2. 5-fold stratified CV — AUC, sensitivity, specificity, Youden threshold
  3. Train final model on full data
  4. Platt-scale calibration (sigmoid) via nested CV
     → reduces threshold instability from std=0.106 to ~0.03-0.05
  5. Report ECE (Expected Calibration Error) before/after calibration
  6. NIS polarity check — if AUC < 0.5 score is inverted, report corrected
  7. Save:
       MODEL_PATH.json          raw XGBoost (for feature importance / interp)
       CALIBRATED_MODEL_PATH    calibrated wrapper (for fusion + inference)
       METRICS_PATH             full CV + calibration metrics
       IMPORTANCE_PATH          feature importance CSV
"""

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from config import *
# Expected from config:
#   CSV_PATH, FINAL_FEATURES, MODEL_PATH, IMPORTANCE_PATH, METRICS_PATH, local

import os
import warnings
warnings.filterwarnings("ignore")

# =====================================================
# PATHS
# =====================================================

# Force .json extension to avoid XGBoost UBJSON warning
_model_path_json = (
    MODEL_PATH if MODEL_PATH.endswith(".json")
    else MODEL_PATH + ".json"
)

CALIBRATED_MODEL_PATH = os.path.join(
    os.path.dirname(MODEL_PATH),
    "xgb_calibrated.pkl"
)

# =====================================================
# LOAD DATA
# =====================================================
# OUTPUT_CSV = os.path.join(
#     local, "outputs", "biomarkers", "round1_biomarkers.csv"
# )

ML_CSV_PATH = "BIOMARKERS.csv"
df = pd.read_csv(ML_CSV_PATH)

# Train XGBoost on each dataset separately and check AUC on held-out portion
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

print( df["dataset"].unique(), "---")

for dataset in df["dataset"].unique():
    sub = df[df["dataset"] == dataset]
    # if len(sub) < 10: continue
    X = sub[FINAL_FEATURES].values
    y = sub["true_label"].values
    if y.sum() < 2 or (y==0).sum() < 2: continue
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    m = XGBClassifier(n_estimators=100, verbosity=0)
    m.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, m.predict_proba(X_te)[:,1])
    print(f"  {dataset:<20} n={len(sub):4d}  AUC={auc:.3f}")

# Keep only selected datasets--> final training
KEEP_DATASETS = [
    "ADFSU",
    # "DS004504",
    "BrainLat",
    "BACA_train",
    # "test_hardware",
    #"P-ADIC",
    "Isfahan",
    # "APAVA",
    "FIGSHARE-128Hz",
    "FIGSHARE-256Hz",
    # "ADSZ-AD",
    # "CAUEEG",
    # "DS005048",
    # "GENEEG",
]

df = df[df["dataset"].isin(KEEP_DATASETS)].copy()

print("="*60)
print("Training datasets:")
print(df["dataset"].value_counts())
print("="*60)

print("\nOriginal Shape:", df.shape)
print(df["true_label"].value_counts())

# =====================================================
# LABEL FILTERING
# =====================================================

train_df = df[~df["true_label"].isin([4, 5, 6])].copy()
train_df.loc[train_df["true_label"] == 2, "true_label"] = 1

print("\nAfter filtering:")
print(train_df["true_label"].value_counts())
print("Training shape:", train_df.shape)

FEATURES = FINAL_FEATURES
TARGET   = "true_label"

X = train_df[FEATURES]
y = train_df[TARGET]

# =====================================================
# THRESHOLD STRATEGY
#   "youden"      → maximise sensitivity + specificity
#   "sensitivity" → fix sensitivity at TARGET_SENSITIVITY
# =====================================================

THRESHOLD_STRATEGY = "youden"
TARGET_SENSITIVITY = 0.95

# =====================================================
# HELPER: Expected Calibration Error
# =====================================================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    ECE — mean absolute gap between predicted probability
    and actual fraction of positives, weighted by bin size.
    Lower is better. Well-calibrated model ≈ 0.02–0.05.
    """
    bins    = np.linspace(0, 1, n_bins + 1)
    ece     = 0.0
    n       = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc  = y_true[mask].mean()
        ece     += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)

# =====================================================
# 5-FOLD CROSS VALIDATION
# =====================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_aucs       = []
cv_sens       = []
cv_spec       = []
cv_thresholds = []
oof_probs     = np.zeros(len(y))   # for ECE before calibration

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scale_pos_weight = (
        len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    )

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    oof_probs[test_idx] = probabilities

    auc = roc_auc_score(y_test, probabilities)
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)

    if THRESHOLD_STRATEGY == "sensitivity":
        idx            = np.argmin(np.abs(tpr - TARGET_SENSITIVITY))
        best_threshold = float(thresholds[idx])
    else:
        best_threshold = float(thresholds[np.argmax(.7*tpr - .3*fpr)])

    predictions = (probabilities >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    cv_aucs.append(auc)
    cv_sens.append(tp / (tp + fn))
    cv_spec.append(tn / (tn + fp))
    cv_thresholds.append(best_threshold)

    print(f"Fold {fold+1}  AUC={auc:.3f}  "
          f"Sens={cv_sens[-1]:.3f}  Spec={cv_spec[-1]:.3f}  "
          f"Thresh={best_threshold:.3f}")

print("\n======================")
print("5-FOLD CV RESULTS")
print("======================")
print(f"AUC        : {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")
print(f"Sensitivity: {np.mean(cv_sens):.4f} +/- {np.std(cv_sens):.4f}")
print(f"Specificity: {np.mean(cv_spec):.4f} +/- {np.std(cv_spec):.4f}")
print(f"Threshold  : {np.mean(cv_thresholds):.4f} +/- {np.std(cv_thresholds):.4f}")

# =====================================================
# CALIBRATION QUALITY BEFORE CALIBRATION
# =====================================================

ece_before = expected_calibration_error(
    y.values.astype(float), oof_probs
)
print(f"\nECE before calibration : {ece_before:.4f}")
print(f"Threshold std before   : {np.std(cv_thresholds):.4f}  "
      f"(target < 0.05 after calibration)")

# =====================================================
# FINAL MODEL — train on full dataset
# =====================================================

scale_pos_weight = len(y[y == 0]) / len(y[y == 1])

final_xgb = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    random_state=42
)
final_xgb.fit(X, y)

# =====================================================
# PLATT SCALING CALIBRATION
# =====================================================
#
# CalibratedClassifierCV with cv="prefit" fits a sigmoid (Platt scaling)
# on top of the already-trained XGBoost using a held-out calibration set.
# This maps raw XGBoost log-odds to well-calibrated probabilities.
#
# Why not cv=5 here?
#   cv="prefit" is correct when the base model is already fully trained.
#   cv=5 would retrain XGBoost inside each fold — we don't want that,
#   we want to calibrate THIS specific trained model.
#
# We use a 20% held-out calibration split (stratified).
# The calibration set never touches the base model training.

print("\n==============================")
print("PLATT SCALING CALIBRATION")
print("==============================")

X_train_cal, X_cal, y_train_cal, y_cal = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Retrain base model on 80% (so calibration set is truly held-out)
xgb_for_cal = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train_cal[y_train_cal==0]) / len(y_train_cal[y_train_cal==1]),
    eval_metric="auc",
    random_state=42
)
xgb_for_cal.fit(X_train_cal, y_train_cal)

# calibrated_model = CalibratedClassifierCV(
#     xgb_for_cal,
#     cv="prefit",
#     method="sigmoid"   # Platt scaling
# )
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

frozen_model = FrozenEstimator(xgb_for_cal)

calibrated_model = CalibratedClassifierCV(
    estimator=frozen_model,
    method="sigmoid"
)

calibrated_model.fit(X_cal, y_cal)

# ── evaluate calibration quality ─────────────────────────────────────────────

# OOF calibrated probs via 5-fold on full data
cal_cv_auc = []
cal_cv_sens = []
cal_cv_spec = []
cal_oof_probs = np.zeros(len(y))
skf_cal = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
cal_thresholds = []

for fold, (tr_idx, te_idx) in enumerate(skf_cal.split(X, y)):
    # Train base + calibrate
    xb = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=len(y.iloc[tr_idx][y.iloc[tr_idx]==0]) /
                         len(y.iloc[tr_idx][y.iloc[tr_idx]==1]),
        eval_metric="auc", random_state=42
    )
    # Hold 25% of train for calibration fitting
    tr2, cal2 = train_test_split(tr_idx, test_size=0.25,
                                  random_state=42,
                                  stratify=y.iloc[tr_idx])
    xb.fit(X.iloc[tr2], y.iloc[tr2])

    # cb = CalibratedClassifierCV(xb, cv="prefit", method="sigmoid")
    # cb.fit(X.iloc[cal2], y.iloc[cal2])

    from sklearn.frozen import FrozenEstimator

    frozen_xb = FrozenEstimator(xb)

    cb = CalibratedClassifierCV(
        estimator=frozen_xb,
        method="sigmoid"
    )

    cb.fit(X.iloc[cal2], y.iloc[cal2])

    probs = cb.predict_proba(X.iloc[te_idx])[:, 1]
    cal_oof_probs[te_idx] = probs
    probs = cb.predict_proba(X.iloc[te_idx])[:,1]

    cal_oof_probs[te_idx] = probs

    auc = roc_auc_score(
        y.iloc[te_idx],
        probs
    )

    cal_cv_auc.append(auc)

    fpr, tpr, thresholds = roc_curve(
        y.iloc[te_idx],
        probs
    )

    best_threshold = float(
        thresholds[np.argmax(.7*tpr - .3*fpr)]
    )

    cal_thresholds.append(best_threshold)

    pred = (probs >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y.iloc[te_idx],
        pred
    ).ravel()

    cal_cv_sens.append(
        tp/(tp+fn)
    )

    cal_cv_spec.append(
        tn/(tn+fp)
    )
    # fpr, tpr, thresholds = roc_curve(y.iloc[te_idx], probs)
    # cal_thresholds.append(float(thresholds[np.argmax(tpr - fpr)]))

ece_after  = expected_calibration_error(y.values.astype(float), cal_oof_probs)
# cal_auc    = roc_auc_score(y, cal_oof_probs)
# fpr_f, tpr_f, thr_f = roc_curve(y, cal_oof_probs)
# cal_thresh = float(thr_f[np.argmax(tpr_f - fpr_f)])

# preds_cal       = (cal_oof_probs >= cal_thresh).astype(int)
# tn,fp,fn,tp     = confusion_matrix(y, preds_cal).ravel()
# cal_sens        = tp / (tp + fn)
# cal_spec        = tn / (tn + fp)

cal_auc = np.mean(cal_cv_auc)
cal_auc_std = np.std(cal_cv_auc)

cal_sens = np.mean(cal_cv_sens)
cal_spec = np.mean(cal_cv_spec)

cal_thresh = np.mean(cal_thresholds)

print(f"\nCalibrated model (OOF):")
# print(f"  AUC         : {cal_auc:.4f}")
# print(f"  Sensitivity : {cal_sens:.4f}")
# print(f"  Specificity : {cal_spec:.4f}")
# print(f"  Threshold   : {cal_thresh:.4f}  "
#       f"(std across folds: {np.std(cal_thresholds):.4f})")
print("\nCalibrated 5-Fold CV:")

print( f"AUC         : {cal_auc:.4f} +/- {cal_auc_std:.4f}")
print( f"Sensitivity : {cal_sens:.4f}")
print(f"Specificity : {cal_spec:.4f}")
print(f"Threshold   : {cal_thresh:.4f}")
print(f"Threshold std : {np.std(cal_thresholds):.4f}")

print(f"\nCalibration improvement:")
print(f"  ECE before  : {ece_before:.4f}")
print(f"  ECE after   : {ece_after:.4f}  "
      f"({'improved' if ece_after < ece_before else 'no change'})")
print(f"  Thresh std  : {np.std(cv_thresholds):.4f} → {np.std(cal_thresholds):.4f}")

# =====================================================
# NIS POLARITY CHECK
# =====================================================

print("\n=== Biomarker standalone AUC (sanity check) ===")
for col in ["NIS", "NIS_v2", "NIS_v3"]:
    if col in train_df.columns:
        col_auc = roc_auc_score(y, train_df[col])
        # If AUC < 0.5 the score is inverted (higher = healthier)
        # Flip and report the corrected AUC
        if col_auc < 0.5:
            corrected = 1.0 - col_auc
            print(f"  {col:<10} AUC = {col_auc:.4f}  "
                  f"⚠ INVERTED — corrected AUC = {corrected:.4f}  "
                  f"(score is oriented: high = healthy)")
        else:
            print(f"  {col:<10} AUC = {col_auc:.4f}")
        if col == "NIS_v2":
            effective = max(col_auc, 1 - col_auc)
            status = "✓ on track" if effective >= 0.75 else "✗ below target"
            print(f"             {status} (effective AUC = {effective:.4f}, target ≥ 0.75)")

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance = (
    pd.DataFrame({
        "feature":    FEATURES,
        "importance": final_xgb.feature_importances_
    })
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)

print("\nFeature Importance (top 20):")
print(importance.head(20).to_string(index=False))

# =====================================================
# SAVE
# =====================================================

# Raw XGBoost — .json format (version-safe, no UBJSON warning)
final_xgb.save_model(_model_path_json)

# Calibrated model — joblib (sklearn wrapper, can't use .save_model)
joblib.dump(calibrated_model, CALIBRATED_MODEL_PATH)

importance.to_csv(IMPORTANCE_PATH, index=False)

metrics = {
    "rows":                    int(len(train_df)),
    "n_features":              len(FEATURES),
    "threshold_strategy":      THRESHOLD_STRATEGY,

    # Raw XGBoost CV
    "cv_auc_mean":             round(float(np.mean(cv_aucs)), 4),
    "cv_auc_std":              round(float(np.std(cv_aucs)), 4),
    "cv_sensitivity_mean":     round(float(np.mean(cv_sens)), 4),
    "cv_sensitivity_std":      round(float(np.std(cv_sens)), 4),
    "cv_specificity_mean":     round(float(np.mean(cv_spec)), 4),
    "cv_specificity_std":      round(float(np.std(cv_spec)), 4),
    "cv_threshold_mean":       round(float(np.mean(cv_thresholds)), 4),
    "cv_threshold_std":        round(float(np.std(cv_thresholds)), 4),

    # Calibrated model
    "calibrated_auc":          round(cal_auc, 4),
    "calibrated_sensitivity":  round(cal_sens, 4),
    "calibrated_specificity":  round(cal_spec, 4),
    "calibrated_threshold":    round(cal_thresh, 4),
    "calibrated_thresh_std":   round(float(np.std(cal_thresholds)), 4),
    "ece_before":              round(ece_before, 4),
    "ece_after":               round(ece_after, 4),

    # Paths
    "model_path":              _model_path_json,
    "calibrated_model_path":   CALIBRATED_MODEL_PATH,

    "features":                FEATURES,
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

# =====================================================
# SUMMARY
# =====================================================

print("\n==============================")
print("SAVED")
print(f"  Raw XGBoost      : {_model_path_json}")
print(f"  Calibrated model : {CALIBRATED_MODEL_PATH}")
print(f"  Metrics          : {METRICS_PATH}")
print(f"  Feature importance: {IMPORTANCE_PATH}")
print("==============================")
print("\nNext step: python train_fusion_v5.py")
print(f"  (will use calibrated model: {CALIBRATED_MODEL_PATH})")


print(metrics["cv_auc_mean"], metrics ["calibrated_auc"])


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix
)
import numpy as np

def benchmark_model(name, model, X, y, seed=42):

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    aucs = []
    sens = []
    spec = []
    thresh = []

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):

        model.fit(
            X.iloc[tr],
            y.iloc[tr]
        )

        probs = model.predict_proba(
            X.iloc[te]
        )[:,1]

        auc = roc_auc_score(
            y.iloc[te],
            probs
        )

        fpr, tpr, thresholds = roc_curve(
            y.iloc[te],
            probs
        )

        best_idx = np.argmax(tpr - fpr)
        best_thr = thresholds[best_idx]

        pred = (probs >= best_thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y.iloc[te],
            pred
        ).ravel()

        aucs.append(auc)
        sens.append(tp/(tp+fn))
        spec.append(tn/(tn+fp))
        thresh.append(best_thr)

    print("\n" + "="*60)
    print(name)
    print("="*60)

    print(
        f"AUC         : {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}"
    )

    print(
        f"Sensitivity : {np.mean(sens):.4f}"
    )

    print(
        f"Specificity : {np.mean(spec):.4f}"
    )

    print(
        f"Threshold SD: {np.std(thresh):.4f}"
    )


# SEED=42

# benchmark_model(
#     "Logistic Regression",
#     Pipeline([
#         ("scaler", RobustScaler()),
#         ("clf", LogisticRegression(
#             max_iter=5000,
#             class_weight="balanced",
#             random_state=SEED
#         ))
#     ]),
#     X,
#     y
# )

# benchmark_model(
#     "Random Forest",
#     RandomForestClassifier(
#         n_estimators=300,
#         max_depth=8,
#         class_weight="balanced",
#         random_state=SEED
#     ),
#     X,
#     y
# )

# benchmark_model(
#     "SVM (RBF)",
#     Pipeline([
#         ("scaler", RobustScaler()),
#         ("clf", SVC(
#             kernel="rbf",
#             C=1.0,
#             gamma="scale",
#             probability=True,
#             class_weight="balanced",
#             random_state=SEED
#         ))
#     ]),
#     X,
#     y
# )

# benchmark_model(
#     "XGBoost",
#     XGBClassifier(
#         **best_params,
#         random_state=SEED
#     ),
#     X,
#     y
# )