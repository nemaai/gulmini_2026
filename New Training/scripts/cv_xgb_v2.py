import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)

# ==========================================
# LOAD
# ==========================================

df = pd.read_csv(
    "/home/ubuntu/training/outputs/fusion_final_clean_v2.csv"
)

df = df[
    df["binary_label"] != -1
].copy()

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

X = df[FEATURES]
y = df["binary_label"]

print("Dataset:", X.shape)
print(y.value_counts())

# ==========================================
# CV
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

accs = []
aucs = []
sens = []
specs = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:,1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    cm = confusion_matrix(y_test, preds)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    accs.append(acc)
    aucs.append(auc)
    sens.append(sensitivity)
    specs.append(specificity)

    print(f"\nFold {fold}")
    print(f"AUC: {auc:.4f}")
    print(f"ACC: {acc:.4f}")
    print(f"SEN: {sensitivity:.4f}")
    print(f"SPEC:{specificity:.4f}")

# ==========================================
# FINAL
# ==========================================

print("\n============================")
print("5-FOLD RESULTS")
print("============================")

print(
    f"AUC         : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
)

print(
    f"Accuracy    : {np.mean(accs):.4f} ± {np.std(accs):.4f}"
)

print(
    f"Sensitivity : {np.mean(sens):.4f} ± {np.std(sens):.4f}"
)

print(
    f"Specificity : {np.mean(specs):.4f} ± {np.std(specs):.4f}"
)