import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix
)

from xgboost import XGBClassifier

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

print("Dataset:", X.shape)
print(y.value_counts())

# =====================================================
# CV
# =====================================================

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

aucs = []
accs = []
sens = []
specs = []

fold = 1

for train_idx, test_idx in skf.split(X, y):

    print(f"\nFold {fold}")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probs)

    acc = accuracy_score(y_test, preds)

    cm = confusion_matrix(y_test, preds)

    tn, fp, fn, tp = cm.ravel()

    sen = tp / (tp + fn)

    spec = tn / (tn + fp)

    aucs.append(auc)
    accs.append(acc)
    sens.append(sen)
    specs.append(spec)

    print("AUC :", round(auc,4))
    print("ACC :", round(acc,4))
    print("SEN :", round(sen,4))
    print("SPEC:", round(spec,4))

    fold += 1

# =====================================================
# FINAL RESULTS
# =====================================================

print("\n============================")
print("5-FOLD FUSION RESULTS")
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