import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================
# LOAD CSV
# ============================================

df = pd.read_csv(
    "outputs/brain_age_results.csv",
    encoding="latin1"
)

print("Loaded Shape:", df.shape)

# ============================================
# CLEAN
# ============================================

df = df.dropna(subset=[
    "Labels",
    "predicted_label",
    "ML_risk_percent",
    "confidence_score",
    "brain_age_gap"
])

# ============================================
# TRUE / PRED LABELS
# ============================================

y_true = df["Labels"].astype(int)
y_pred = df["predicted_label"].astype(int)

# ============================================
# BASIC METRICS
# ============================================

accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred
)

recall = recall_score(
    y_true,
    y_pred
)

f1 = f1_score(
    y_true,
    y_pred
)

# ============================================
# CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_true,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

# ============================================
# SPECIFICITY
# ============================================

specificity = tn / (tn + fp)

# ============================================
# PRINT RESULTS
# ============================================

print("\n================================")
print("CLASSIFICATION METRICS")
print("================================")

print(f"Accuracy     : {accuracy*100:.2f}%")
print(f"Precision    : {precision*100:.2f}%")
print(f"Sensitivity  : {recall*100:.2f}%")
print(f"Specificity  : {specificity*100:.2f}%")
print(f"F1 Score     : {f1*100:.2f}%")

# ============================================
# CONFUSION MATRIX
# ============================================

print("\n================================")
print("CONFUSION MATRIX")
print("================================")

print(cm)

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

# ============================================
# FULL REPORT
# ============================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

print(
    classification_report(
        y_true,
        y_pred
    )
)

# ============================================
# BRAIN AGE GAP BY RISK BAND
# ============================================

print("\n================================")
print("BRAIN AGE GAP BY BAND")
print("================================")

risk_gap = df.groupby(
    "BAND"
)["brain_age_gap"].agg([
    "mean",
    "std",
    "count"
])

print(risk_gap)

# ============================================
# HEALTHY VS ABNORMAL GAP
# ============================================

print("\n================================")
print("HEALTHY VS ABNORMAL")
print("================================")

label_gap = df.groupby(
    "Labels"
)["brain_age_gap"].agg([
    "mean",
    "std",
    "count"
])

print(label_gap)

# ============================================
# CORRELATION
# ============================================

print("\n================================")
print("CORRELATION")
print("================================")

corr = df["brain_age_gap"].corr(
    df["ML_risk_percent"]
)

print(
    f"Brain Age Gap vs ML Risk Correlation: "
    f"{corr:.3f}"
)

# ============================================
# CONFIDENCE DISTRIBUTION
# ============================================

print("\n================================")
print("CONFIDENCE DISTRIBUTION")
print("================================")

print(
    df["analysis_reliability"]
    .value_counts()
)

# ============================================
# CONFIDENCE VS CORRECTNESS
# ============================================

print("\n================================")
print("CONFIDENCE VS CORRECTNESS")
print("================================")

df["correct_prediction"] = (
    df["Labels"] ==
    df["predicted_label"]
)

confidence_stats = df.groupby(
    "correct_prediction"
)["confidence_score"].agg([
    "mean",
    "std",
    "count"
])

print(confidence_stats)

# ============================================
# OVERALL SUMMARY
# ============================================

print("\n================================")
print("SUMMARY")
print("================================")

print(
    f"Average Brain Age Gap: "
    f"{df['brain_age_gap'].mean():.2f}"
)

print(
    f"Average Confidence Score: "
    f"{df['confidence_score'].mean():.2f}"
)

print(
    f"Total Samples: "
    f"{len(df)}"
)