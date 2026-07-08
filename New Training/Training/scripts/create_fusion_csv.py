import pandas as pd
import json
import numpy as np

INPUT_CSV = "/home/ubuntu/training/outputs/raw_biomarkers_v2.csv"

SCALER_JSON = "/home/ubuntu/training/configs/new_reference_scaler.json"

OUTPUT_CSV = "/home/ubuntu/training/outputs/fusion_final_clean_v2.csv"

FEATURES = [
    "posterior_dominance_index",
    "occipital_entropy",
    "alpha_peak_gradient",
    "entropy_gradient",
    "theta_alpha_ratio_frontal"
]

with open(SCALER_JSON, "r") as f:
    scaler = json.load(f)

MEAN = scaler["mean"]
STD = scaler["scale"]

df = pd.read_csv(INPUT_CSV)

for i, feat in enumerate(FEATURES):

    df[f"{feat}_z"] = (
        df[feat] - MEAN[i]
    ) / (STD[i] + 1e-6)

df["PCR"] = (
    df["posterior_dominance_index_z"]
    - df["occipital_entropy_z"]
)

df["CLI"] = (
    df["alpha_peak_gradient_z"]
    - df["entropy_gradient_z"]
)

df["NIS"] = (
    0.5 * df["posterior_dominance_index_z"]
    + 0.3 * df["alpha_peak_gradient_z"]
    - 0.2 * df["occipital_entropy_z"]
)

df["internal_brain_health_score"] = (
    0.4 * df["PCR"]
    + 0.35 * df["CLI"]
    + 0.25 * df["NIS"]
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("Saved:", OUTPUT_CSV)
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())