import pandas as pd
import json

FEATURES = [
    "posterior_dominance_index",
    "occipital_entropy",
    "alpha_peak_gradient",
    "entropy_gradient",
    "theta_alpha_ratio_frontal"
]

df = pd.read_csv(
    "/home/ubuntu/training/outputs/raw_biomarkers_v2.csv"
)

scaler = {
    "mean": df[FEATURES].mean().tolist(),
    "scale": df[FEATURES].std().tolist()
}

with open(
    "/home/ubuntu/training/configs/new_reference_scaler.json",
    "w"
) as f:
    json.dump(scaler, f, indent=2)

print("Saved scaler")
print(scaler)