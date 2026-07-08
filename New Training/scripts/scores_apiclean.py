import numpy as np
import json
import os

# =====================================================
# LOAD REFERENCE SCALER (TRAINED OFFLINE)
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "reference_scaler.json")

try:
    with open(SCALER_PATH, "r") as f:
        SCALER = json.load(f)

    MEAN = SCALER["mean"]
    STD = SCALER["scale"]

    print("Reference scaler loaded")

except Exception:
    MEAN = None
    STD = None
    print("Reference scaler NOT loaded")


FEATURES = [
    "posterior_dominance_index",
    "occipital_entropy",
    "alpha_peak_gradient",
    "entropy_gradient",
    "theta_alpha_ratio_frontal"
]

# =====================================================
# Z-SCORE USING REFERENCE DISTRIBUTION
# =====================================================

def compute_z(value, mean, std):
    return (value - mean) / (std + 1e-6)

# =====================================================
# DERIVED METRICS (IDENTICAL LOGIC)
# =====================================================

def compute_scores_from_row(feats):

    if MEAN is None or STD is None:
        raise RuntimeError("Reference scaler missing or invalid")

    Z = {}

    for i, key in enumerate(FEATURES):
        Z[f"{key}_z"] = compute_z(
            feats[key],
            MEAN[i],
            STD[i]
        )

    PCR = (
        Z["posterior_dominance_index_z"]
        - Z["occipital_entropy_z"]
    )

    CLI = (
        Z["alpha_peak_gradient_z"]
        - Z["entropy_gradient_z"]
    )

    NIS = (
        0.5 * Z["posterior_dominance_index_z"]
        + 0.3 * Z["alpha_peak_gradient_z"]
        - 0.2 * Z["occipital_entropy_z"]
    )

    internal_brain_health_score = (
        0.4 * PCR
        + 0.35 * CLI
        + 0.25 * NIS
    )

    risk_percent = np.clip(
    100 / (1 + np.exp(internal_brain_health_score)),
    0,
    100)

    return {
        "PCR": PCR,
        "CLI": CLI,
        "NIS": NIS,
        "internal_brain_health_score": internal_brain_health_score,
        "risk_percent": risk_percent
    }

# =====================================================
# SAFETY
# =====================================================

if __name__ == "__main__":
    print("scores_clean API module ready")
