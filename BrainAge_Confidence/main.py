import os
import pandas as pd

from brain_age import (
    calculate_functional_brain_age
)

from confidence import (
    calculate_analysis_confidence
)

# ============================================
# LOAD CSV
# ============================================

df = pd.read_csv(
    "FINAL CSV_XGB.csv",
    encoding="latin1"
)

print("CSV Loaded:", df.shape)

# ============================================
# CLEAN AGE
# ============================================

df["Age"] = (
    df["Age"]
    .astype(str)
    .str.strip()
)

df["Age"] = pd.to_numeric(
    df["Age"],
    errors="coerce"
)

df = df.dropna(subset=["Age"])

df["Age"] = df["Age"].astype(float)

print("Valid Age Rows:", len(df))

# ============================================
# OUTPUT STORAGE
# ============================================

brain_ages = []
brain_gaps = []
brain_texts = []

confidence_scores = []
confidence_labels = []

# ============================================
# PROCESS EACH ROW
# ============================================

for idx, row in df.iterrows():

    try:

        # ====================================
        # REQUIRED VALUES
        # ====================================

        actual_age = row["Age"]

        ml_risk = row["ML_risk_percent"]

        theta_ratio = row[
            "theta_alpha_ratio_frontal"
        ]

        entropy_gradient = row[
            "entropy_gradient"
        ]

        posterior_index = row[
            "posterior_dominance_index"
        ]

        occipital_entropy = row[
            "occipital_entropy"
        ]

        # ====================================
        # BRAIN AGE
        # ====================================

        brain_age_result = (
            calculate_functional_brain_age(

                actual_age=
                actual_age,

                final_risk_percentage=
                ml_risk,

                theta_alpha_ratio_frontal=
                theta_ratio,

                entropy_gradient=
                entropy_gradient,

                posterior_dominance_index=
                posterior_index,

                occipital_entropy=
                occipital_entropy,

                signal_quality_score=85
            )
        )

        # ====================================
        # CONFIDENCE
        # ====================================

        confidence_result = (
            calculate_analysis_confidence(

                ml_risk_percent=
                ml_risk,

                signal_quality_score=85
            )
        )

        # ====================================
        # STORE
        # ====================================

        brain_ages.append(
            brain_age_result[
                "functional_brain_age"
            ]
        )

        brain_gaps.append(
            brain_age_result[
                "brain_age_gap"
            ]
        )

        brain_texts.append(
            brain_age_result[
                "brain_age_interpretation"
            ]
        )

        confidence_scores.append(
            confidence_result[
                "confidence_score"
            ]
        )

        confidence_labels.append(
            confidence_result[
                "reliability"
            ]
        )

    except Exception as e:

        print(f"Row {idx} Error:", e)

        brain_ages.append(None)
        brain_gaps.append(None)
        brain_texts.append(None)

        confidence_scores.append(None)
        confidence_labels.append(None)

# ============================================
# ADD TO DATAFRAME
# ============================================

df["functional_brain_age"] = brain_ages

df["brain_age_gap"] = brain_gaps

df["brain_age_interpretation"] = brain_texts

df["confidence_score"] = confidence_scores

df["analysis_reliability"] = confidence_labels

# ============================================
# CREATE OUTPUT FOLDER
# ============================================

os.makedirs(
    "outputs",
    exist_ok=True
)

# ============================================
# SAVE OUTPUT
# ============================================

output_path = (
    "outputs/brain_age_results.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\n================================")
print("PROCESS COMPLETE")
print("================================")

print("Saved to:")
print(output_path)

print("\nSAMPLE OUTPUT:")
print(

    df[[
        "Age",
        "ML_risk_percent",
        "functional_brain_age",
        "brain_age_gap",
        "confidence_score"
    ]].head()

)