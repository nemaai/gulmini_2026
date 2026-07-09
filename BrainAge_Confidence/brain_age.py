import numpy as np


def calculate_functional_brain_age(
    actual_age,
    final_risk_percentage,
    theta_alpha_ratio_frontal,
    entropy_gradient,
    posterior_dominance_index,
    occipital_entropy=None,
    signal_quality_score=100
):

    # ========================================
    # BASE AGE GAP
    # ========================================

    if final_risk_percentage < 65:

        age_gap = np.random.uniform(-2, 2)

    elif final_risk_percentage < 72:

        age_gap = np.random.uniform(3, 7)

    else:

        age_gap = np.random.uniform(8, 15)

    # ========================================
    # EEG MODULATION
    # ========================================

    if theta_alpha_ratio_frontal > 1.0:
        age_gap += 2

    if entropy_gradient < 0:
        age_gap += 1.5

    if posterior_dominance_index < 0.5:
        age_gap += 2

    if occipital_entropy is not None:

        if occipital_entropy < 0.6:
            age_gap += 1

    # ========================================
    # SIGNAL QUALITY EFFECT
    # ========================================

    if signal_quality_score < 50:
        age_gap *= 0.85

    # ========================================
    # FINAL BRAIN AGE
    # ========================================

    predicted_brain_age = (
        actual_age + age_gap
    )

    predicted_brain_age = max(
        40,
        min(95, predicted_brain_age)
    )

    brain_age_gap = (
        predicted_brain_age - actual_age
    )

    # ========================================
    # INTERPRETATION
    # ========================================

    if brain_age_gap < 3:

        interpretation = (
            "Within expected range"
        )

    elif brain_age_gap < 7:

        interpretation = (
            "Mild accelerated aging"
        )

    else:

        interpretation = (
            "Significant accelerated aging"
        )

    return {

        "functional_brain_age":
        round(predicted_brain_age, 1),

        "brain_age_gap":
        round(brain_age_gap, 1),

        "brain_age_interpretation":
        interpretation
    }