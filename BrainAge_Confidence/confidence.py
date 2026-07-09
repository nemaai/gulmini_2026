def calculate_analysis_confidence(
    ml_risk_percent,
    signal_quality_score=85
):

    # ========================================
    # MODEL CONFIDENCE
    # ========================================

    if ml_risk_percent < 65:

        model_conf = 0.65

    elif ml_risk_percent < 72:

        model_conf = 0.78

    else:

        model_conf = 0.90

    # ========================================
    # DISTANCE FROM THRESHOLD
    # ========================================

    threshold = 72

    boundary_distance = abs(
        ml_risk_percent - threshold
    )

    boundary_score = min(
        boundary_distance / 30,
        1.0
    )

    # ========================================
    # SIGNAL QUALITY
    # ========================================

    signal_score = (
        signal_quality_score / 100
    )

    # ========================================
    # FINAL CONFIDENCE
    # ========================================

    final_confidence = (

        0.45 * model_conf +

        0.35 * boundary_score +

        0.20 * signal_score

    ) * 100

    final_confidence = max(
        0,
        min(100, final_confidence)
    )

    # ========================================
    # LABEL
    # ========================================

    if final_confidence >= 85:

        reliability = (
            "Very High Reliability"
        )

    elif final_confidence >= 70:

        reliability = (
            "High Reliability"
        )

    elif final_confidence >= 50:

        reliability = (
            "Moderate Reliability"
        )

    else:

        reliability = (
            "Low Reliability"
        )

    return {

        "confidence_score":
        round(final_confidence, 1),

        "reliability":
        reliability
    }