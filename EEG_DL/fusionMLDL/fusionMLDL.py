import joblib

# LOAD MODEL
emotiv_model = joblib.load("emotiv_xgb_model.pkl")

def fuse_emotiv_dl(dl_prob, emotiv_features):

    emotiv_prob = emotiv_model.predict_proba([emotiv_features])[0][1]

    # BASE WEIGHTS
    w_emotiv = 0.7
    w_dl = 0.3

    # CONFIDENCE ADJUSTMENT
    if dl_prob > 0.85 or dl_prob < 0.15:
        w_dl = 0.4
        w_emotiv = 0.6

    if emotiv_prob > 0.85 or emotiv_prob < 0.15:
        w_emotiv = 0.8
        w_dl = 0.2

    final_prob = (w_emotiv * emotiv_prob) + (w_dl * dl_prob)

    return {
        "emotiv_prob": float(emotiv_prob),
        "dl_prob": float(dl_prob),
        "final_prob": float(final_prob)
    }