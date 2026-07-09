import os
# local = "/home/ubuntu/eeg_train_pipeline"
# local = "/Users/nidhi/Documents/Test Server/NEMA Train /Training"
local = os.path.dirname(os.path.abspath(__file__))
BUCKET = "dementia-research2025"
external_cohort_dir = local + "/outputs/external/"
external_cohort_manifest = local + "/outputs/external/padic_external_manifest.csv"
#Used in create_raw_biomarker
# DATASETS = ["AD_AUDITORY_19CH_REORDERED",
# "ADFSU_19CH_NPY",
# "BrainLat_19CH_NPY",
# # "caueeg_npy",
# # "ds004504",
# "Isfahan_MCI",
# "baca_resampled_train"
# ]


DATASETS = [
    "ADFSU",
    "DS004504",
    "BrainLat",
    "P-ADIC",
    "Isfahan",
]

#Used BY Spectral & Biomarker api
REGION_INDEX = {
    "frontal":0,
    "central":1,
    "parietal":2,
    "occipital":3,
    "temporal":4
}

# Band Definitions
BANDS = {
    "delta": (1,4),
    "theta": (4,8),
    "alpha": (8,13),
    "beta":  (13,30)
}

#used by eeg_utils file

EMOTIV_CHANNELS = [
    "AF3","F7","F3","FC5","T7","P7","O1",
    "O2","P8","T8","FC6","F4","F8","AF4"
]

FEATURES = [

    # Existing
    "posterior_dominance_index",
    "occipital_entropy",
    "alpha_peak_gradient",
    "entropy_gradient",
    "theta_alpha_ratio_frontal",

    "PCR",
    "CLI",
    "NIS",

    # New
    "global_slope",
    "occipital_slope",

    "posterior_alpha_power",
    "posterior_theta_power",
    "posterior_theta_alpha_ratio",

    "occipital_alpha_peak",
    "parietal_alpha_peak",

    "global_entropy",

    "frontal_occipital_similarity",
    "frontal_occipital_distance",

    # Composite
    "NIS_v3",

"NIS_v2"
]


MODEL_PATH = (
    local + "/model/"
    "xgb_v2.pkl"
)

METRICS_PATH = (
    local + "/metrics/"
    "xgb_v2_metrics.json"
)

IMPORTANCE_PATH = (
    local + "/metrics/"
    "xgb_v2_feature_importance.csv"
)

# ml_CSV_PATH = (
#    local +  "/outputs/"
#     "fusion_final_clean_v2.csv"
# )
import os
# ML_CSV_PATH = os.path.join(
#     local, "outputs", "biomarkers", "fusion_biomarkers.csv"
# )
ML_CSV_PATH = os.path.join(local, "BIOMARKERS.csv")


candidate_features = [

    "posterior_theta_alpha_ratio",
    "memory_theta_alpha_ratio",
    "posterior_alpha_power",
    "posterior_theta_power",

    "posterior_dominance_index",

    "cognitive_decline_index",

    "global_entropy",
    "global_slope",

    "occipital_alpha_peak",
    "parietal_alpha_peak",

    "gamma_activity_ratio"
]


oldFEATURES = [

    "posterior_dominance_index",
    "occipital_entropy",
    "alpha_peak_gradient",
    "entropy_gradient",
    "theta_alpha_ratio_frontal",

    "PCR",
    "CLI",
    "NIS",
    # "NIS_v2"
    "internal_brain_health_score"
]

FINAL_FEATURES = [

    # strongest novel feature
    "frontal_occipital_distance",

    # complementary
    "frontal_occipital_similarity",

    # strongest biomarker
    "cognitive_decline_index",

    # classical EEG
    "PCR",
    "CLI",

    "posterior_dominance_index",

    "occipital_entropy",
    "entropy_gradient",

    "posterior_theta_alpha_ratio",
    "theta_alpha_ratio_frontal", 
    "NIS_v3"
]