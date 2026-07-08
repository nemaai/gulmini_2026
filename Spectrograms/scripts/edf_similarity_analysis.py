# For EACH test EDF:

# Find all its spectrogram windows

# Compare each window against:

# normal_bank/T4
# Compute:
# mean SSIM
# min SSIM
# std deviation
# abnormal window %
# Save JSON output


import os
import json
import logging
import cv2
import numpy as np

from skimage.metrics import structural_similarity as ssim

# ==========================================
# LOGGING
# ==========================================

LOG_DIR = "Spectrogram/logs"

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(
                LOG_DIR,
                "edf_similarity_analysis.log"
            )
        ),
        logging.StreamHandler()
    ]
)

# ==========================================
# CONFIG
# ==========================================

TEST_BANK = "Spectrogram/test_bank/T4"

NORMAL_BANK = "Spectrogram/normal_bank/T4"

OUTPUT_DIR = "Spectrogram/outputs"

ABNORMAL_THRESHOLD = 0.70

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD NORMAL IMAGES
# ==========================================

normal_images = []

normal_files = [
    f for f in os.listdir(NORMAL_BANK)
    if f.endswith(".png")
]

logging.info(
    f"Found {len(normal_files)} "
    f"normal spectrograms"
)

for file in normal_files:

    path = os.path.join(
        NORMAL_BANK,
        file
    )

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is not None:

        normal_images.append(img)

# ==========================================
# GROUP TEST WINDOWS BY EDF
# ==========================================

test_files = [
    f for f in os.listdir(TEST_BANK)
    if f.endswith(".png")
]

edf_groups = {}

for file in test_files:

    # Example:
    # patient1_T4_120s.png

    parts = file.split("_T4_")

    if len(parts) != 2:
        continue

    edf_name = parts[0]

    if edf_name not in edf_groups:
        edf_groups[edf_name] = []

    edf_groups[edf_name].append(file)

logging.info(
    f"Found {len(edf_groups)} "
    f"test EDF groups"
)

# ==========================================
# ANALYZE EACH EDF
# ==========================================

for edf_name, window_files in edf_groups.items():

    logging.info("=" * 60)

    logging.info(
        f"Analyzing EDF: {edf_name}"
    )

    max_scores = []

    abnormal_count = 0

    total_windows = 0

    # --------------------------------------
    # PROCESS EACH WINDOW
    # --------------------------------------

    for window_file in window_files:

        try:

            window_path = os.path.join(
                TEST_BANK,
                window_file
            )

            test_img = cv2.imread(
                window_path,
                cv2.IMREAD_GRAYSCALE
            )

            if test_img is None:
                continue

            best_score = -1

            # ----------------------------------
            # COMPARE AGAINST ALL NORMALS
            # ----------------------------------

            for normal_img in normal_images:

                resized_normal = cv2.resize(
                    normal_img,
                    (
                        test_img.shape[1],
                        test_img.shape[0]
                    )
                )

                score, _ = ssim(
                    test_img,
                    resized_normal,
                    full=True
                )

                if score > best_score:
                    best_score = score

            max_scores.append(best_score)

            total_windows += 1

            if best_score < ABNORMAL_THRESHOLD:
                abnormal_count += 1

            logging.info(
                f"{window_file} "
                f"--> Best SSIM: "
                f"{best_score:.4f}"
            )

        except Exception as e:

            logging.error(
                f"{window_file} --> {str(e)}"
            )

    # ======================================
    # FINAL METRICS
    # ======================================

    mean_ssim = float(
        np.mean(max_scores)
    )

    min_ssim = float(
        np.min(max_scores)
    )

    std_ssim = float(
        np.std(max_scores)
    )

    abnormal_percent = float(
        (abnormal_count / total_windows) * 100
    )

    # ======================================
    # FINAL STATUS
    # ======================================

    if mean_ssim > 0.85:

        status = "NORMAL"

    elif mean_ssim > 0.70:

        status = "MILD_DEVIATION"

    else:

        status = "ABNORMAL"

    # ======================================
    # RESULT JSON
    # ======================================

    result = {

        "edf": edf_name,

        "channel": "T4",

        "windows": total_windows,

        "mean_ssim": round(
            mean_ssim,
            4
        ),

        "min_ssim": round(
            min_ssim,
            4
        ),

        "std_ssim": round(
            std_ssim,
            4
        ),

        "abnormal_window_percent": round(
            abnormal_percent,
            2
        ),

        "status": status
    }

    # ======================================
    # SAVE JSON
    # ======================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{edf_name}.json"
    )

    with open(output_path, "w") as f:

        json.dump(
            result,
            f,
            indent=4
        )

    logging.info(
        f"Saved JSON: {output_path}"
    )

    logging.info(
        json.dumps(
            result,
            indent=4
        )
    )

logging.info("=" * 60)

logging.info("Analysis complete.")