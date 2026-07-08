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
                "window_ssim_analysis.log"
            )
        ),
        logging.StreamHandler()
    ]
)

# ==========================================
# CONFIG
# ==========================================

TEST_BANK = "Spectrogram/normal_test_bank/T4"

NORMAL_BANK = "Spectrogram/normal_bank/T4"

OUTPUT_DIR = "Spectrogram/outputs/normal_window_analysis"

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
    f"Loaded {len(normal_files)} "
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
# GROUP TEST FILES
# ==========================================

test_files = [
    f for f in os.listdir(TEST_BANK)
    if f.endswith(".png")
]

edf_groups = {}

for file in test_files:

    parts = file.split("_T4_")

    if len(parts) != 2:
        continue

    edf_name = parts[0]

    if edf_name not in edf_groups:
        edf_groups[edf_name] = []

    edf_groups[edf_name].append(file)

# ==========================================
# PROCESS EDFs
# ==========================================

for edf_name, window_files in edf_groups.items():

    logging.info(
        f"Analyzing: {edf_name}"
    )

    window_results = []

    for window_file in sorted(window_files):

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

            # ----------------------------------
            # Extract window time
            # ----------------------------------

            # Example:
            # 41_T4_120s.png

            time_part = (
                window_file
                .split("_T4_")[1]
                .replace(".png", "")
            )

            window_results.append({

                "window": time_part,

                "best_ssim": round(
                    float(best_score),
                    4
                )
            })

            logging.info(
                f"{window_file} "
                f"--> {best_score:.4f}"
            )

        except Exception as e:

            logging.error(
                f"{window_file} --> {str(e)}"
            )

    # ======================================
    # SAVE JSON
    # ======================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{edf_name}_windows.json"
    )

    with open(output_path, "w") as f:

        json.dump(
            window_results,
            f,
            indent=4
        )

    logging.info(
        f"Saved: {output_path}"
    )

logging.info("Done.")