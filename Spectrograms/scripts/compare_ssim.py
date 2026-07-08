import os
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
                "compare_ssim.log"
            )
        ),
        logging.StreamHandler()
    ]
)

# ==========================================
# CONFIG
# ==========================================

TEST_IMAGE = (
    "Spectrogram/test_input/test_T4.png"
)

NORMAL_BANK = (
    "Spectrogram/normal_bank/T4"
)

# ==========================================
# LOAD TEST IMAGE
# ==========================================

test_img = cv2.imread(
    TEST_IMAGE,
    cv2.IMREAD_GRAYSCALE
)

if test_img is None:

    raise ValueError(
        "Could not load test image"
    )

logging.info(
    f"Loaded test image: {TEST_IMAGE}"
)

# ==========================================
# NORMAL IMAGES
# ==========================================

normal_images = [
    f for f in os.listdir(NORMAL_BANK)
    if f.endswith(".png")
]

logging.info(
    f"Found {len(normal_images)} "
    f"normal spectrograms"
)

scores = []

# ==========================================
# COMPARE
# ==========================================

for img_name in normal_images:

    try:

        img_path = os.path.join(
            NORMAL_BANK,
            img_name
        )

        normal_img = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        if normal_img is None:
            continue

        normal_img = cv2.resize(
            normal_img,
            (
                test_img.shape[1],
                test_img.shape[0]
            )
        )

        score, diff = ssim(
            test_img,
            normal_img,
            full=True
        )

        scores.append(score)

        logging.info(
            f"{img_name} "
            f"--> SSIM: {score:.4f}"
        )

    except Exception as e:

        logging.error(
            f"{img_name} --> {str(e)}"
        )

# ==========================================
# FINAL RESULT
# ==========================================

mean_score = np.mean(scores)

logging.info("=" * 50)

logging.info(
    f"Mean SSIM: {mean_score:.4f}"
)

# ==========================================
# INTERPRETATION
# ==========================================

if mean_score > 0.85:

    result = "NORMAL"

elif mean_score > 0.70:

    result = "MILD DEVIATION"

else:

    result = "ABNORMAL"

logging.info(
    f"Final Result: {result}"
)