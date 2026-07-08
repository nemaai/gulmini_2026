import os
import uuid

from data_pipeline.data_pipeline import process_npy_api

INPUT_DIR = "storage/input"
OUTPUT_DIR = "storage/output"


def process_pipeline(file):

    processing_id = str(uuid.uuid4())

    input_folder = os.path.join(INPUT_DIR, processing_id)
    output_folder = os.path.join(OUTPUT_DIR, processing_id)

    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    input_path = os.path.join(input_folder, file.filename)

    file.save(input_path)

    result = process_npy_api(
        npy_path=input_path,
        output_dir=output_folder,
    )

    return {
        "processing_id": processing_id,
        "status": "completed",
        **result,
    }