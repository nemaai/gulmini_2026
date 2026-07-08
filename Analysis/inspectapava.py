import os
import boto3

from pipeline.process_record import (
    process_record
)

from pipeline.quality_adapter import (
    run_quality
)

from pipeline.save_processed import (
    save_processed
)

from pipeline.inventory import (
    append_inventory
)

BUCKET = "dementia-research2025"

PREFIX = "nema_final_used/APAVA/AD_Data/"

TEMP_DIR = "temp"

os.makedirs(
    TEMP_DIR,
    exist_ok=True
)

s3 = boto3.client(
    "s3"
)

paginator = s3.get_paginator(
    "list_objects_v2"
)

for page in paginator.paginate(

    Bucket=BUCKET,

    Prefix=PREFIX

):

    for obj in page.get(
        "Contents",
        []
    ):

        key = obj["Key"]

        if not key.endswith(".mat"):
            continue

        filename = os.path.basename(
            key
        )

        subject = filename.replace(
            ".mat",
            ""
        )

        local_file = os.path.join(
            TEMP_DIR,
            filename
        )

        print(
            "\nProcessing:",
            filename
        )

        try:

            s3.download_file(

                BUCKET,

                key,

                local_file
            )

            result = process_record(
                local_file
            )

            quality = run_quality(
                result
            )

            output_dir = os.path.join(

                "processed",

                "APAVA",

                subject
            )

            save_processed(

                output_dir,

                result,

                quality
            )

            append_inventory(

                "APAVA",

                key,

                result,

                quality
            )

            os.remove(
                local_file
            )

            print(
                "QUALITY:",
                quality[
                    "overall_quality"
                ]
            )

        except Exception as e:

            print(
                "FAILED"
            )

            print(e)