from pipeline.dataset_registry import (
    DATASETS
)

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


for dataset, cfg in DATASETS.items():

    print("\n=================")
    print(dataset)
    print("=================")

    try:

        result = process_record(
            cfg["file"]
        )

        quality = run_quality(
            result
        )

        save_processed(

            cfg["output"],

            result,

            quality
        )

        append_inventory(

            dataset,

            cfg["file"],

            result,

            quality
        )

        print(
            "QUALITY:",
            quality["overall_quality"]
        )

        print(
            "SHAPE:",
            result["eeg"].shape
        )

    except Exception as e:

        print(
            "FAILED"
        )

        print(e)