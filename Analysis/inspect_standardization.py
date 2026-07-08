from pipeline.process_record import (
    process_record
)

files = [

    r"samples\preproctrials01.mat",

    r"samples\Suj_501_extero.set"
]

for f in files:

    print("\n====================")

    result = process_record(
        f
    )

    print(
        result["eeg"].shape
    )

    print(
        result["sfreq"]
    )

    print(
        result["channels"][:20]
    )