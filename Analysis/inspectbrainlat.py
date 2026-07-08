# inspect_brainlat_coords.py

from eeg_loader.universal_loader import load_eeg

record = load_eeg(
    r"samples\Suj_501_extero.set"
)

raw = record.data

for ch in raw.info["chs"][:10]:

    print(
        ch["ch_name"]
    )

    print(
        ch["loc"][:3]
    )

    print("-"*50)