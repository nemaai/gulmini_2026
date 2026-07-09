import mne
import numpy as np

TARGET_PEARL = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T7","C3","Cz","C4","T8",
    "P7","P3","Pz","P4","P8",
    "O1","O2",
]

RENAME_PEARL = {
    "T7":"T3",
    "T8":"T4",
    "P7":"T5",
    "P8":"T6",
}

raw = mne.io.read_raw_brainvision(
    "pearl_test/sub-01_task-rest_eeg.vhdr",
    preload=True,
    verbose=False,
)

print("=" * 60)
print("Original")
print("=" * 60)
print("sfreq:", raw.info["sfreq"])
print("channels:", len(raw.ch_names))
print(raw.ch_names[:25])

available = raw.ch_names

ch_map = {}

for tch in TARGET_PEARL:

    for cand in [tch, tch.upper()]:

        if cand in available:
            ch_map[tch] = cand
            break

    if tch not in ch_map:

        for src, dst in RENAME_PEARL.items():

            if dst == tch and src in available:
                ch_map[tch] = src
                break

missing = [x for x in TARGET_PEARL if x not in ch_map]

print("\nMissing channels:")
print(missing)

assert len(missing) == 0

raw.pick_channels(
    [ch_map[t] for t in TARGET_PEARL],
    ordered=True,
)

rename = {
    ch_map[t]: t
    for t in TARGET_PEARL
    if ch_map[t] != t
}

if rename:
    raw.rename_channels(rename)

raw.rename_channels({
    t: RENAME_PEARL.get(t, t)
    for t in TARGET_PEARL
    if t in RENAME_PEARL
})

print("\nAfter selection")
print(raw.ch_names)

sfreq = raw.info["sfreq"]

ec_onset = int(
    max(
        0,
        raw.n_times / sfreq - 360
    ) * sfreq
)

eeg, _ = raw[:, ec_onset:]

print("\nFinal shape:", eeg.shape)
print("Duration:", eeg.shape[1] / sfreq, "sec")
print("dtype:", eeg.dtype)

assert eeg.shape[0] == 19

print("\nSUCCESS")