import mne
import numpy as np

TARGET = [
    "Fp1","Fp2",
    "F7","F3","Fz","F4","F8",
    "T7","C3","Cz","C4","T8",
    "P7","P3","Pz","P4","P8",
    "O1","O2",
]

RENAME = {
    "T7":"T3",
    "T8":"T4",
    "P7":"T5",
    "P8":"T6",
}

raw = mne.io.read_raw_brainvision(
    "sub-87966293_ses-1_task-restEC_eeg.vhdr",
    preload=True,
    verbose=False,
)

print("=" * 60)
print("Original")
print("=" * 60)
print("sfreq :", raw.info["sfreq"])
print("channels :", len(raw.ch_names))

available = raw.ch_names

ch_map = {}

for tch in TARGET:

    for cand in [tch, tch.upper()]:

        if cand in available:
            ch_map[tch] = cand
            break

missing = [t for t in TARGET if t not in ch_map]

print("\nMissing:")
print(missing)

assert len(missing) == 0

raw.pick_channels(
    [ch_map[t] for t in TARGET],
    ordered=True,
)

rename = {
    ch_map[t]: t
    for t in TARGET
    if ch_map[t] != t
}

if rename:
    raw.rename_channels(rename)

raw.rename_channels({
    t: RENAME.get(t, t)
    for t in TARGET
    if t in RENAME
})

print("\nSelected channels")
print(raw.ch_names)

eeg, _ = raw[:]

eeg = eeg.astype(np.float32)

print("\nFinal")
print("=" * 60)
print("Shape :", eeg.shape)
print("dtype :", eeg.dtype)
print("Duration :", eeg.shape[1] / raw.info["sfreq"], "sec")

assert eeg.shape[0] == 19

np.save("tdbrain.npy", eeg)

x = np.load("tdbrain.npy")

print("\nReload")
print("=" * 60)
print(x.shape)
print(x.dtype)

assert x.shape == eeg.shape

print("\nSUCCESS")