import numpy as np
from pathlib import Path

CHANNELS_GENEEG = [
    "Fp1","Fp2",
    "F3","F4",
    "F7","F8",
    "C3","C4",
    "P3","P4",
    "O1","O2",
    "T3","T4",
    "Fz","Cz","Pz",
]

MASTER_19 = [
    "Fp1","Fp2",
    "F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5",
    "P3","Pz","P4",
    "T6",
    "O1","O2",
]

eeg = np.loadtxt("ctrl_0001.eeg").T.astype(np.float32)

print("=" * 60)
print("Original")
print("=" * 60)
print("Shape :", eeg.shape)
print("dtype :", eeg.dtype)

signals = {
    ch: eeg[i]
    for i, ch in enumerate(CHANNELS_GENEEG)
}

harmonised = []

missing = []

for ch in MASTER_19:

    if ch in signals:

        harmonised.append(signals[ch])

    elif ch in ("T5","T6"):

        harmonised.append(
            np.zeros_like(eeg[0], dtype=np.float32)
        )

        missing.append(ch)

    else:

        raise RuntimeError(ch)

eeg19 = np.vstack(harmonised)

print("\nConverted")
print("=" * 60)
print("Shape :", eeg19.shape)
print("dtype :", eeg19.dtype)
print("Missing :", missing)

assert eeg19.shape[0] == 19
assert eeg19.dtype == np.float32
assert missing == ["T5","T6"]

np.save("ctrl_0001.npy", eeg19)

x = np.load("ctrl_0001.npy")

print("\nReload")
print("=" * 60)
print(x.shape)
print(x.dtype)

assert x.shape == eeg19.shape

print("\nSUCCESS")