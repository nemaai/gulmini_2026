from scipy.io import loadmat
import mne

mat = loadmat("NBT.S0020.100604.ECR1.mat")

eeg = mat["CLEANSignal"]

print(eeg.shape)

info = mne.create_info(
    ch_names=[f"CH{i}" for i in range(eeg.shape[1])],
    sfreq=256,
    ch_types="eeg",
)

raw = mne.io.RawArray(eeg.T, info)

print(raw)