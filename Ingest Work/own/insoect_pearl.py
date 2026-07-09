import mne

raw = mne.io.read_raw_brainvision(
    "pearl_test\sub-01_task-rest_eeg.vhdr",
    preload=False,
    verbose=False,
)

TARGET = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8",
    "T7","C3","Cz","C4","T8",
    "P7","P3","Pz","P4","P8",
    "O1","O2",
]

print("Missing:", [c for c in TARGET if c not in raw.ch_names])