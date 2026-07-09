from pathlib import Path
import numpy as np

MASTER_19 = [
    "Fp1","Fp2",
    "F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5","P3","Pz","P4","T6",
    "O1","O2",
]

SFREQ = 256


def load_adfsu(patient_dir):
    patient_dir = Path(patient_dir)

    signals = {}

    for ch in MASTER_19:
        file = patient_dir / f"{ch}.txt"

        if not file.exists():
            raise FileNotFoundError(file)

        signals[ch] = np.loadtxt(file)

    eeg = np.vstack([signals[ch] for ch in MASTER_19])

    return eeg.astype(np.float32), SFREQ

if __name__ == "__main__":
    eeg, sfreq = load_adfsu("Patient59")

    print("EEG Shape :", eeg.shape)
    print("Sampling  :", sfreq)
    print("dtype     :", eeg.dtype)

    print("\nFirst channel (Fp1)")
    print(eeg[0][:10])