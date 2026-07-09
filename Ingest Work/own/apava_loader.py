from scipy.io import loadmat
import numpy as np

MASTER_19 = [
    "Fp1","Fp2",
    "F7","F3","Fz","F4","F8",
    "T3","C3","Cz","C4","T4",
    "T5","P3","Pz","P4","T6",
    "O1","O2",
]

def reconstruct_midline(name, signals):
    pairs = {
        "Fz": ("F3","F4"),
        "Cz": ("C3","C4"),
        "Pz": ("P3","P4"),
    }

    a, b = pairs[name]
    return (signals[a] + signals[b]) / 2.0


def load_apava(mat_file):

    mat = loadmat(
        mat_file,
        squeeze_me=True,
        struct_as_record=False,
    )

    data = mat["data"]

    labels = list(data.label)

    eeg_trials = []

    for trial in data.trial:

        signals = {
            ch: trial[i]
            for i, ch in enumerate(labels)
        }

        # reconstruct missing midline channels
        for ch in ("Fz", "Cz", "Pz"):
            signals[ch] = reconstruct_midline(ch, signals)

        eeg = np.vstack([signals[ch] for ch in MASTER_19])

        eeg_trials.append(eeg.astype(np.float32))

    return eeg_trials, data.fsample

if __name__ == "__main__":
    trials, sfreq = load_apava("preproctrials01.mat")

    print("Trials :", len(trials))
    print("Shape  :", trials[0].shape)
    print("SFREQ  :", sfreq)