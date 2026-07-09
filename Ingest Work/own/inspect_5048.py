import mne

raw = mne.io.read_raw_eeglab(
    "sub-01_task-40HzAuditoryEntrainment_eeg.set",
    preload=False,
    verbose=False,
)

print("Sampling:", raw.info["sfreq"])
print("Channels :", len(raw.ch_names))
print(raw.ch_names)

print("\nAnnotations:")
print(raw.annotations)

print("\nDuration:", raw.n_times/raw.info["sfreq"])