import numpy as np

EMOTIV_CHANNELS = [
    "AF3","F7","F3","FC5","T7","P7","O1",
    "O2","P8","T8","FC6","F4","F8","AF4"
]

def convert_14_to_19(window):

    AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4 = range(14)

    FP1 = (window[AF3] + window[F3]) / 2
    FP2 = (window[AF4] + window[F4]) / 2

    FZ = (window[F3] + window[F4]) / 2
    CZ = (window[FC5] + window[FC6]) / 2
    PZ = (window[P7] + window[P8]) / 2

    return np.stack([
        FP1, FP2,
        window[F7], window[F3], FZ, window[F4], window[F8],
        window[T7], window[FC5], CZ, window[FC6], window[T8],
        window[P7], PZ, window[P8],
        window[O1], window[O2],
        window[T7], window[T8]
    ])

def convert_window(window):

    FP1,FP2,F7,F3,FZ,F4,F8,T3,C3,CZ,C4,T4,T5,P3,PZ,P4,T6,O1,O2 = range(19)

    AF3 = (window[:,FP1] + window[:,F3]) / 2
    AF4 = (window[:,FP2] + window[:,F4]) / 2

    FC5 = (window[:,F3] + window[:,C3]) / 2
    FC6 = (window[:,F4] + window[:,C4]) / 2

    return np.stack([
        AF3,
        window[:,F7],
        window[:,F3],
        FC5,
        window[:,T3],
        window[:,T5],
        window[:,O1],
        window[:,O2],
        window[:,T6],
        window[:,T4],
        FC6,
        window[:,F4],
        window[:,F8],
        AF4
    ], axis=1)


def to_regions(window):

    AF3,F7,F3,FC5,T7,P7,O1,O2,P8,T8,FC6,F4,F8,AF4 = range(14)

    frontal = np.mean(window[:,[AF3,F7,F3,FC5,FC6,F4,F8,AF4]], axis=1)
    parietal = np.mean(window[:,[P7,P8]], axis=1)
    occipital = np.mean(window[:,[O1,O2]], axis=1)
    temporal = np.mean(window[:,[T7,T8]], axis=1)

    central = (frontal + parietal)/2

    return np.stack([
        frontal,
        central,
        parietal,
        occipital,
        temporal
    ], axis=1)

def extract_emotiv_14(raw_eeg, channel_names):

    # HANDLE MISSING CHANNEL NAMES
    if channel_names is None or len(channel_names) == 0:
        print("No channel names found ? skipping emotiv extraction")
        return None, []

    ch_map = {ch.upper(): i for i, ch in enumerate(channel_names)}

    selected = []
    found = []

    for ch in EMOTIV_CHANNELS:

        if ch in ch_map:
            selected.append(raw_eeg[ch_map[ch]])
            found.append(ch)
        else:
            selected.append(None)

    # fill missing channels
    for i in range(len(selected)):
        if selected[i] is None:
            selected[i] = np.mean(raw_eeg, axis=0)

    eeg_14 = np.stack(selected)

    return eeg_14, found

def map_19_to_emotiv(raw_eeg, channel_names):

    if channel_names is None:
        return None, []

    ch_map = {ch.upper(): i for i, ch in enumerate(channel_names)}

    try:
        # Standard 10-20 mapping
        FP1 = raw_eeg[ch_map["FP1"]]
        FP2 = raw_eeg[ch_map["FP2"]]
        F3  = raw_eeg[ch_map["F3"]]
        F4  = raw_eeg[ch_map["F4"]]
        F7  = raw_eeg[ch_map["F7"]]
        F8  = raw_eeg[ch_map["F8"]]
        C3  = raw_eeg[ch_map["C3"]]
        C4  = raw_eeg[ch_map["C4"]]
        P3  = raw_eeg[ch_map["P3"]]
        P4  = raw_eeg[ch_map["P4"]]
        O1  = raw_eeg[ch_map["O1"]]
        O2  = raw_eeg[ch_map["O2"]]

        # Build approximations
        AF3 = (FP1 + F3) / 2
        AF4 = (FP2 + F4) / 2

        FC5 = (F3 + C3) / 2
        FC6 = (F4 + C4) / 2

        T7 = raw_eeg[ch_map.get("T3", ch_map.get("T7"))]
        T8 = raw_eeg[ch_map.get("T4", ch_map.get("T8"))]

        P7 = raw_eeg[ch_map.get("T5", ch_map.get("P7"))]
        P8 = raw_eeg[ch_map.get("T6", ch_map.get("P8"))]

        emotiv_14 = np.stack([
            AF3, F7, F3, FC5, T7, P7, O1,
            O2, P8, T8, FC6, F4, F8, AF4
        ])

        found = EMOTIV_CHANNELS.copy()

        print("? Mapped 19-channel ? Emotiv 14")

        return emotiv_14, found

    except Exception as e:
        print("19?14 mapping failed:", str(e))
        return None, []