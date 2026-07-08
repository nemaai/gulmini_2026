## Objective

Evaluate whether EEG spectrogram analysis can identify deviations from normal EEG patterns and provide an additional validation layer for EEG risk stratification.


## Workflow

EDF
↓
T4 Channel Extraction
↓
Filtering (0.5–30 Hz)
↓
Sliding Windows (20s, 10s step)
↓
STFT Spectrogram Generation
↓
Normal Spectrogram Bank
↓
SSIM Similarity Analysis
↓
EDF-Level Metrics



## Scripts

### test_edf.py

Validate EDF loading, channels, and sampling frequency.

### generate_spectrogram.py

Generate a spectrogram from a single EDF window.

### generate_bank.py

Generate the Normal Spectrogram Bank from normal EDFs.

Output:
normal_bank/T4/


### generate_testbank.py

Generate the Test Spectrogram Bank from test EDFs.

Output:
test_bank/T4/


### compare_ssim.py

Compare a single spectrogram against the normal bank using SSIM.

### edf_similarity_analysis.py

Compare all test windows against the normal bank and generate:

* Mean SSIM
* Min SSIM
* Std SSIM
* Abnormal Window %

Output:
outputs/*.json


### json-csv.py

Combine JSON outputs into:

outputs/summary.csv


### window_ssim_analysis.py

Generate window-level SSIM scores for temporal anomaly analysis.

### normal_window_ssim_analysis.py

Window-level SSIM analysis for normal EEGs.



## Current Status

Completed:

* EDF loading
* Spectrogram generation
* Normal/Test spectrogram banks
* SSIM similarity scoring
* EDF-level metrics
* Window-level analysis

Observation:

* T4 spectrogram similarity alone showed limited separation between low-risk and high-risk EEGs.
* Further research is required to evaluate alternative similarity methods, multi-channel analysis, and advanced anomaly detection approaches.
