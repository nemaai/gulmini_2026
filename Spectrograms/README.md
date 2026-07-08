# EEG Spectrogram Analysis (Proof of Concept)

## Overview

This repository contains a Proof of Concept (PoC) for spectrogram-based EEG anomaly analysis. The objective is to evaluate whether EEG spectrograms can provide additional information for distinguishing normal and high-risk EEG recordings.

The implementation focuses on T4-channel analysis using STFT-based spectrogram generation and SSIM-based similarity scoring.

---

# Repository Structure

```text
Spectrogram/
│
├── generated_specs/                  # Sample generated spectrograms
├── scripts/                          # Processing scripts
│
├── summary.csv                       # Test EDF similarity metrics
├── summary_normal.csv                # Normal EDF similarity metrics
│
├── 01_T4_120s.png                    # Sample high-risk spectrogram
└── 36_T4_120s.png                    # Sample low-risk spectrogram
```

---

# Processing Workflow

```text
EDF
    ↓
Load EEG
    ↓
Extract T4 Channel
    ↓
Bandpass Filter (0.5–30 Hz)
    ↓
Sliding Window Generation
    ↓
STFT Spectrogram Generation
    ↓
Normal Spectrogram Bank
    ↓
SSIM Similarity Analysis
    ↓
EDF-Level Metrics
```

---

# Scripts

### test_edf.py

* Validate EDF loading
* Verify channels and sampling frequency

### generate_spectrogram.py

* Generate a spectrogram from a selected EDF window

### generate_bank.py

* Generate the reference spectrogram bank from normal EEGs

### generate_testbank.py

* Generate spectrograms for test EEGs using the same processing pipeline

### compare_ssim.py

* Compare a single spectrogram with the normal spectrogram bank using SSIM

### edf_similarity_analysis.py

* Compare all windows of an EDF against the normal bank
* Generate EDF-level metrics:

  * Mean SSIM
  * Minimum SSIM
  * Standard Deviation
  * Percentage of Abnormal Windows

### json-csv.py

* Combine all JSON outputs into a single summary CSV

### window_ssim_analysis.py

* Generate window-level SSIM scores for test EEGs

### normal_window_ssim_analysis.py

* Generate window-level SSIM scores for normal EEGs

---

# Processing Parameters

| Parameter          | Value      |
| ------------------ | ---------- |
| Channel            | T4         |
| Bandpass Filter    | 0.5–30 Hz  |
| Window Length      | 20 seconds |
| Step Size          | 10 seconds |
| Spectrogram Method | STFT       |
| Similarity Metric  | SSIM       |

---

# Outputs

**summary.csv**

* EDF-wise similarity metrics for test EEGs.

**summary_normal.csv**

* EDF-wise similarity metrics for normal EEGs.

Sample spectrograms are available for visual inspection in the project root and `generated_specs/`.

---

# Current Findings

* Successfully generated spectrograms from EEG EDF recordings.
* Built a normal spectrogram reference bank.
* Implemented SSIM-based similarity analysis.
* Generated EDF-level and window-level similarity metrics.
* Initial evaluation showed that T4-only spectrogram similarity provides limited separation between low-risk and high-risk EEGs.

---

# Future Work

* Multi-channel spectrogram analysis
* Alternative similarity metrics
* Temporal anomaly analysis
* Advanced anomaly detection methods
* Integration with the existing EEG risk stratification pipeline after validation
