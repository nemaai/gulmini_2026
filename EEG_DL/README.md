# EEG Spectrogram Analysis (Proof of Concept)

## Overview

This repository contains a Proof of Concept (PoC) for spectrogram-based EEG anomaly analysis. The objective is to evaluate whether spectrogram-derived features can provide additional information for EEG anomaly detection and support future EEG risk assessment workflows.

The current implementation focuses on generating spectrogram representations from EEG recordings, constructing a reference spectrogram bank, and performing similarity-based analysis at both window and recording levels.

---

# Project Workflow

```text
EDF
│
├── EEG Loading
├── Channel Selection
├── Signal Filtering
├── Sliding Window Generation
├── STFT Spectrogram Generation
│
├── Reference Spectrogram Bank
│
└── Similarity Analysis
        │
        ├── Window-Level Analysis
        ├── Recording-Level Analysis
        └── Summary Generation
```

---

# Repository Structure

```text
EEG_DL/
│
├── core/
│   ├── EEG loading
│   ├── preprocessing
│   ├── channel utilities
│   ├── biomarker computation
│   ├── scoring
│   └── supporting configuration
│
├── fusionMLDL/
│   ├── fusion models
│   ├── evaluation
│   └── supporting resources
│
├── scripts
├── trained models
├── evaluation outputs
└── configuration files
```

---

# Modules

### Core

Responsible for EEG processing and feature generation.

Includes:

* EEG loading
* Signal preprocessing
* Channel handling
* Biomarker computation
* Score generation
* Risk-band generation

---

### Deep Learning

Implements model training, inference and evaluation utilities for EEG-based prediction.

---

### Fusion

Contains the hybrid ML + DL implementation along with evaluation scripts and supporting utilities.

---

# Processing Pipeline

The implementation follows a sequential processing pipeline:

1. Load EEG recording
2. Preprocess EEG signal
3. Select required channels
4. Generate spectral representation
5. Extract required features
6. Perform similarity / model inference
7. Generate output metrics
8. Export evaluation results

---

# Outputs

The project generates:

* Spectrogram representations
* Recording-level metrics
* Window-level metrics
* Evaluation summaries
* Model outputs
* Supporting logs

---

# Current Status

Completed:

* EEG preprocessing workflow
* Spectrogram generation
* Reference bank creation
* Similarity-based analysis
* Recording-level evaluation
* Window-level evaluation
* Result summarization

---

# Future Scope

Potential areas for further investigation include:

* Multi-channel spectral analysis
* Alternative similarity measures
* Advanced anomaly detection methods
* Temporal analysis of EEG recordings
* Integration into the complete EEG inference workflow

---

# Notes

This repository represents the current Proof of Concept implementation and serves as the baseline for future development and evaluation. The overall structure has been kept modular to simplify experimentation, benchmarking and integration of additional approaches.
