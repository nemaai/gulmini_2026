# EEG Analysis Pipeline

## Overview

This repository contains the EEG analysis and preprocessing pipeline used to load EEG recordings from multiple formats, standardize them into a common representation, perform preprocessing and quality assessment, and prepare the data for downstream biomarker extraction and model inference.

The implementation is modular, allowing individual components to be executed independently or as part of the complete processing workflow.

---

# Repository Structure

```text
Analysis/
│
├── core/                    # EEG quality assessment & biomarkers
├── eeg_loader/              # Universal EEG loading framework
│   ├── adapters/            # Format-specific loaders
│   ├── channel_mapping.py
│   ├── format_detector.py
│   ├── standardize.py
│   ├── resample.py
│   └── universal_loader.py
│
├── pipeline/                # End-to-end processing pipeline
├── samples/                 # Sample EEG files
├── temp/                    # Temporary outputs
│
├── inspect*.py              # Inspection utilities
├── validate_all_format.py   # Format validation
└── test_quality_input.py    # Quality pipeline testing
```

---

# Processing Workflow

```text
EEG Recording
        │
        ▼
Format Detection
        │
        ▼
Format Adapter
        │
        ▼
Universal Loader
        │
        ▼
Channel Standardization
        │
        ▼
Resampling
        │
        ▼
Quality Assessment
        │
        ▼
Biomarker Generation
        │
        ▼
Processed Output
```

---

# Folder Description

### core/

Contains EEG processing modules used after data loading.

Includes:

* EEG Quality Assessment
* Sleep Marker Detection
* Biomarker Computation

---

### eeg_loader/

Universal EEG loading framework supporting multiple EEG formats.

Responsibilities:

* Format detection
* Dataset-specific adapters
* Channel mapping
* Standardization
* Resampling
* Unified EEG object generation

Supported formats include:

* EDF
* EEGLAB
* BrainVision
* MAT
* Dataset-specific formats

---

### pipeline/

Implements the complete processing workflow.

Main modules:

* `process_record.py` – Process a single recording
* `run_dataset.py` – Batch dataset processing
* `quality_adapter.py` – Integrates quality assessment
* `save_processed.py` – Stores processed outputs

---

### samples/

Contains sample EEG recordings for testing supported formats.

---

# Execution

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Validate Supported Formats

```bash
python validate_all_format.py
```

---

## Test Quality Pipeline

```bash
python test_quality_input.py
```

---

## Process Dataset

```bash
python pipeline/run_dataset.py
```

---

## Process Single Recording

```bash
python pipeline/process_record.py
```

---

## Inspection Utilities

Run individual inspection scripts when validating adapters or checking preprocessing behaviour.

Example:

```bash
python inspect_standardization.py
```

---

# Outputs

The pipeline generates:

* Standardized EEG recordings
* Quality assessment results
* Biomarker-ready EEG data
* Processed outputs
* Processing logs

---

# Current Status

Implemented:

* Multi-format EEG loading
* Dataset-specific adapters
* Automatic format detection
* Channel standardization
* Resampling utilities
* EEG quality assessment
* Sleep marker detection
* Biomarker preprocessing
* Dataset and single-record processing workflows

---

# Notes

* The repository follows a modular architecture separating EEG loading, preprocessing, quality assessment, and pipeline execution.
* Individual components can be executed independently for testing or validation.
* Processing scripts inside the `pipeline/` directory provide entry points for single-record and batch execution.
