# EEG Ingest & Training Pipeline

## Overview

This repository contains the EEG data ingestion, preprocessing, biomarker generation and model training workflow used for preparing datasets from multiple EEG sources into a unified format suitable for downstream analysis and model development.

The pipeline supports dataset-specific ingestion, standardized preprocessing, biomarker generation and model training through modular scripts.

---

# Repository Structure

```text
Ingest Work/
│
├── config.py                      # Project configuration
├── new_ingest.py                  # Main ingestion entry point
├── ingest_all_datasets.py         # Batch ingestion
├── ingest_new_datasets.py         # Ingestion for new datasets
├── ingest_nicolet.py              # Nicolet dataset ingestion
├── convert_edf_batch.py           # EDF conversion utility
│
├── data_pipeline.py               # EEG preprocessing pipeline
├── create_raw_biomarker_csv.py    # Biomarker generation
├── final_create_raw_biomarker.py  # Final biomarker generation
│
├── ml_model_train_v1.py           # Model training
├── run_training.sh                # Training execution script
│
├── manifest_*.csv                 # Dataset manifests
├── BIOMARKERS.csv                 # Biomarker definitions
│
├── model/                         # Trained models
├── metrics/                       # Evaluation metrics
├── outputs/                       # Generated outputs
└── own/                           # Dataset-specific utilities
```

---

# Workflow

```text
Raw EEG Dataset
        │
        ▼
Dataset Ingestion
        │
        ▼
Format Conversion
        │
        ▼
EEG Preprocessing
        │
        ▼
Biomarker Generation
        │
        ▼
Training Dataset Creation
        │
        ▼
Model Training
        │
        ▼
Evaluation
```

---

# Execution

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Ingest Dataset

```bash
python new_ingest.py
```

or

```bash
python ingest_all_datasets.py
```

---

## Convert EDF Files

```bash
python convert_edf_batch.py
```

---

## Generate Biomarkers

```bash
python final_create_raw_biomarker.py
```

or

```bash
python create_raw_biomarker_csv.py
```

---

## Train Model

```bash
python ml_model_train_v1.py
```

or

```bash
bash run_training.sh
```

---

# Components

### Ingestion

Responsible for loading EEG recordings from supported datasets and converting them into the standardized processing format.

### Data Pipeline

Handles preprocessing and intermediate processing required before biomarker generation.

### Biomarker Generation

Computes EEG biomarkers and exports structured feature datasets for downstream model training.

### Model Training

Uses generated biomarker datasets to train and evaluate machine learning models.

### Dataset Utilities

Dataset-specific scripts and manifests are provided for handling individual datasets and supporting ingestion workflows.

---

# Outputs

The pipeline generates:

* Processed EEG data
* Biomarker datasets
* Training-ready feature files
* Trained models
* Evaluation metrics
* Processing outputs

---

# Current Status

Implemented:

* Multi-dataset ingestion
* EEG preprocessing pipeline
* Batch EDF conversion
* Biomarker generation
* Training dataset preparation
* Machine learning model training
* Model evaluation

---

# Notes

* The repository is organized into independent modules for ingestion, preprocessing, biomarker generation and model training.
* Dataset manifests are maintained separately to simplify dataset management.
* Individual scripts may be executed independently depending on the required workflow.
* Generated outputs, trained models and evaluation metrics are stored in their respective directories.
