# New Training Pipeline

## Overview

This repository contains the training and evaluation workflow for EEG-based cognitive risk prediction. It includes utilities for dataset preparation, biomarker generation, model training, evaluation, transfer learning, and supporting analysis scripts.

The project is organized into independent modules so that data preparation, model development, evaluation, and transfer learning can be executed separately.

---

# Repository Structure

```text
New Training/
│
├── BACA/
│   ├── Dataset preparation
│   ├── Subject filtering
│   ├── Longitudinal analysis
│   ├── Session analysis
│   └── Visualization scripts
│
├── Training/
│   ├── configs/
│   ├── model/
│   ├── outputs/
│   ├── scripts/
│   └── metrices/
│
└── Transfer Learning/
    ├── datasets/
    ├── manifests/
    ├── models/
    ├── outputs/
    ├── logs/
    └── scripts/
```

---

# Project Workflow

```text
Raw EEG Data
        │
        ▼
Dataset Preparation
        │
        ▼
Feature & Biomarker Generation
        │
        ▼
Training Dataset Creation
        │
        ▼
Model Training
        │
        ├── Deep Learning
        ├── Machine Learning
        └── Fusion
        │
        ▼
Evaluation
        │
        ▼
Transfer Learning / Fine-tuning
```

---

# Module Description

## BACA

Contains utilities related to dataset preparation and exploratory analysis.

Includes:

* Subject filtering
* Label preparation
* Session analysis
* Longitudinal analysis
* Visualization scripts
* Supporting utilities

---

## Training

Contains the complete training workflow.

Major components:

### configs/

Project configuration files including:

* Feature configuration
* Reference scaling
* Supporting parameters

### scripts/

Contains utilities for:

* Biomarker generation
* Dataset preparation
* Feature generation
* Model training
* Cross-validation
* Evaluation
* Probability generation
* Testing utilities

### model/

Stores trained models generated during experimentation.

### outputs/

Training outputs and generated datasets.

### metrices/

Training metrics and feature importance reports.

---

## Transfer Learning

Contains utilities for adapting existing models to new datasets.

Includes:

* Fine-tuning scripts
* Base models
* Output generation
* Dataset manifests
* Experiment logs

---

# Training Pipeline

The general workflow consists of:

1. Dataset preparation
2. Feature generation
3. Biomarker computation
4. Model training
5. Performance evaluation
6. Result generation
7. Transfer learning (where applicable)

Each stage is implemented independently to simplify experimentation and future modifications.

---

# Outputs

The repository generates:

* Processed datasets
* Configuration files
* Trained models
* Evaluation reports
* Feature importance reports
* Prediction outputs
* Analysis plots

---

# Current Status

Implemented:

* Dataset preparation utilities
* Biomarker generation pipeline
* Model training workflow
* Model evaluation
* Cross-validation
* Fusion workflow
* Transfer learning framework
* Supporting analysis scripts

---

# Notes

* The repository is organized into modular components to allow independent execution of data preparation, model development, evaluation, and transfer learning.
* Individual scripts are designed to be executed as required depending on the training or evaluation workflow.
* Configuration files, trained models, evaluation outputs, and supporting utilities are maintained separately to simplify experimentation and future development.
