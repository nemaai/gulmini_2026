# Brain Age & Confidence Analysis

## Overview

This repository contains a lightweight post-processing pipeline for generating **Functional Brain Age** and **Analysis Confidence** from EEG-derived biomarkers and model outputs.

The pipeline operates on model prediction outputs and generates additional clinical indicators that can be incorporated into downstream reporting or analysis workflows.

---

# Repository Structure

```text
BrainAge_Confidence/
│
├── main.py                     # Main execution script
├── brain_age.py                # Functional brain age calculation
├── confidence.py               # Analysis confidence calculation
├── metrices.py                 # Performance evaluation
│
├── outputs/
│   └── brain_age_results.csv
│
└── FINAL CSV_XGB.csv           # Input data
```

---

# Workflow

```text
Model Output
      │
      ▼
Load Prediction Data
      │
      ├── Brain Age Estimation
      ├── Confidence Estimation
      │
      ▼
Generate Final Results
      │
      ▼
Performance Evaluation
```

---

# Components

### `main.py`

Main entry point.

Responsibilities:

* Load prediction dataset
* Execute Brain Age estimation
* Execute Confidence estimation
* Generate consolidated output

Run:

```bash
python main.py
```

---

### `brain_age.py`

Computes Functional Brain Age using model outputs and selected EEG-derived parameters.

Returns:

* Estimated Brain Age
* Brain Age Gap
* Brain Age Interpretation

---

### `confidence.py`

Calculates an overall confidence score for the generated prediction.

Returns:

* Confidence Score
* Confidence Category

---

### `metrices.py`

Evaluates generated predictions using standard classification metrics.

Outputs include:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

Run:

```bash
python metrices.py
```

---

# Execution

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Generate Brain Age & Confidence

```bash
python main.py
```

## Evaluate Results

```bash
python metrices.py
```

---

# Outputs

Generated results are stored in:

```text
outputs/
└── brain_age_results.csv
```

The output includes:

* Brain Age
* Brain Age Gap
* Confidence Score
* Confidence Category
* Prediction Summary

---

# Notes

* `main.py` serves as the primary execution script.
* Individual modules are designed to perform independent computations and are orchestrated through the main pipeline.
* Evaluation is performed separately using `metrices.py`.
* Output files are written to the `outputs/` directory for downstream analysis or reporting.
