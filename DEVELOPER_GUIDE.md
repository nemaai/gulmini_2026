# Developer Guide

## Prerequisites

- Python 3.11+
- pip
- Git

---

# Repository Setup

Clone the repository:

```bash
git clone <repository_url>
cd gulmini_2026
```

Each module should be installed independently.

Example:

```bash
cd Analysis
pip install -r requirements.txt
```

---

# Module Guide

## Analysis

### Purpose

- EEG loading
- Preprocessing
- Quality assessment
- Biomarker generation

### Install

```bash
cd Analysis
pip install -r requirements.txt
```

### Run

```bash
python validate_all_format.py
```

Dataset processing

```bash
python pipeline/run_dataset.py
```

Single recording

```bash
python pipeline/process_record.py
```

---

## BrainAge_Confidence

### Purpose

Brain Age estimation and confidence scoring.

### Install

```bash
cd BrainAge_Confidence
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

Metrics

```bash
python metrices.py
```

---

## EEG_DL

### Purpose

Deep Learning and Fusion model training.

### Install

```bash
cd EEG_DL
pip install -r requirements.txt
```

### Run

Training

```bash
python trainDLv2.py
```

Evaluation

```bash
python evaluate_model.py
```

Fusion

```bash
python fusionMLDL/fusion_final.py
```

---

## EEGapi_V2

### Purpose

REST APIs for EEG processing and reporting.

### Install

```bash
cd EEGapi_V2
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

---

## Ingest Work

### Purpose

Dataset ingestion and preprocessing.

### Install

```bash
cd "Ingest Work"
pip install -r requirements.txt
```

### Run

Dataset ingestion

```bash
python new_ingest.py
```

Batch ingestion

```bash
python ingest_all_datasets.py
```

Biomarker generation

```bash
python final_create_raw_biomarker.py
```

Training

```bash
python ml_model_train_v1.py
```

---

## New Pipeline

### Purpose

Standalone EEG preprocessing service.

### Install

```bash
cd "New Pipeline"
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

---

## New Training

### Purpose

Training and transfer learning framework.

### Install

```bash
cd "New Training"
pip install -r requirements.txt
```

### Run

Execute the required training or evaluation scripts from the corresponding module.

---

## Spectrogram

### Purpose

Spectrogram-based EEG anomaly analysis.

### Install

```bash
cd Spectrogram
pip install -r requirements.txt
```

### Run

```bash
python scripts/test_edf.py
python scripts/generate_bank.py
python scripts/generate_testbank.py
python scripts/edf_similarity_analysis.py
python scripts/window_ssim_analysis.py
```

---

# General Notes

- Each module is independently executable.
- Outputs and logs are generated within the respective module directories.
- Refer to each module's `README.md` for implementation details and expected outputs.
- Install dependencies using the local `requirements.txt` before running any module.
