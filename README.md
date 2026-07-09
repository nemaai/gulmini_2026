# NemaAI EEG Analysis Platform (2026)

## Overview

This repository contains the research, development, training and deployment components of the NemaAI EEG Platform. The project is organized into independent modules covering EEG ingestion, preprocessing, analysis, model training, inference, API services and supporting research initiatives.

Each module is self-contained with its own documentation, dependencies and execution workflow.

---

# Repository Structure

```
gulmini_2026/
│
├── Analysis/
├── BrainAge_Confidence/
├── EEG_DL/
├── EEGapi_V2/
├── Ingest Work/
├── New Pipeline/
├── New Training/
└── Spectrogram/
```

---

# Module Overview

| Module | Description |
|---------|-------------|
| Analysis | EEG loading, preprocessing, quality assessment and biomarker generation |
| BrainAge_Confidence | Functional Brain Age estimation and confidence analysis |
| EEG_DL | Deep Learning models, Fusion models and evaluation |
| EEGapi_V2 | Flask backend APIs for EEG processing and reporting |
| Ingest Work | Dataset ingestion, preprocessing and training utilities |
| New Pipeline | Standalone EEG preprocessing pipeline |
| New Training | Training framework and transfer learning workflows |
| Spectrogram | Spectrogram-based EEG anomaly analysis (PoC) |

Each module contains its own `README.md` and `requirements.txt`.

---

# General Workflow

```
Dataset
    │
    ▼
Ingest Work
    │
    ▼
Analysis
    │
    ▼
New Pipeline
    │
    ▼
EEG_DL / New Training
    │
    ▼
BrainAge_Confidence
    │
    ▼
EEGapi_V2
```

---

# Installation

Each module maintains an independent environment.

Install dependencies from the respective module:

```bash
cd Analysis
pip install -r requirements.txt
```

Refer to the module-specific README for execution instructions.

---

# Documentation

Each module includes:

- README.md
- requirements.txt

For setup instructions, execution commands and project structure, refer to:

```
DEVELOPER_GUIDE.md
```

This guide has been prepared to facilitate a smooth technical handover and enable future development and maintenance of the repository.
