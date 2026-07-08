# New EEG Processing Pipeline

## Overview

This repository contains the standalone EEG preprocessing pipeline used to standardize EEG recordings before downstream biomarker extraction and model inference.

The pipeline accepts an EEG recording (`.npy`) through a REST API, performs preprocessing and quality checks, and generates standardized outputs along with processing metadata.

---

# Repository Structure

```text
New Pipeline/
│
├── app.py
├── requirements.txt
│
├── routes/
│   └── pipeline_routes.py
│
├── services/
│   └── pipeline_service.py
│
├── data_pipeline/
│   └── data_pipeline.py
│
└── storage/
    ├── input/
    └── output/
```

---

# Workflow

```text
Client
    │
POST /api/pipeline/process
    │
    ▼
pipeline_routes.py
    │
    ▼
pipeline_service.py
    │
    ▼
data_pipeline.py
    │
    ├── Load Input
    ├── EEG Standardization
    ├── Signal Preprocessing
    ├── Quality Check
    ├── Metadata Generation
    └── Save Outputs
```

---

# Components

## app.py

Application entry point.

* Initializes Flask application
* Registers API routes
* Starts the service

---

## routes/pipeline_routes.py

Defines the pipeline API.

Current endpoint:

```
POST /api/pipeline/process
```

Receives an input `.npy` file and forwards it to the processing service.

---

## services/pipeline_service.py

Service layer responsible for:

* Creating a unique processing ID
* Managing input/output folders
* Saving uploaded files
* Invoking the processing pipeline
* Returning API responses

---

## data_pipeline/data_pipeline.py

Core EEG processing module.

Current processing includes:

* EEG loading
* Channel standardization
* Signal preprocessing
* Quality control
* Metadata generation
* Processed output generation

This module contains the primary implementation of the preprocessing pipeline and can also be executed independently if required.

---

# Storage

```
storage/

input/
    <processing_id>/
        input.npy

output/
    <processing_id>/
        processed.npy
        metadata.json
        qc.json
```

Each API request creates a unique processing directory for storing inputs and generated outputs.

---

# API

### Process EEG

```
POST /api/pipeline/process
```

Input:

* EEG `.npy` file

Response:

* Processing ID
* Status
* Processing outputs returned by the pipeline

---

# Outputs

The pipeline generates:

* Processed EEG (`processed.npy`)
* Processing metadata (`metadata.json`)
* Quality assessment (`qc.json`)

---

# Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

# Notes

* The repository is structured using a layered architecture (**Routes → Services → Processing Pipeline**).
* All EEG preprocessing logic resides in `data_pipeline.py`.
* Input and output files are isolated using a unique processing ID for each request.
* This repository serves as the standalone preprocessing stage and can be integrated with downstream biomarker extraction and inference modules.
