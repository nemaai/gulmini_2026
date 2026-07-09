#!/bin/bash.   
#cOMMAND TO RUN
#./run_training.sh > training.log 2>&1 &tail -f training.log

set -e

echo "======================================="
echo "NEMA TRAINING PIPELINE"
echo "Started: $(date)"
echo "======================================="

# --------------------------------------------------
# Activate Conda
# --------------------------------------------------

# source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate nema
echo $CONDA_DEFAULT_ENV

echo ""
echo "Using Python:"
which python

echo ""
python --version

# --------------------------------------------------
# Move to Project
# --------------------------------------------------

cd /home/ubuntu/eeg_train_pipeline
# --------------------------------------------------
# STEP 0

# 1. Ingest round 1 datasets
python ingest_all_datasets.py --dataset ADFSU    --bucket dementia-research2025
python ingest_all_datasets.py --dataset DS004504  --bucket dementia-research2025
python ingest_all_datasets.py --dataset BrainLat  --bucket dementia-research2025
python ingest_all_datasets.py --dataset P-ADIC-dem --bucket dementia-research2025
python ingest_all_datasets.py --dataset P-ADIC-ctrl --bucket dementia-research2025
python ingest_all_datasets.py --dataset Isfahan   --bucket dementia-research2025
python ingest_caueeg.py   #--> seperate for caueeg


# 2. Standardise (resample + preprocess + QC) Mode : Training
python data_pipeline.py \
    --bucket dementia-research2025 \
    --mode train --all_datasets \
    --raw_prefix nema_final_used/all_npy_raw/

python data_pipeline.py \
    --bucket dementia-research2025 \
    --mode train \
    --dataset CAUEEG \
    --raw_prefix nema_final_used/final_npy_raw/CAUEEG/

    python data_pipeline.py \
    --bucket dementia-research2025 \
    --mode train \
    --dataset test_hardware \
    --raw_prefix nema_final_used/final_npy_raw/test_hardware/

# 3. Split
python split_manifest.py \
    --bucket dementia-research2025 \
    --manifest_prefix nema_final_used/manifests/ \
    --out_prefix      nema_final_used/splits/ \
    --datasets ADFSU DS004504 BrainLat P-ADIC Isfahan CAUEEG --include_review





#4. for ML 4th step is create raw biomarker and running create_raw_fusion. for DL it's build_dataset_cache.py
python -m scripts12.biomarkers.final_create_raw_biomarker #edit the data you want to keep
python -m scripts12.utils.create_fusion_csv
python -m scripts12.training_script.ml_model_train_v1


#FOR DL
python -m scripts12.training_script.build_dataset_cache
python -m scripts12.training_script.train_eegnet_v6


#5. Step is to run calibrate current model and run fusion v6 model
python -m scripts12.fusion_model.calibrate_v6
python -m scripts12.fusion_model.train_fusion_v6


echo ""
echo "======================================="
echo "PIPELINE COMPLETED"
echo "Finished: $(date)"
echo "======================================="

#6 Testing using External cohort code v2 
#Documented in Test.sh


#APPENDIX FOR BETTER UNDERSTANDING DATA FEATURES
python quality check feature 
python quality test biomarker  #(std & mean less on any feature we drop them to reduce unwanted features)


tail -f {local}/logs/eegnet_v6.log          # epoch summaries as they arrive
watch -n 30 cat {local}/logs/training_history.csv   # CSV refreshed every 30s

# added a layer build_dataset_cache. ->to reduce s3 contact


python script/generate_dl_probs.py
python -m scripts12.utils.generate_dl_probs

# Inference works with reports generated as html
python eeg_inference.py \
  --eeg patient.npy \
  --fusion_v5_dir /path/to/model/ \
  --patient "Jane Smith" --age 74