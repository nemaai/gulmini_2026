import boto3
import numpy as np
import io
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import sys
import gc

#############################################
# LOGGING
#############################################

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("training_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.info

#############################################
# CONFIG
#############################################

BUCKET = "dementia-research2025"

PREFIXES = [
    "New EEG Base/EEG/",
    "lead_pipeline_dementia/REEG-BACA-19/Feature/"
]

EPOCHS = 5
BATCH_SIZE = 64
LR = 0.001

#############################################
# AWS S3
#############################################

s3 = boto3.client(
    's3',
    aws_access_key_id=' ',
    aws_secret_access_key=' ',
    region_name='ap-south-1')

#############################################
# MODEL
#############################################

class EEGNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(19,32,5,padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32,64,5,padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64,1)

    def forward(self,x):

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))

        x = self.pool(x).squeeze(-1)

        return self.fc(x)

#############################################
# LIST FILES
#############################################

def list_files():

    files = []

    for prefix in PREFIXES:

        log(f"Scanning prefix: {prefix}")

        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=prefix
        )

        if "Contents" not in response:
            continue

        for obj in response["Contents"]:

            key = obj["Key"]

            if key.endswith(".npy"):
                files.append(key)

    log(f"Total files found: {len(files)}")

    return files

#############################################
# LOAD EEG
#############################################

def load_eeg(key):

    obj = s3.get_object(Bucket=BUCKET, Key=key)

    data = np.load(io.BytesIO(obj["Body"].read()))

    # segments × time × channels -> segments × channels × time
    if data.shape[2] in [19,128]:
        data = np.transpose(data,(0,2,1))

    channels = data.shape[1]

    if channels != 19:
        log(f"Skipping {key} (channels={channels})")
        return None

    return data

#############################################
# NORMALIZE
#############################################

def normalize(X):

    mean = X.mean(axis=2,keepdims=True)
    std = X.std(axis=2,keepdims=True)

    return (X-mean)/(std+1e-6)

#############################################
# TRAINING
#############################################

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log(f"Using device: {device}")

    model = EEGNet().to(device)

    optimizer = optim.Adam(model.parameters(),lr=LR)

    loss_fn = nn.BCEWithLogitsLoss()

    files = list_files()

    for epoch in range(EPOCHS):

        log(f"Starting epoch {epoch+1}")

        for key in files:

            data = load_eeg(key)

            if data is None:
                continue

            X = normalize(data)

            label = 1 if "dementia" in key.lower() else 0

            y = np.full((X.shape[0],1),label)

            X = torch.tensor(X).float()
            y = torch.tensor(y).float()

            dataset = torch.utils.data.TensorDataset(X,y)

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
            )

            for bx,by in loader:

                bx = bx.to(device)
                by = by.to(device)

                pred = model(bx)

                loss = loss_fn(pred,by)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            del X,y,data,loader,dataset
            gc.collect()

        log(f"Epoch {epoch+1} completed")

    torch.save(model.state_dict(),"eeg_dl_model.pt")

    log("Training complete")
    log("Model saved: eeg_dl_model.pt")

#############################################

if __name__ == "__main__":

    log("Starting EEG Deep Learning Training")

    train()