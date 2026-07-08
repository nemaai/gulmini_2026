import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "/home/ubuntu/training/outputs/fusion_final_clean_v2.csv"
)

df = df[df["binary_label"] != -1].copy()

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["binary_label"],
    random_state=42
)

train_df.to_csv(
    "/home/ubuntu/training/outputs/dl_train_manifest.csv",
    index=False
)

test_df.to_csv(
    "/home/ubuntu/training/outputs/dl_test_manifest.csv",
    index=False
)

print("Train:", len(train_df))
print("Test :", len(test_df))