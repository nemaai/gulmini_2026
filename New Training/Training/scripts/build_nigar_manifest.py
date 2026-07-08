import os
import pandas as pd

from sklearn.model_selection import train_test_split

NPY_DIR = "nigar_npy"

rows = []

for file in os.listdir(NPY_DIR):

    if not file.endswith(".npy"):
        continue

    fname = file.lower()

    # ==========================
    # LABELS
    # ==========================

    if fname.startswith("cn"):
        label = 0

    elif (
        fname.startswith("mild")
        or fname.startswith("mod")
        or fname.startswith("sv")
    ):
        label = 1

    else:
        continue

    rows.append({

        "file_source":
            os.path.join(
                NPY_DIR,
                file
            ),

        "binary_label":
            label
    })

df = pd.DataFrame(rows)

print(
    "\nTotal files:",
    len(df)
)

print(
    df["binary_label"]
    .value_counts()
)

# ==========================
# STRATIFIED SPLIT
# ==========================

train_df, test_df = train_test_split(

    df,

    test_size=0.20,

    random_state=42,

    stratify=df[
        "binary_label"
    ]
)

train_df.to_csv(
    "nigar_train.csv",
    index=False
)

test_df.to_csv(
    "nigar_test.csv",
    index=False
)

df.to_csv(
    "nigar_manifest.csv",
    index=False
)

print(
    "\nTrain:",
    len(train_df)
)

print(
    "Test:",
    len(test_df)
)

print(
    "\nSaved:"
)

print(
    "nigar_manifest.csv"
)

print(
    "nigar_train.csv"
)

print(
    "nigar_test.csv"
)