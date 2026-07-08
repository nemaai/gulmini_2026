import os
import json
import pandas as pd

# ==========================================
# CONFIG
# ==========================================

OUTPUT_DIR = "Spectrogram/outputs"

# ==========================================
# LOAD JSON FILES
# ==========================================

rows = []

json_files = [
    f for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".json")
]

for file in json_files:

    path = os.path.join(
        OUTPUT_DIR,
        file
    )

    with open(path, "r") as f:

        data = json.load(f)

        rows.append(data)

# ==========================================
# CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(rows)

# ==========================================
# SAVE CSV
# ==========================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "summary.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print("\n========================")
print(df)

print("\nSaved CSV:")
print(csv_path)