import pandas as pd

df = pd.read_excel("BACA/session2_subjects.xlsx")

exclude = sorted([
    int(x.replace("sub-", ""))
    for x in df["participant_id"]
])

print(len(exclude))

import json

with open("exclude_subjects.json", "w") as f:
    json.dump(exclude, f)