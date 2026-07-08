import pandas as pd

df = pd.read_excel("BACA/session2_subjects.xlsx")

exclude = {
    int(x.replace("sub-", ""))
    for x in df["participant_id"]
}

keep = []

for i in range(1, 609):
    if i not in exclude:
        keep.append(i)

print("Keep:", len(keep))

with open("keep_subjects.txt", "w") as f:
    for k in keep:
        f.write(f"{k:03d}\n")