import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# LOAD
# ============================================================

results = pd.read_csv("ds005385_results.csv")
meta = pd.read_excel("session2_subjects.xlsx")

meta = meta.rename(
    columns={
        "participant_id": "subject"
    }
)

merged = results.merge(
    meta[
        [
            "subject",
            "age",
            "sex"
        ]
    ],
    on="subject",
    how="left"
)

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("Rows:", len(merged))
print("Subjects:", merged["subject"].nunique())
print("Missing Age:", merged["age"].isna().sum())

# ============================================================
# OVERALL RISK
# ============================================================

print("\nOVERALL RISK DISTRIBUTION")
print(
    merged["fusion_percent"].describe()
)

# ============================================================
# SESSION ANALYSIS
# ============================================================

session_stats = (
    merged
    .groupby("session")["fusion_percent"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )
)

print("\nSESSION ANALYSIS")
print(session_stats)

session_stats.to_csv(
    "session_analysis.csv"
)

plt.figure(figsize=(8,5))

plt.plot(
    session_stats.index,
    session_stats["mean"],
    marker="o"
)

plt.title(
    "Average Fusion Risk by Session"
)

plt.ylabel(
    "Fusion Risk (%)"
)

plt.xlabel(
    "Session"
)

plt.grid(True)

plt.savefig(
    "01_session_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ============================================================
# TASK ANALYSIS
# ============================================================

task_stats = (
    merged
    .groupby("task")["fusion_percent"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std"
        ]
    )
)

print("\nTASK ANALYSIS")
print(task_stats)

task_stats.to_csv(
    "task_analysis.csv"
)

plt.figure(figsize=(8,5))

plt.bar(
    task_stats.index,
    task_stats["mean"]
)

plt.title(
    "Eyes Open vs Eyes Closed"
)

plt.ylabel(
    "Fusion Risk (%)"
)

plt.savefig(
    "02_task_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ============================================================
# ACQUISITION ANALYSIS
# ============================================================

acq_stats = (
    merged
    .groupby("acquisition")["fusion_percent"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std"
        ]
    )
)

print("\nPRE VS POST")
print(acq_stats)

acq_stats.to_csv(
    "acquisition_analysis.csv"
)

plt.figure(figsize=(8,5))

plt.bar(
    acq_stats.index,
    acq_stats["mean"]
)

plt.title(
    "Pre vs Post Acquisition"
)

plt.ylabel(
    "Fusion Risk (%)"
)

plt.savefig(
    "03_pre_post_comparison.png",
    bbox_inches="tight"
)

plt.show()

# ============================================================
# AGE ANALYSIS
# ============================================================

subject_age = (
    merged
    .groupby("subject")
    .agg(
        {
            "age":"first",
            "fusion_percent":"mean"
        }
    )
    .reset_index()
)

plt.figure(figsize=(8,5))

plt.scatter(
    subject_age["age"],
    subject_age["fusion_percent"]
)

plt.title(
    "Age vs Average Fusion Risk"
)

plt.xlabel(
    "Age"
)

plt.ylabel(
    "Fusion Risk (%)"
)

plt.grid(True)

plt.savefig(
    "04_age_vs_risk.png",
    bbox_inches="tight"
)

plt.show()

# ============================================================
# AGE GROUP ANALYSIS
# ============================================================

merged["age_group"] = pd.cut(
    merged["age"],
    bins=[
        20,
        40,
        50,
        60,
        70,
        100
    ],
    labels=[
        "20-39",
        "40-49",
        "50-59",
        "60-69",
        "70+"
    ]
)

age_group_stats = (
    merged
    .groupby("age_group")["fusion_percent"]
    .agg(
        [
            "count",
            "mean",
            "median"
        ]
    )
)

print("\nAGE GROUP ANALYSIS")
print(age_group_stats)

plt.figure(figsize=(8,5))

plt.plot(
    age_group_stats.index.astype(str),
    age_group_stats["mean"],
    marker="o"
)

plt.title(
    "Risk Across Age Groups"
)

plt.ylabel(
    "Fusion Risk (%)"
)

plt.savefig(
    "05_age_group_risk.png",
    bbox_inches="tight"
)

plt.show()

# ============================================================
# SUBJECT LEVEL LONGITUDINAL ANALYSIS
# ============================================================

subject_session = (
    merged
    .groupby(
        [
            "subject",
            "session"
        ]
    )["fusion_percent"]
    .mean()
    .reset_index()
)

pivot = subject_session.pivot(
    index="subject",
    columns="session",
    values="fusion_percent"
)

pivot.to_csv(
    "subject_session_risk.csv"
)

print("\nSUBJECTS WITH BOTH SESSIONS")
print(pivot.shape)

# ============================================================
# SESSION CHANGE
# ============================================================

if (
    "ses-1" in pivot.columns
    and
    "ses-2" in pivot.columns
):

    pivot = pivot.dropna()

    pivot["risk_change"] = (
        pivot["ses-2"]
        -
        pivot["ses-1"]
    )

    print("\nRISK CHANGE SUMMARY")
    print(
        pivot["risk_change"].describe()
    )

    improved = (
        pivot["risk_change"] < 0
    ).sum()

    worsened = (
        pivot["risk_change"] > 0
    ).sum()

    unchanged = (
        pivot["risk_change"] == 0
    ).sum()

    print("\nImproved:", improved)
    print("Worsened:", worsened)
    print("Unchanged:", unchanged)

    # --------------------------------------------------------
    # SUBJECT TRAJECTORIES
    # --------------------------------------------------------

    sample = pivot.head(30)

    plt.figure(figsize=(12,6))

    for _, row in sample.iterrows():

        plt.plot(
            [
                "Session 1",
                "Session 2"
            ],
            [
                row["ses-1"],
                row["ses-2"]
            ],
            alpha=0.5
        )

    plt.title(
        "Subject-Level Longitudinal Risk Trajectories"
    )

    plt.ylabel(
        "Fusion Risk (%)"
    )

    plt.grid(True)

    plt.savefig(
        "06_subject_trajectories.png",
        bbox_inches="tight"
    )

    plt.show()

# ============================================================
# HIGH RISK SUBJECTS
# ============================================================

subject_avg = (
    merged
    .groupby("subject")["fusion_percent"]
    .mean()
    .reset_index()
)

high_risk = subject_avg[
    subject_avg["fusion_percent"] >= 75
]

high_risk.to_csv(
    "high_risk_subjects.csv",
    index=False
)

print("\nHIGH RISK SUBJECTS")
print(len(high_risk))

# ============================================================
# SAVE MASTER
# ============================================================

merged.to_csv(
    "merged_longitudinal_results.csv",
    index=False
)

print("\nDONE")
print("Generated:")
print("01_session_comparison.png")
print("02_task_comparison.png")
print("03_pre_post_comparison.png")
print("04_age_vs_risk.png")
print("05_age_group_risk.png")
print("06_subject_trajectories.png")