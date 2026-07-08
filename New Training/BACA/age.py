import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# LOAD
# =====================================================

df = pd.read_csv("merged_longitudinal_results.csv")

# =====================================================
# AGE GROUPS
# =====================================================

df["age_group"] = pd.cut(
    df["age"],
    bins=[20, 40, 50, 60, 70, 100],
    labels=["20-39", "40-49", "50-59", "60-69", "70+"],
    right=False
)

# =====================================================
# SESSION-WISE AGE ANALYSIS
# =====================================================

age_session = (
    df.groupby(
        ["age_group", "session"]
    )["fusion_percent"]
    .mean()
    .reset_index()
)

print("\nAGE GROUP × SESSION")
print(age_session)

# =====================================================
# PIVOT
# =====================================================

pivot = age_session.pivot(
    index="age_group",
    columns="session",
    values="fusion_percent"
)

print("\nPIVOT TABLE")
print(pivot)

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(10,6))

plt.plot(
    pivot.index,
    pivot["ses-1"],
    marker="o",
    linewidth=3,
    label="Session 1"
)

plt.plot(
    pivot.index,
    pivot["ses-2"],
    marker="o",
    linewidth=3,
    label="Session 2"
)

plt.title(
    "Session-wise Dementia Risk Across Age Groups",
    fontsize=18
)

plt.xlabel("Age Group")
plt.ylabel("Average Fusion Risk (%)")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "age_group_session_comparison.png",
    dpi=300
)

plt.show()

# =====================================================
# SUMMARY
# =====================================================

print("\nSESSION DIFFERENCE")

pivot["difference"] = (
    pivot["ses-2"]
    - pivot["ses-1"]
)

print(
    pivot[["ses-1", "ses-2", "difference"]]
)

print(
    "\nLargest Increase:"
)

print(
    pivot["difference"].idxmax(),
    round(
        pivot["difference"].max(),
        2
    )
)

print(
    "\nLargest Decrease:"
)

print(
    pivot["difference"].idxmin(),
    round(
        pivot["difference"].min(),
        2
    )
)