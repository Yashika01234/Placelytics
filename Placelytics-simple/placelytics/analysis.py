"""Placelytics - simple placement analysis (SYNTHETIC data).
In VS Code each '# %%' block can be run on its own ("Run Cell"), like a notebook. Or just run: python analysis.py
"""
# %% 1. Load
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("data/placements.csv")
print(df.shape)
print(df.head())

# %% 2. Clean
print("Duplicates:", df.duplicated().sum())
print("Missing values:\n", df.isna().sum())
df = df.drop_duplicates()
df["cgpa"] = df.cgpa.fillna(df.groupby("branch").cgpa.transform("median"))   # fill missing CGPA with branch median
placed = df[df.placed == 1].copy()
placed["package_lpa"] = placed.package_lpa.fillna(placed.groupby("company").package_lpa.transform("median"))
df.loc[placed.index, "package_lpa"] = placed.package_lpa
print("After cleaning:", df.shape)

# %% 3. Key numbers
print(f"Overall placement rate : {df.placed.mean() * 100:.1f}%")
print(f"Median package         : {placed.package_lpa.median():.2f} LPA")
print(f"Average package        : {placed.package_lpa.mean():.2f} LPA")
print(f"Highest package        : {placed.package_lpa.max():.2f} LPA")

# %% 4. Branch-wise placement
by_branch = df.groupby("branch").agg(students=("student_id", "size"), placed=("placed", "sum"),
                                     avg_cgpa=("cgpa", "mean"), median_lpa=("package_lpa", "median"))
by_branch["placement_pct"] = (100 * by_branch.placed / by_branch.students).round(1)
by_branch = by_branch.sort_values("placement_pct", ascending=False)
print(by_branch.round(2))

# %% 5. Charts (saved to the charts/ folder)
fig, ax = plt.subplots(figsize=(7, 4)); by_branch.placement_pct.plot.bar(ax=ax, color="#6C8CFF")
ax.set_title("Placement % by branch"); ax.set_ylabel("%"); plt.xticks(rotation=0); fig.tight_layout(); fig.savefig("charts/1_placement_by_branch.png", dpi=130)

fig, ax = plt.subplots(figsize=(7, 4)); placed.package_lpa.plot.hist(bins=20, ax=ax, color="#2DD4BF")
ax.axvline(placed.package_lpa.median(), color="red", ls="--", label="median"); ax.legend()
ax.set_title("Package distribution (LPA)"); fig.tight_layout(); fig.savefig("charts/2_package_distribution.png", dpi=130)

fig, ax = plt.subplots(figsize=(7, 4)); df.boxplot(column="cgpa", by="placed", ax=ax)
ax.set_title("CGPA: not placed (0) vs placed (1)"); plt.suptitle(""); fig.tight_layout(); fig.savefig("charts/3_cgpa_vs_placed.png", dpi=130)

yr = df.groupby("batch_year").placed.mean() * 100
fig, ax = plt.subplots(figsize=(7, 4)); yr.plot(marker="o", ax=ax); ax.set_title("Placement % by year"); ax.set_xticks(yr.index)
fig.tight_layout(); fig.savefig("charts/4_year_trend.png", dpi=130)

top = placed.company.value_counts()
fig, ax = plt.subplots(figsize=(7, 4)); top.sort_values().plot.barh(ax=ax, color="#F59E0B"); ax.set_title("Offers by company")
fig.tight_layout(); fig.savefig("charts/5_top_recruiters.png", dpi=130)

# %% 6. Does an internship help?
df["has_internship"] = df.internships > 0
intern = df.groupby("has_internship").placed.mean() * 100
print("Placement % without / with internship:\n", intern.round(1))
# controlling for CGPA (a fair comparison): same CGPA band, with vs without internship
df["cgpa_band"] = pd.cut(df.cgpa, [0, 7, 8, 10], labels=["<7", "7-8", "8+"])
fair = df.pivot_table(index="cgpa_band", columns="has_internship", values="placed", aggfunc="mean", observed=True) * 100
print("Placement % by CGPA band:\n", fair.round(1))

# %% 7. Insights
best, worst = by_branch.index[0], by_branch.index[-1]
gap = (placed.package_lpa.mean() / placed.package_lpa.median() - 1) * 100
print(f"- {best} has the highest placement rate ({by_branch.placement_pct.iloc[0]}%), {worst} the lowest ({by_branch.placement_pct.iloc[-1]}%).")
print(f"- Placed students have a higher average CGPA ({placed.cgpa.mean():.2f}) than non-placed ({df[df.placed == 0].cgpa.mean():.2f}).")
print(f"- Students with an internship are placed {intern[True]:.0f}% of the time vs {intern[False]:.0f}% without.")
if abs(gap) >= 5:
    print(f"- Mean package is {gap:.0f}% above the median, so a few high offers pull the average up. Prefer the median.")
else:
    print(f"- Mean ({placed.package_lpa.mean():.1f}) and median ({placed.package_lpa.median():.1f}) packages are close, so the distribution is fairly even.")
