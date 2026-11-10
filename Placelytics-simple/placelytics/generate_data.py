"""Creates a SYNTHETIC placement dataset for a fictional college (not real data)."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 800
branches = ["CSE", "IT", "ECE", "EEE", "ME", "CE"]
df = pd.DataFrame({
    "student_id": [f"S{i:04d}" for i in range(1, N + 1)],
    "batch_year": rng.choice([2022, 2023, 2024], N),
    "branch": rng.choice(branches, N, p=[.28, .15, .25, .10, .13, .09]),
})
mean_cgpa = {"CSE": 8.0, "IT": 7.9, "ECE": 7.7, "EEE": 7.4, "ME": 7.3, "CE": 7.3}
df["cgpa"] = np.clip(rng.normal(df.branch.map(mean_cgpa), 0.7), 5.0, 9.9).round(2)
df["internships"] = np.minimum(rng.poisson(np.clip(0.5 + 0.35 * (df.cgpa - 7), 0.05, None)), 3)
df["projects"] = np.minimum(rng.poisson(2, N), 5)
df["certifications"] = np.minimum(rng.poisson(1.5, N), 5)

bonus = df.branch.map({"CSE": 1.0, "IT": 0.9, "ECE": 0.2, "EEE": -0.2, "ME": -0.3, "CE": -0.4})
z = -8.2 + 1.05 * df.cgpa + 0.6 * df.internships + 0.15 * df.projects + 0.1 * df.certifications + bonus
df["placed"] = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)

companies = {"Nimbus Infotech": 4.0, "Apex Capital": 9.0, "Orbit Labs": 12.0, "Zenith Motors": 5.5,
             "Lumen Analytics": 8.0, "Cobalt Power": 5.0, "Vertex Semiconductors": 10.0, "Falcon Advisory": 7.0}
base = np.array(list(companies.values()))
rows = df.placed == 1
df["company"] = None
df["package_lpa"] = np.nan
for i in df.index[rows]:
    w = np.exp((df.cgpa[i] - 7.5) * (base - base.mean()) / 6)  # better CGPA -> higher-paying firms
    c = rng.choice(len(base), p=w / w.sum())
    df.loc[i, "company"] = list(companies)[c]
    df.loc[i, "package_lpa"] = round(base[c] * rng.lognormal(0, 0.12), 2)

# a little mess, so there is something to clean
df.loc[rng.random(N) < .02, "cgpa"] = np.nan
df.loc[rows & (rng.random(N) < .01), "package_lpa"] = np.nan
df = pd.concat([df, df.sample(5, random_state=1)], ignore_index=True)
df.to_csv("data/placements.csv", index=False)
print(f"Saved data/placements.csv  ({len(df)} rows, SYNTHETIC)")
