"""Seeded synthetic data generator for the fictional XYZ College.

EVERYTHING produced here is synthetic. Assumptions are documented in README.md (section 'Synthetic-data assumptions').
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import BRANCHES, RAW, SEED, SKILLS, YEARS

BR_SHARE = [.22, .15, .20, .15, .17, .11]
BR_CGPA = {"CSE": 8.0, "IT": 7.9, "ECE": 7.7, "EEE": 7.4, "ME": 7.3, "CE": 7.3}
BR_INTERN = {"CSE": 1.0, "IT": .95, "ECE": .8, "EEE": .6, "ME": .55, "CE": .5}
# P(student has skill | branch), order = config.SKILLS
PROFILE = {
    "CSE": [.70, .55, .50, .50, .65, .30, .40, .20, .15, .30, .35, .08, .03, .02, .01],
    "IT":  [.65, .55, .40, .55, .55, .25, .50, .25, .20, .30, .35, .06, .02, .02, .01],
    "ECE": [.45, .25, .45, .30, .30, .15, .15, .10, .10, .30, .35, .55, .35, .05, .10],
    "EEE": [.35, .20, .30, .25, .20, .10, .08, .06, .10, .30, .35, .35, .10, .10, .45],
    "ME":  [.25, .10, .20, .20, .12, .06, .05, .04, .10, .35, .35, .10, .02, .65, .10],
    "CE":  [.20, .08, .10, .20, .08, .04, .05, .03, .12, .40, .40, .03, .01, .60, .05],
}
TREND_SKILLS = {"Python", "SQL", "Machine Learning", "Cloud", "Data Analytics"}  # supply drifts up slowly
ALL, TECH = BRANCHES, ["CSE", "IT", "ECE"]
ROLES = {  # role: (skill pool, base LPA for tier-2, allowed branches)
    "Software Engineer": (["DSA", "Java", "Python", "C++", "SQL", "Web Dev"], 8.0, ["CSE", "IT", "ECE", "EEE"]),
    "QA Engineer": (["Java", "SQL", "Python", "Communication"], 5.5, TECH),
    "Web Developer": (["Web Dev", "Java", "SQL", "DSA"], 6.0, TECH),
    "Cloud/DevOps Engineer": (["Cloud", "Python", "DSA", "SQL"], 8.5, TECH),
    "Data Scientist": (["Python", "Machine Learning", "SQL", "DSA"], 10.0, ["CSE", "IT", "ECE", "EEE"]),
    "Data Analyst": (["SQL", "Excel", "Data Analytics", "Python", "Communication"], 7.0, ALL),
    "Business Analyst": (["Excel", "SQL", "Communication", "Data Analytics"], 7.0, ALL),
    "Embedded Engineer": (["Embedded Systems", "C++", "Python"], 6.5, ["ECE", "EEE", "CSE"]),
    "VLSI Design Engineer": (["VLSI", "Embedded Systems", "C++"], 9.0, ["ECE", "EEE"]),
    "Design Engineer": (["CAD", "Excel", "Communication"], 5.0, ["ME", "CE", "EEE"]),
    "Power Systems Engineer": (["Power Systems", "Excel", "Communication"], 5.5, ["EEE"]),
    "Site Engineer": (["CAD", "Excel", "Communication"], 4.5, ["CE", "ME"]),
    "Graduate Engineer Trainee": (["Communication", "Excel", "CAD"], 4.8, ALL),
}
GROWTH_ROLES = {"Data Scientist", "Cloud/DevOps Engineer", "Data Analyst"}
INDUSTRIES = {  # industry: (name suffix, roles, hiring multiplier per year 2020..2024)
    "IT Services": ("Infotech", ["Software Engineer", "QA Engineer", "Web Developer", "Cloud/DevOps Engineer"], [.85, 1.1, 1.2, .7, .8]),
    "Product Software": ("Labs", ["Software Engineer", "Data Scientist", "Cloud/DevOps Engineer", "Web Developer"], [.8, 1.0, 1.2, .75, .9]),
    "FinTech & BFSI": ("Capital", ["Data Analyst", "Business Analyst", "Software Engineer"], [.8, 1.0, 1.1, 1.2, 1.35]),
    "Consulting & Analytics": ("Advisory", ["Data Analyst", "Business Analyst", "Data Scientist"], [.9, 1.0, 1.1, 1.0, 1.1]),
    "Core Engineering": ("Engineering", ["Design Engineer", "Graduate Engineer Trainee", "Power Systems Engineer", "Site Engineer"], [1.0, 1.0, .95, .8, .65]),
    "Manufacturing & Auto": ("Motors", ["Design Engineer", "Graduate Engineer Trainee", "QA Engineer"], [.9, 1.0, 1.0, .9, .85]),
    "Electronics & Semiconductor": ("Semiconductors", ["Embedded Engineer", "VLSI Design Engineer", "Graduate Engineer Trainee"], [.9, 1.0, 1.1, 1.15, 1.25]),
    "Energy & Infra": ("Power & Infra", ["Power Systems Engineer", "Site Engineer", "Graduate Engineer Trainee"], [1.0, 1.0, 1.0, .85, .75]),
}
PREFIX = ["Nimbus", "Quanta", "Veridian", "Apex", "Solace", "Kestrel", "Orbit", "Lumen", "Ardent", "Zenith", "Halcyon",
          "Meridian", "Cobalt", "Vertex", "Aurora", "Titan", "Nova", "Summit", "Pioneer", "Falcon", "Granite", "Helix",
          "Ionic", "Jovian", "Keystone", "Lattice", "Magnet", "Nexus", "Onyx", "Prism", "Quartz", "Radian", "Sigma",
          "Tessera", "Umbra", "Vector", "Willow", "Xenon", "Yonder", "Zephyr", "Alloy", "Beacon", "Cascade", "Delta",
          "Ember", "Fathom", "Gantry", "Harbor"]
TIER_MULT = {1: 2.2, 2: 1.35, 3: 1.0}
OPEN_BASE = {1: 9, 2: 14, 3: 20}
MIN_CGPA = {1: [7.5, 8.0], 2: [6.5, 7.0, 7.5], 3: [6.0, 6.5]}
WEEKS = {1: (1, 7), 2: (5, 16), 3: (12, 27)}


def _sig(x):
    return 1 / (1 + np.exp(-x))


def make_students(rng):
    parts, skill_parts = [], []
    trend = np.array([s in TREND_SKILLS for s in SKILLS], float)
    for y in YEARS:
        n = int(rng.normal(760, 25))
        b = rng.choice(BRANCHES, n, p=BR_SHARE)
        a = rng.standard_normal(n)  # latent academic/career ability
        cgpa = np.clip(np.array([BR_CGPA[x] for x in b]) + .75 * a + rng.normal(0, .4, n), 4.5, 9.95).round(2)
        lam = np.array([BR_INTERN[x] for x in b]) * np.exp(.35 * a) * (.8 + .12 * (y - 2020))
        back = rng.poisson(np.clip(.22 - .25 * a, .02, None))
        P = np.array([PROFILE[x] for x in b])
        S = rng.random(P.shape) < _sig(np.log(P / (1 - P)) + .5 * a[:, None] + .10 * (y - 2022) * trend)
        parts.append(pd.DataFrame({
            "student_id": [f"S{y}{i + 1:04d}" for i in range(n)], "batch_year": y, "branch": b, "cgpa": cgpa,
            "internships": np.minimum(rng.poisson(lam), 4),
            "certifications": np.minimum(rng.poisson(np.clip(1.2 + .35 * a, .1, None)), 6),
            "projects": np.minimum(rng.poisson(np.clip(2 + .4 * a, .3, None)), 6),
            "active_backlogs": back,
            "seeking_placement": (rng.random(n) < .92).astype(int)}))
        skill_parts.append(S)
    st = pd.concat(parts, ignore_index=True)
    st["eligible"] = ((st.cgpa >= 6.0) & (st.active_backlogs == 0)).astype(int)
    return st, np.vstack(skill_parts)


def make_companies(rng):
    rows, i = [], 0
    for ind, (suffix, _, _) in INDUSTRIES.items():
        for tier in rng.permutation([1, 2, 2, 3, 3, 3]):
            rows.append((f"C{i + 1:02d}", f"{PREFIX[i]} {suffix}", ind, int(tier)))
            i += 1
    return pd.DataFrame(rows, columns=["company_id", "company_name", "industry", "tier"])


def make_drives(rng, companies):
    drives, dskills = [], []
    for y in YEARS:
        for c in companies.itertuples():
            _, roles, mult = INDUSTRIES[c.industry]
            m = mult[y - 2020]
            if rng.random() > min(.95, .72 * m):
                continue
            w = np.array([1 + .15 * (y - 2020) if r in GROWTH_ROLES else 1. for r in roles])
            for role in rng.choice(roles, 1 + int(rng.random() < .3), replace=False, p=w / w.sum()):
                pool, _, allowed = ROLES[role]
                did = f"D{len(drives) + 1:04d}"
                lo, hi = WEEKS[c.tier]
                drives.append({"drive_id": did, "company_id": c.company_id, "batch_year": y, "role": role,
                               "min_cgpa": float(rng.choice(MIN_CGPA[c.tier])),
                               "openings": max(2, int(rng.poisson(OPEN_BASE[c.tier] * m)) + 1),
                               "season_week": int(rng.integers(lo, hi)), "allowed_branches": "|".join(allowed)})
                for s in rng.choice(pool, min(len(pool), int(rng.integers(3, 5))), replace=False):
                    dskills.append((did, s))
    return pd.DataFrame(drives), pd.DataFrame(dskills, columns=["drive_id", "skill"])


def simulate_applications(rng, students, S, drives, dskills, companies):
    idx = {s: i for i, s in enumerate(SKILLS)}
    tier = companies.set_index("company_id").tier
    req = dskills.groupby("drive_id").skill.apply(list)
    sid = students.student_id.values
    out = []
    for y in YEARS:
        ix = np.where(students.batch_year.values == y)[0]
        cg, br = students.cgpa.values[ix], students.branch.values[ix]
        it, ce, pr = (students[c].values[ix] for c in ("internships", "certifications", "projects"))
        ok0 = (students.eligible.values[ix] == 1) & (students.seeking_placement.values[ix] == 1)
        appetite = rng.lognormal(0, .3, len(ix))
        best = np.full(len(ix), 9)
        for d in drives[drives.batch_year == y].sort_values("season_week").itertuples():
            t = int(tier[d.company_id])
            cols = [idx[s] for s in req[d.drive_id]]
            ok = ok0 & (best > t) & np.isin(br, d.allowed_branches.split("|")) & (cg >= d.min_cgpa)
            c = np.where(ok)[0]
            if not len(c):
                continue
            match = S[ix[c]][:, cols].mean(1)
            keep = rng.random(len(c)) < np.clip((.10 + .65 * match) * appetite[c], 0, .95)
            c, match = c[keep], match[keep]
            if not len(c):
                continue
            z = -1 + 3.2 * match + .8 * (cg[c] - 7.5) + .3 * it[c] + .12 * ce[c] + .10 * pr[c] - .5 * (t == 1)
            sl = rng.random(len(c)) < _sig(z)
            iv = sl & (rng.random(len(c)) < _sig(.3 + 1.5 * match + .6 * (cg[c] - 7.5) - .3 * (t == 1)))
            sel = np.zeros(len(c), bool)
            n_sel = min(d.openings, int(round(.55 * iv.sum())))
            if n_sel:
                score = 1.2 * match + .45 * (cg[c] - 7.5) + .3 * it[c] + .4 * rng.standard_normal(len(c))
                cand = np.where(iv)[0]
                sel[cand[np.argsort(-score[cand])[:n_sel]]] = True
            offer = np.full(len(c), np.nan)
            k = sel.sum()
            if k:
                sal = ROLES[d.role][1] * TIER_MULT[t] * 1.04 ** (y - 2020) * rng.lognormal(0, .12, k) \
                      * (1 + .04 * np.maximum(cg[c[sel]] - 8, 0))
                sal *= np.where(rng.random(k) < .012, rng.lognormal(.8, .2, k), 1.0)  # genuine 'super-dream' outliers
                offer[sel] = sal.round(2)
                best[c[sel]] = np.minimum(best[c[sel]], t)
            out.append(pd.DataFrame({"student_id": sid[ix[c]], "drive_id": d.drive_id, "shortlisted": sl.astype(int),
                                     "interviewed": iv.astype(int), "selected": sel.astype(int), "offer_lpa": offer}))
    apps = pd.concat(out, ignore_index=True)
    apps.insert(0, "app_id", [f"A{i + 1:06d}" for i in range(len(apps))])
    return apps


def inject_dirt(rng, students, apps):
    """Add realistic data-quality problems (documented in README) for cleaning practice."""
    s, a = students.copy(), apps.copy()
    s["cgpa"] = s.cgpa.astype(float)
    s["certifications"] = s.certifications.astype(float)
    s.loc[rng.random(len(s)) < .02, "cgpa"] = np.nan
    bad = rng.choice(len(s), 8, replace=False)
    s.loc[s.index[bad], "cgpa"] = rng.choice([0.0, 78.5, 11.2, 82.0], 8)
    s.loc[rng.random(len(s)) < .03, "certifications"] = np.nan
    v = rng.random(len(s)) < .012
    s.loc[v, "branch"] = [rng.choice([b.lower(), b + " ", ".".join(b) + "."]) for b in s.loc[v, "branch"]]
    s = pd.concat([s, s.sample(12, random_state=1)], ignore_index=True)  # duplicate rows
    sel = a.index[a.selected == 1]
    a.loc[rng.choice(sel, 6, replace=False), "offer_lpa"] *= 100000  # rupees entered instead of LPA
    a.loc[sel[rng.random(len(sel)) < .015], "offer_lpa"] = np.nan  # missing offer amount
    return s, a


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    students, S = make_students(rng)
    companies = make_companies(rng)
    drives, dskills = make_drives(rng, companies)
    apps = simulate_applications(rng, students, S, drives, dskills, companies)
    r, c = np.nonzero(S)
    sskills = pd.DataFrame({"student_id": students.student_id.values[r], "skill": np.array(SKILLS)[c]})
    st_raw, ap_raw = inject_dirt(rng, students, apps)
    return {"students": st_raw, "student_skills": sskills, "companies": companies, "drives": drives,
            "drive_skills": dskills, "applications": ap_raw}


def write_raw(tables, path=RAW):
    path.mkdir(parents=True, exist_ok=True)
    for k, df in tables.items():
        df.to_csv(path / f"{k}.csv", index=False)
