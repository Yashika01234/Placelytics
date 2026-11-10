"""What-if simulator: Monte-Carlo simulation of the hiring funnel under hypothetical improvements.

Scenario analysis, NOT a forecast. Stage probabilities are fitted from the (filtered) data; per-drive selection probability is capped by
seat supply (openings / expected interviewees). Student-level salary is not modelled.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .config import SKILLS
from .filters import Frames


@dataclass
class Scenario:
    eligibility_pp: float = 0.0   # extra eligible students, as % of all students
    skill_pp: float = 0.0         # close x% of the gap between applicants' skill match and 100%
    conversion_pct: float = 0.0   # relative uplift in interview -> offer conversion
    openings_pct: float = 0.0     # relative growth in recruiter openings


def _fit(X, y):
    if len(np.unique(y)) < 2:
        const = float(y.mean()) if len(y) else 0.0
        return lambda Z: np.full(len(Z), const)
    m = LogisticRegression(max_iter=500).fit(X, y)
    return lambda Z: m.predict_proba(Z)[:, 1]


def _uncapped_rate(n_sel, n_int, openings):
    """Selected/interviewed ratio in drives that did NOT hit their seat cap (cap-free hiring appetite)."""
    m = (n_int > 0) & (n_sel < openings)
    return float(n_sel[m].sum() / n_int[m].sum()) if m.any() else 0.5


def prepare(F: Frames, so: pd.DataFrame) -> dict:
    a = F.apps.merge(so[["student_id", "cgpa", "internships", "certifications"]], on="student_id")
    sidx = a.student_id.astype("category")
    scode, didx = sidx.cat.codes.values, a.drive_id.astype("category")
    dcode = didx.cat.codes.values
    ssm = pd.crosstab(F.student_skills.student_id, F.student_skills.skill).reindex(index=sidx.cat.categories, columns=SKILLS, fill_value=0).values
    dsm = pd.crosstab(F.drive_skills.drive_id, F.drive_skills.skill).reindex(index=didx.cat.categories, columns=SKILLS, fill_value=0).values
    match = (ssm[scode] * dsm[dcode]).sum(1) / np.maximum(dsm[dcode].sum(1), 1)
    t1 = (a.tier.values == 1).astype(float)
    X1 = np.c_[match, a.cgpa, a.internships, a.certifications, t1]
    sl = a.shortlisted.values
    p_sl = _fit(X1, sl)
    X2 = np.c_[match, a.cgpa, t1]
    p_iv = _fit(X2[sl == 1], a.interviewed.values[sl == 1])
    nd = len(didx.cat.categories)
    n_int = np.bincount(dcode, weights=a.interviewed.values, minlength=nd)
    n_sel = np.bincount(dcode, weights=a.selected.values, minlength=nd)
    openings = a.groupby("drive_id").openings.first().reindex(didx.cat.categories).values
    dmeta = a.groupby("drive_id")[["season_week", "tier"]].first().reindex(didx.cat.categories)
    return dict(dweek=dmeta.season_week.values, dtier=dmeta.tier.values.astype(int), a=a, scode=scode, dcode=dcode, match=match, X1=X1, X2=X2, p_sl=p_sl, p_iv=p_iv,
                r_global=_uncapped_rate(n_sel, n_int, a.groupby("drive_id").openings.first().reindex(didx.cat.categories).values), openings=openings,
                n_students=len(so), pool=int(so.in_pool.sum()), n_stu_codes=len(sidx.cat.categories),
                actual_placed=int(so[so.in_pool].placed.sum()), actual_offers=int(a.selected.sum()))


def simulate(P: dict, sc: Scenario, n_sims: int = 60, seed: int = 7) -> dict:
    scode, dcode, match, X1, X2 = P["scode"], P["dcode"], P["match"].copy(), P["X1"].copy(), P["X2"].copy()
    cg_all = P["a"].cgpa.values
    k = int(round(sc.eligibility_pp / 100 * P["n_students"]))
    if k:  # new eligible students: bootstrap from the lower half of current applicants by CGPA (marginal students)
        rng0 = np.random.default_rng(seed)
        order = np.argsort(scode, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(scode[order])) + 1]
        ends = np.r_[starts[1:], len(order)]
        stu_cg = cg_all[order[starts]]
        marginal = np.where(stu_cg <= np.median(stu_cg))[0]
        pick = rng0.choice(marginal, k)
        rows = np.concatenate([order[starts[p]:ends[p]] for p in pick])
        new_codes = np.concatenate([np.full(ends[p] - starts[p], P["n_stu_codes"] + j) for j, p in enumerate(pick)])
        scode, dcode = np.r_[scode, new_codes], np.r_[dcode, dcode[rows]]
        match, X1, X2 = np.r_[match, match[rows]], np.vstack([X1, X1[rows]]), np.vstack([X2, X2[rows]])
        cg_all = np.r_[cg_all, cg_all[rows]]
    if sc.skill_pp:
        d = sc.skill_pp / 100 * (1 - match)
        match = match + d
        X1[:, 0] += d; X2[:, 0] += d
    psl, piv = P["p_sl"](X1), P["p_iv"](X2)
    r = np.full(len(P["openings"]), min(P["r_global"] * (1 + sc.conversion_pct / 100), 1.0))
    cap = np.round(P["openings"] * (1 + sc.openings_pct / 100)).astype(int)
    nd, m = len(cap), len(scode)
    pool = P["pool"] + k
    # Per-application win probability = P(shortlist) * P(interview) * P(selected | interviewed), where the last term is
    # limited by seat supply: min(appetite r, openings / expected interviewees in that drive).
    p_int = psl * piv
    exp_int = np.bincount(dcode, weights=p_int, minlength=nd)
    sel_p = np.minimum(r, np.divide(cap, exp_int, out=np.full(nd, 1.0), where=exp_int > 0))
    q = np.clip(p_int * sel_p[dcode], 0, 1)
    placed, offers = [], []
    for i in range(n_sims):
        won = np.random.default_rng(seed * 1000 + i).random(m) < q
        offers.append(int(won.sum())); placed.append(len(np.unique(scode[won])))
    pr = 100 * np.array(placed) / max(pool, 1)
    return dict(placement_rate=pr.mean(), rate_lo=np.percentile(pr, 5), rate_hi=np.percentile(pr, 95),
                placed=float(np.mean(placed)), offers=float(np.mean(offers)), pool=pool)


def compare(P: dict, sc: Scenario, n_sims: int = 60) -> dict:
    """Delta-anchored result: projected = observed baseline + (simulated scenario - simulated baseline).

    The simulator ignores the 'stop applying after a dream offer' behaviour present in the data, so its absolute level is
    biased low; differences between two runs with common random numbers are far more reliable than levels.
    """
    base, scen = simulate(P, Scenario(), n_sims), simulate(P, sc, n_sims)
    pool_new = scen["pool"]
    d_placed = scen["placed"] - base["placed"]
    return dict(baseline=base, scenario=scen, actual_placed=P["actual_placed"], actual_pool=P["pool"],
                actual_rate=100 * P["actual_placed"] / max(P["pool"], 1), actual_offers=P["actual_offers"],
                projected_placed=P["actual_placed"] + d_placed, projected_pool=pool_new,
                projected_rate=100 * (P["actual_placed"] + d_placed) / max(pool_new, 1),
                delta_placed=d_placed, delta_offers=scen["offers"] - base["offers"],
                delta_rate_pp=100 * (P["actual_placed"] + d_placed) / max(pool_new, 1) - 100 * P["actual_placed"] / max(P["pool"], 1))


def lever_sensitivity(P: dict, n_sims: int = 30) -> pd.DataFrame:
    base = simulate(P, Scenario(), n_sims)
    levers = {"+5pp more eligible students": Scenario(eligibility_pp=5), "Close 25% of skill-match gap": Scenario(skill_pp=25),
              "+20% interview-to-offer conversion": Scenario(conversion_pct=20), "+20% recruiter openings": Scenario(openings_pct=20)}
    return pd.DataFrame([{"lever": k, "delta_placed_students": (o := simulate(P, v, n_sims))["placed"] - base["placed"],
                          "delta_placement_pp": o["placement_rate"] - base["placement_rate"]} for k, v in levers.items()]) \
        .sort_values("delta_placed_students")
