"""All analytical computations. Pure functions over (filtered) Frames; nothing is hardcoded."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from .config import SKILLS
from .filters import Frames

NAN = np.nan


def _div(a, b):
    return np.where(np.asarray(b) > 0, 100 * np.asarray(a) / np.where(np.asarray(b) > 0, b, 1), NAN)


def student_outcomes(F: Frames) -> pd.DataFrame:
    g = F.apps.groupby("student_id").agg(n_apps=("app_id", "size"), shortlisted=("shortlisted", "max"),
                                         interviewed=("interviewed", "max"), placed=("selected", "max"),
                                         best_offer=("offer_lpa", "max"))
    ns = F.student_skills.groupby("student_id").size().rename("n_skills")
    s = F.students.merge(g, left_on="student_id", right_index=True, how="left").merge(
        ns, left_on="student_id", right_index=True, how="left")
    for c in ["n_apps", "shortlisted", "interviewed", "placed", "n_skills"]:
        s[c] = s[c].fillna(0).astype(int)
    s["in_pool"] = (s.eligible == 1) & (s.seeking_placement == 1)
    s["applied"] = (s.n_apps > 0).astype(int)
    return s


def gini(x) -> float:
    x = np.sort(np.asarray(x, float))
    if len(x) == 0 or x.sum() == 0:
        return NAN
    n = len(x)
    return float(2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def kpis(F: Frames, so: pd.DataFrame) -> dict:
    pool, offers = so[so.in_pool], F.apps[F.apps.selected == 1]
    placed = pool[pool.placed == 1]
    openings = F.drives.openings.sum()
    return dict(students=len(so), eligible=int((so.eligible == 1).sum()),
                eligibility_rate=100 * (so.eligible == 1).mean() if len(so) else NAN,
                pool=len(pool), placed=len(placed), placement_rate=100 * len(placed) / len(pool) if len(pool) else NAN,
                median_lpa=placed.best_offer.median(), mean_lpa=placed.best_offer.mean(),
                p90_lpa=placed.best_offer.quantile(.9) if len(placed) else NAN, max_lpa=placed.best_offer.max(),
                recruiters=offers.company_id.nunique(), drives=len(F.drives), offers=len(offers),
                fill_rate=100 * len(offers) / openings if openings else NAN)


def funnel(so: pd.DataFrame) -> pd.DataFrame:
    el = so[so.eligible == 1]
    counts = [len(el), int(el.applied.sum()), int(el.shortlisted.sum()), int(el.interviewed.sum()), int(el.placed.sum())]
    df = pd.DataFrame({"stage": ["Eligible", "Applied", "Shortlisted", "Interviewed", "Selected"], "students": counts})
    df["pct_of_previous"] = _div(df.students, df.students.shift(1).fillna(df.students))
    df["pct_of_eligible"] = _div(df.students, counts[0])
    return df


def stage_conversion(apps: pd.DataFrame, by) -> pd.DataFrame:
    g = apps.groupby(by).agg(applications=("app_id", "size"), shortlisted=("shortlisted", "sum"),
                             interviewed=("interviewed", "sum"), selected=("selected", "sum")).reset_index()
    g["apply_to_shortlist"] = _div(g.shortlisted, g.applications)
    g["shortlist_to_interview"] = _div(g.interviewed, g.shortlisted)
    g["interview_to_offer"] = _div(g.selected, g.interviewed)
    return g


def sankey_stages(apps: pd.DataFrame):
    n, sl, iv, se = len(apps), int(apps.shortlisted.sum()), int(apps.interviewed.sum()), int(apps.selected.sum())
    labels = ["Applications", "Shortlisted", "Interviewed", "Selected", "Screened out", "Dropped pre-interview", "Rejected after interview"]
    return labels, [0, 1, 2, 0, 1, 2], [1, 2, 3, 4, 5, 6], [sl, iv, se, n - sl, sl - iv, iv - se]


def sankey_branch_industry(apps: pd.DataFrame) -> pd.DataFrame:
    return apps[apps.selected == 1].groupby(["branch", "industry"]).size().rename("offers").reset_index()


def salary_summary(best) -> dict:
    x = pd.Series(best).dropna()
    if x.empty:
        return {}
    q = x.quantile([.1, .25, .5, .75, .9, .99])
    srt = np.sort(x.values)[::-1]
    k = max(1, int(np.ceil(.1 * len(x))))
    iqr = q[.75] - q[.25]
    return dict(n=len(x), mean=x.mean(), std=x.std(), p10=q[.1], p25=q[.25], median=q[.5], p75=q[.75], p90=q[.9], p99=q[.99],
                gini=gini(x), p90_p10=q[.9] / q[.1], top10_share=100 * srt[:k].sum() / srt.sum(),
                skew=float(stats.skew(x)), outliers=int((x > q[.75] + 1.5 * iqr).sum() + (x < q[.25] - 1.5 * iqr).sum()),
                upper_fence=q[.75] + 1.5 * iqr)


def dept_benchmark(so: pd.DataFrame) -> pd.DataFrame:
    g = so[so.in_pool].groupby(["branch", "batch_year"]).agg(
        pool=("student_id", "size"), placed=("placed", "sum"), median_lpa=("best_offer", "median"),
        mean_cgpa=("cgpa", "mean"), mean_internships=("internships", "mean")).reset_index()
    g["placement_rate"] = _div(g.placed, g.pool)
    g = g.sort_values(["branch", "batch_year"])
    g["yoy_pp"] = g.groupby("branch").placement_rate.diff()
    return g


def placement_risk(F: Frames, so: pd.DataFrame) -> pd.DataFrame:
    bd = dept_benchmark(so)
    if bd.empty:
        return pd.DataFrame()
    yr = bd.batch_year.max()
    allr = so[so.in_pool & (so.batch_year == yr)]
    overall = 100 * allr.placed.mean() if len(allr) else NAN
    offers = F.apps[(F.apps.selected == 1) & (F.apps.batch_year == yr)]
    rows = []
    for b, g in bd.groupby("branch"):
        g = g.sort_values("batch_year")
        slope = np.polyfit(g.batch_year, g.placement_rate, 1)[0] if len(g) >= 3 else NAN
        latest = g.placement_rate.iloc[-1] if g.batch_year.iloc[-1] == yr else NAN
        yoy = g.yoy_pp.iloc[-1] if g.batch_year.iloc[-1] == yr else NAN
        ob = offers[offers.branch == b]
        conc = 100 * ob.company_name.value_counts(normalize=True).iloc[0] if len(ob) else NAN
        flags = []
        if slope <= -3: flags.append("Declining 3-yr+ trend")
        if latest - overall <= -10: flags.append("10pp+ below institute")
        if yoy <= -8: flags.append("Sharp YoY drop")
        if conc >= 30: flags.append("Recruiter concentration")
        rows.append(dict(branch=b, latest_year=yr, latest_rate=latest, trend_pp_per_year=slope, gap_vs_institute_pp=latest - overall,
                         yoy_pp=yoy, top_recruiter_share=conc, flags=", ".join(flags) or "-", risk_score=len(flags),
                         risk_level="High" if len(flags) >= 2 else "Moderate" if len(flags) == 1 else "Low"))
    return pd.DataFrame(rows).sort_values(["risk_score", "latest_rate"], ascending=[False, True])


def stage_risk(F: Frames) -> pd.DataFrame:
    c = stage_conversion(F.apps, ["branch", "batch_year"])
    if c.empty or c.batch_year.nunique() < 2:
        return pd.DataFrame()
    yr = c.batch_year.max()
    rows = []
    for stage in ["apply_to_shortlist", "shortlist_to_interview", "interview_to_offer"]:
        for b, g in c.groupby("branch"):
            last, prev = g[g.batch_year == yr][stage], g[g.batch_year < yr][stage]
            if last.empty or prev.dropna().empty:
                continue
            d = last.iloc[0] - prev.mean()
            rows.append(dict(branch=b, stage=stage, latest=last.iloc[0], prior_avg=prev.mean(), delta_pp=d, flag=bool(d <= -5)))
    return pd.DataFrame(rows)


def _wilson(k, n, z=1.96):
    if n == 0:
        return NAN, NAN
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def internship_impact(so: pd.DataFrame):
    p = so[so.in_pool].copy()
    p["bucket"] = p.internships.clip(upper=3).map({0: "0", 1: "1", 2: "2", 3: "3+"})
    p["cgpa_band"] = pd.cut(p.cgpa, [0, 7, 8, 10.01], right=False, labels=["<7.0", "7.0-7.99", "8.0+"])
    t = p.groupby("bucket").agg(pool=("student_id", "size"), placed=("placed", "sum"), median_lpa=("best_offer", "median")).reset_index()
    t["placement_rate"] = _div(t.placed, t.pool)
    ci = [_wilson(k, n) for k, n in zip(t.placed, t.pool)]
    t["ci_lo"], t["ci_hi"] = [c[0] for c in ci], [c[1] for c in ci]
    p["any_intern"] = (p.internships > 0).astype(int)
    strat = p.groupby(["cgpa_band", "any_intern"], observed=True).agg(pool=("student_id", "size"), placed=("placed", "sum")).reset_index()
    strat["placement_rate"] = _div(strat.placed, strat.pool)
    return t, strat, mantel_haenszel(p)


def mantel_haenszel(p: pd.DataFrame) -> dict:
    """Crude vs CGPA-band-adjusted odds ratio of placement for 'any internship' (RGB confidence interval)."""
    def tab(g):
        a = ((g.any_intern == 1) & (g.placed == 1)).sum(); b = ((g.any_intern == 1) & (g.placed == 0)).sum()
        c = ((g.any_intern == 0) & (g.placed == 1)).sum(); d = ((g.any_intern == 0) & (g.placed == 0)).sum()
        return a, b, c, d
    a, b, c, d = tab(p)
    crude = (a * d) / (b * c) if b * c else NAN
    R = S = PR = PSQR = QS = 0.0
    for _, g in p.groupby("cgpa_band", observed=True):
        a, b, c, d = map(float, tab(g)); n = a + b + c + d
        if n == 0: continue
        r, s_, P, Q = a * d / n, b * c / n, (a + d) / n, (b + c) / n
        R += r; S += s_; PR += P * r; PSQR += P * s_ + Q * r; QS += Q * s_
    if S == 0 or R == 0:
        return dict(crude_or=crude, mh_or=NAN, ci_lo=NAN, ci_hi=NAN)
    mh = R / S
    se = np.sqrt(PR / (2 * R * R) + PSQR / (2 * R * S) + QS / (2 * S * S))
    return dict(crude_or=crude, mh_or=mh, ci_lo=np.exp(np.log(mh) - 1.96 * se), ci_hi=np.exp(np.log(mh) + 1.96 * se))


def skill_gap(F: Frames, so: pd.DataFrame) -> pd.DataFrame:
    dsk = F.drive_skills.merge(F.drives[["drive_id", "openings"]], on="drive_id")
    tot = F.drives.openings.sum()
    demand = (dsk.groupby("skill").openings.sum() / tot).reindex(SKILLS).fillna(0) if tot else pd.Series(0.0, index=SKILLS)
    el = so[so.eligible == 1]
    sk = F.student_skills[F.student_skills.student_id.isin(el.student_id)]
    supply = (sk.groupby("skill").size() / max(len(el), 1)).reindex(SKILLS).fillna(0)
    pool = so[so.in_pool].set_index("student_id")
    mat = pd.crosstab(F.student_skills.student_id, F.student_skills.skill).reindex(index=pool.index, columns=SKILLS, fill_value=0)
    lift = []
    for s in SKILLS:
        has = mat[s] == 1
        w, wo = pool.placed[has], pool.placed[~has]
        lift.append((100 * w.mean() if len(w) else NAN, 100 * wo.mean() if len(wo) else NAN))
    df = pd.DataFrame({"skill": SKILLS, "demand_pct": 100 * demand.values, "supply_pct": 100 * supply.values,
                       "placement_with": [l[0] for l in lift], "placement_without": [l[1] for l in lift]})
    df["gap_pp"] = df.demand_pct - df.supply_pct
    df["placement_lift_pp"] = df.placement_with - df.placement_without
    return df.sort_values("gap_pp", ascending=False).reset_index(drop=True)


def skill_trend(F: Frames) -> pd.DataFrame:
    dsk = F.drive_skills.merge(F.drives[["drive_id", "openings", "batch_year"]], on="drive_id")
    tot = F.drives.groupby("batch_year").openings.sum()
    t = dsk.groupby(["skill", "batch_year"]).openings.sum().unstack(fill_value=0)
    return (100 * t.div(tot, axis=1)).reindex(SKILLS).fillna(0)


def recruiters(F: Frames) -> pd.DataFrame:
    d = F.drives.groupby(["company_name", "industry", "tier"]).agg(drives=("drive_id", "size"), openings=("openings", "sum"),
                                                                   avg_min_cgpa=("min_cgpa", "mean")).reset_index()
    a = F.apps.groupby("company_name").agg(applications=("app_id", "size"), offers=("selected", "sum"),
                                           median_offer_lpa=("offer_lpa", "median")).reset_index()
    r = d.merge(a, on="company_name", how="left").fillna({"applications": 0, "offers": 0})
    r["offer_rate_per_application"] = _div(r.offers, r.applications)
    r["fill_rate"] = _div(r.offers, r.openings)
    sk = F.drive_skills.merge(F.drives[["drive_id", "company_name", "openings"]], on="drive_id")
    top = sk.groupby(["company_name", "skill"]).openings.sum().reset_index().sort_values("openings", ascending=False)
    r["top_skills"] = r.company_name.map(top.groupby("company_name").skill.apply(lambda s: ", ".join(s.head(3))))
    return r.sort_values("offers", ascending=False).reset_index(drop=True)


def industry_trend(F: Frames) -> pd.DataFrame:
    return F.apps[F.apps.selected == 1].groupby(["batch_year", "industry"]).agg(
        offers=("app_id", "size"), median_lpa=("offer_lpa", "median")).reset_index()


def hhi_by_year(F: Frames) -> pd.DataFrame:
    o = F.apps[F.apps.selected == 1].groupby(["batch_year", "company_name"]).size().rename("n").reset_index()
    o["share"] = o.n / o.groupby("batch_year").n.transform("sum")
    return o.groupby("batch_year").share.apply(lambda s: 10000 * (s ** 2).sum()).rename("hhi").reset_index()


def cohort_curve(F: Frames, so: pd.DataFrame) -> pd.DataFrame:
    first = F.apps[F.apps.selected == 1].groupby("student_id").season_week.min().rename("week")
    p = so[so.in_pool][["student_id", "batch_year"]].merge(first, left_on="student_id", right_index=True, how="left")
    rows = []
    for y, g in p.groupby("batch_year"):
        for w in range(1, 28):
            rows.append((y, w, 100 * (g.week <= w).sum() / len(g)))
    return pd.DataFrame(rows, columns=["batch_year", "season_week", "cum_placed_pct"])


def cgpa_cohort(so: pd.DataFrame) -> pd.DataFrame:
    p = so[so.in_pool].copy()
    p["cgpa_band"] = pd.cut(p.cgpa, [0, 6.5, 7, 7.5, 8, 8.5, 10.01], right=False, labels=["6.0-6.49", "6.5-6.99", "7.0-7.49", "7.5-7.99", "8.0-8.49", "8.5+"])
    g = p.groupby(["cgpa_band", "batch_year"], observed=True).agg(pool=("student_id", "size"), placed=("placed", "sum")).reset_index()
    g["placement_rate"] = _div(g.placed, g.pool)
    return g


def correlation(so: pd.DataFrame) -> pd.DataFrame:
    p = so[so.in_pool][["cgpa", "internships", "certifications", "projects", "n_skills", "n_apps", "placed", "best_offer"]]
    return p.corr(method="spearman")


def outlier_report(so: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, x in [("CGPA", so.cgpa), ("Final offer (LPA)", so.best_offer.dropna())]:
        x = x.dropna()
        if len(x) < 4:
            continue
        q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
        mad = np.median(np.abs(x - x.median())) or NAN
        mz = 0.6745 * (x - x.median()) / mad
        rows.append(dict(variable=name, n=len(x), iqr_outliers=int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()),
                         modified_z_gt_3_5=int((mz.abs() > 3.5).sum()), upper_fence=q3 + 1.5 * iqr, lower_fence=q1 - 1.5 * iqr))
    return pd.DataFrame(rows)
