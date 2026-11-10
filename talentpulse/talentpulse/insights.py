"""Auto-generated, data-driven insight sentences (computed from whatever filters are active)."""
import numpy as np
from . import analytics as A


def generate(F, so) -> list[str]:
    out = []
    k = A.kpis(F, so)
    bd = A.dept_benchmark(so)
    if bd.empty:
        return ["No data for the selected filters."]
    yr = bd.batch_year.max()
    yearly = so[so.in_pool].groupby("batch_year").placed.mean() * 100
    if len(yearly) > 1:
        out.append(f"Placement rate was **{yearly.iloc[-1]:.1f}%** in {yearly.index[-1]} "
                   f"({yearly.iloc[-1] - yearly.iloc[-2]:+.1f} pp vs {yearly.index[-2]}); range across years {yearly.min():.1f}-{yearly.max():.1f}%.")
    last = bd[bd.batch_year == yr].sort_values("placement_rate")
    if len(last) > 1:
        out.append(f"In {yr}, **{last.iloc[-1].branch}** led at {last.iloc[-1].placement_rate:.1f}% while **{last.iloc[0].branch}** trailed at "
                   f"{last.iloc[0].placement_rate:.1f}% - a {last.iloc[-1].placement_rate - last.iloc[0].placement_rate:.1f} pp spread.")
    s = A.salary_summary(so.best_offer)
    if s:
        out.append(f"Median package is **{s['median']:.1f} LPA** (mean {s['mean']:.1f}); the top 10% of placed students hold **{s['top10_share']:.0f}%** of total "
                   f"package value (Gini {s['gini']:.2f}) - the mean overstates the typical outcome by {100 * (s['mean'] / s['median'] - 1):.0f}%.")
    sg = A.skill_gap(F, so)
    if len(sg) and sg.demand_pct.sum() > 0:
        t = sg.iloc[0]
        out.append(f"Largest skill gap: **{t.skill}** - demanded in {t.demand_pct:.0f}% of openings but held by {t.supply_pct:.0f}% of eligible students "
                   f"(placement lift {t.placement_lift_pp:+.0f} pp for those who have it).")
    _, _, mh = A.internship_impact(so)
    if not np.isnan(mh["mh_or"]):
        out.append(f"Internship experience is associated with **{mh['mh_or']:.1f}x** higher odds of placement after adjusting for CGPA band "
                   f"(95% CI {mh['ci_lo']:.1f}-{mh['ci_hi']:.1f}; crude OR {mh['crude_or']:.1f}). Association, not proof of causation.")
    risk = A.placement_risk(F, so)
    if len(risk):
        hi = risk[risk.risk_score > 0]
        out.append("Risk watch: " + "; ".join(f"**{r.branch}** ({r.flags})" for r in hi.itertuples()) + "." if len(hi) else
                   "Risk watch: no branch triggers a risk flag under the configured thresholds.")
    sr = A.stage_risk(F)
    if len(sr) and sr.flag.any():
        w = sr[sr.flag].sort_values("delta_pp").iloc[0]
        out.append(f"Stage alert: **{w.branch}** {w.stage.replace('_', ' ')} conversion fell {abs(w.delta_pp):.1f} pp vs its prior-year average.")
    return out
