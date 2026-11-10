"""TalentPulse - Campus Hiring Intelligence & Predictive Analytics (SYNTHETIC DATA).  Run: streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from talentpulse import analytics as A, insights as I, model as M, whatif as W
from talentpulse.config import DB_PATH, DISCLAIMER, PROC
from talentpulse.db import load_queries, run_query
from talentpulse.filters import Filters, apply_filters, full_filters, load_frames
from talentpulse.generate_data import ROLES  # noqa: F401  (documented assumptions live next to generator)

st.set_page_config(page_title="TalentPulse", page_icon="📡", layout="wide")
ACCENT = ["#6C8CFF", "#2DD4BF", "#F59E0B", "#F472B6", "#A78BFA", "#34D399", "#F87171", "#60A5FA"]
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1500px}
.kpi{background:linear-gradient(145deg,#141B2E,#0F1525);border:1px solid #232C45;border-radius:14px;padding:14px 16px;margin-bottom:10px}
.kpi .l{color:#8B97B5;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em}
.kpi .v{color:#E6EAF5;font-size:1.65rem;font-weight:700;line-height:1.25}.kpi .s{color:#6C8CFF;font-size:.72rem}
.banner{background:#2A1F0A;border:1px solid #6B4E12;color:#F5C77A;padding:8px 14px;border-radius:10px;font-size:.82rem;margin-bottom:10px}
h1{font-weight:800!important;letter-spacing:-.02em}
</style>""", unsafe_allow_html=True)


def show(fig, h=380):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=45, b=10), colorway=ACCENT, title_font_size=15)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def f(v, spec=".1f", suf=""):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:{spec}}{suf}"


def dl(df, name, key):
    st.download_button(f"⬇ Download {name}.csv", df.to_csv(index=False).encode(), f"{name}.csv", "text/csv", key=key)


if not DB_PATH.exists():
    from run_pipeline import run_pipeline
    with st.spinner("Building synthetic dataset (first run only)..."):
        run_pipeline(verbose=False)


@st.cache_data(show_spinner=False)
def get_frames(_mtime):
    return load_frames()


@st.cache_resource(show_spinner="Training models (temporal validation)...")
def get_model(_mtime):
    full = get_frames(_mtime)
    return M.evaluate(A.student_outcomes(apply_filters(full, full_filters(full))))


@st.cache_resource(show_spinner="Preparing simulator...")
def get_whatif(key, _G, _so):
    return W.prepare(_G, _so)


mt = DB_PATH.stat().st_mtime
F = get_frames(mt)
fa = full_filters(F)

# ------------------------------------------------------------------ sidebar filters
with st.sidebar:
    st.markdown("## 📡 TalentPulse")
    st.caption("Campus Hiring Intelligence · XYZ College (fictional)")
    years = st.multiselect("Academic year (batch)", fa.years, default=fa.years)
    branches = st.multiselect("Branch", fa.branches, default=fa.branches)
    inds = st.multiselect("Industry", fa.industries, default=fa.industries)
    comp_opts = sorted(F.companies[F.companies.industry.isin(inds)].company_name) if inds else []
    comps = st.multiselect("Company", comp_opts, default=comp_opts)
    sal = st.slider("Offer salary range (LPA)", fa.salary[0], fa.salary[1], fa.salary, step=0.5)
    st.caption("Salary range removes offers outside the range. Placement rate = placed ÷ eligible & placement-seeking students. "
               "With a company/industry filter it means 'holds an offer from the selected recruiters'.")
    st.warning(DISCLAIMER)

flt = Filters(years, branches, comps, inds, sal)
G = apply_filters(F, flt)
so = A.student_outcomes(G)
pool = so[so.in_pool]
st.markdown("# TalentPulse")
st.markdown(f'<div class="banner">⚠ {DISCLAIMER}</div>', unsafe_allow_html=True)
if pool.empty or G.apps.empty:
    st.error("No data for the selected filters - widen the filters in the sidebar.")
    st.stop()
K = A.kpis(G, so)

tabs = st.tabs(["Overview", "Funnel", "Recruiters", "Salary", "Departments & Risk", "Skill Gap", "Internship Impact",
                "EDA & Cohorts", "What-if Simulator", "Predictive Model", "SQL Lab & Data"])

# ------------------------------------------------------------------ Overview
with tabs[0]:
    r1 = st.columns(4); r2 = st.columns(4)
    kpi(r1[0], "Students", f"{K['students']:,}", f"{K['eligible']:,} eligible ({f(K['eligibility_rate'], '.0f')}%)")
    kpi(r1[1], "Placement rate", f(K["placement_rate"], ".1f", "%"), f"{K['placed']:,} of {K['pool']:,} in pool")
    kpi(r1[2], "Median package", f(K["median_lpa"], ".2f", " LPA"), f"mean {f(K['mean_lpa'], '.2f')} LPA")
    kpi(r1[3], "P90 package", f(K["p90_lpa"], ".1f", " LPA"), f"max {f(K['max_lpa'], '.1f')} LPA")
    kpi(r2[0], "Recruiters hiring", f"{K['recruiters']}", f"{K['drives']} drives")
    kpi(r2[1], "Offers made", f"{K['offers']:,}", f"{f(K['offers'] / max(K['placed'], 1), '.2f')} per placed student")
    kpi(r2[2], "Seat fill rate", f(K["fill_rate"], ".0f", "%"), "offers ÷ openings")
    kpi(r2[3], "Funnel yield", f(100 * K['placed'] / max(K['eligible'], 1), ".0f", "%"), "eligible → selected")
    st.markdown("#### Auto-generated insights (recomputed for the current filters)")
    for line in I.generate(G, so):
        st.markdown(f"- {line}")
    bd = A.dept_benchmark(so)
    yr = pool.groupby("batch_year").agg(rate=("placed", "mean"), med=("best_offer", "median")).reset_index()
    yr["rate"] *= 100
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Scatter(x=yr.batch_year, y=yr.rate, mode="lines+markers", name="Placement %", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=yr.batch_year, y=yr.med, mode="lines+markers", name="Median LPA", yaxis="y2"))
    fig.update_layout(title="Placement rate & median package by year", yaxis=dict(title="%"), yaxis2=dict(overlaying="y", side="right", title="LPA"),
                      xaxis=dict(dtick=1))
    with c1: show(fig)
    last = bd[bd.batch_year == bd.batch_year.max()].sort_values("placement_rate")
    with c2: show(px.bar(last, x="placement_rate", y="branch", orientation="h", text=last.placement_rate.round(1),
                         title=f"Placement rate by branch, {bd.batch_year.max()}"))
    hm = bd.pivot(index="branch", columns="batch_year", values="placement_rate")
    show(px.imshow(hm, text_auto=".0f", color_continuous_scale="Tealgrn", aspect="auto", title="Placement % - branch × year"), 330)

# ------------------------------------------------------------------ Funnel
with tabs[1]:
    fn = A.funnel(so)
    c1, c2 = st.columns(2)
    with c1:
        show(go.Figure(go.Funnel(y=fn.stage, x=fn.students, textinfo="value+percent previous")).update_layout(title="Student funnel: eligible → selected"))
    with c2:
        lab, s_, t_, v_ = A.sankey_stages(G.apps)
        show(go.Figure(go.Sankey(node=dict(label=lab, pad=18, color=ACCENT[:4] + ["#7A3B3B"] * 3),
                                 link=dict(source=s_, target=t_, value=v_))).update_layout(title="Application flow & drop-off (application level)"))
    table(fn.round(1))
    sc = A.stage_conversion(G.apps, "branch").melt("branch", ["apply_to_shortlist", "shortlist_to_interview", "interview_to_offer"], "stage", "pct")
    c3, c4 = st.columns(2)
    with c3: show(px.bar(sc, x="branch", y="pct", color="stage", barmode="group", title="Stage conversion by branch (%)"))
    sy = A.stage_conversion(G.apps, "batch_year").melt("batch_year", ["apply_to_shortlist", "shortlist_to_interview", "interview_to_offer"], "stage", "pct")
    with c4: show(px.line(sy, x="batch_year", y="pct", color="stage", markers=True, title="Stage conversion by year (%)").update_xaxes(dtick=1))
    dl(fn, "funnel", "dl_funnel")

# ------------------------------------------------------------------ Recruiters
with tabs[2]:
    rc = A.recruiters(G)
    c1, c2 = st.columns(2)
    with c1:
        tm = G.apps[G.apps.selected == 1].groupby(["industry", "company_name"]).agg(offers=("app_id", "size"), median_lpa=("offer_lpa", "median")).reset_index()
        show(px.treemap(tm, path=["industry", "company_name"], values="offers", color="median_lpa", color_continuous_scale="Viridis",
                        title="Offers by industry → company (colour = median LPA)"), 420)
    with c2:
        it = A.industry_trend(G)
        show(px.line(it, x="batch_year", y="offers", color="industry", markers=True, title="Hiring trend by industry (offers)").update_xaxes(dtick=1), 420)
    c3, c4 = st.columns(2)
    with c3:
        st_ = A.skill_trend(G)
        show(px.imshow(st_, text_auto=".0f", aspect="auto", color_continuous_scale="Purples", title="Recruiter skill demand (% of openings) - skill × year"), 460)
    with c4:
        top = rc.head(12).iloc[::-1]
        show(px.bar(top, x="offers", y="company_name", color="industry", orientation="h", title="Top recruiters by offers"), 460)
        h = A.hhi_by_year(G)
        st.caption("Recruiter concentration (HHI, 0-10000; <1500 = diversified): " + ", ".join(f"{int(r.batch_year)}: {r.hhi:.0f}" for r in h.itertuples()))
    table(rc.round(1))
    dl(rc, "recruiter_intelligence", "dl_rc")

# ------------------------------------------------------------------ Salary
with tabs[3]:
    sm = A.salary_summary(so.best_offer)
    if sm:
        cols = st.columns(5)
        kpi(cols[0], "Median / Mean", f"{sm['median']:.1f} / {sm['mean']:.1f}", "LPA, final offer per student")
        kpi(cols[1], "P10 · P90", f"{sm['p10']:.1f} · {sm['p90']:.1f}", f"P90/P10 = {sm['p90_p10']:.1f}x")
        kpi(cols[2], "Gini", f"{sm['gini']:.3f}", "0 = equal, 1 = concentrated")
        kpi(cols[3], "Top-10% share", f"{sm['top10_share']:.0f}%", "of total package value")
        kpi(cols[4], "Outliers (Tukey)", f"{sm['outliers']}", f"fence > {sm['upper_fence']:.1f} LPA")
        placed = so[so.placed == 1]
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(placed, x="best_offer", nbins=40, title="Distribution of final offers (LPA)")
            for q, lab in [("median", "median"), ("p90", "P90")]:
                fig.add_vline(x=sm[q], line_dash="dash", annotation_text=lab)
            show(fig)
        with c2: show(px.box(placed, x="branch", y="best_offer", color="branch", points="outliers", title="Package by branch"))
        c3, c4 = st.columns(2)
        with c3:
            po = G.apps[G.apps.selected == 1]
            show(px.box(po, x="industry", y="offer_lpa", color="industry", title="Offers by industry").update_layout(showlegend=False).update_xaxes(tickangle=30))
        with c4: show(px.scatter(placed, x="cgpa", y="best_offer", color="branch", opacity=.6, title="CGPA vs final package"))
        pct = placed.groupby("batch_year").best_offer.describe(percentiles=[.1, .25, .5, .75, .9]).reset_index().round(2)
        table(pct)
        dl(pct, "salary_percentiles_by_year", "dl_sal")

# ------------------------------------------------------------------ Departments & Risk
with tabs[4]:
    bd = A.dept_benchmark(so)
    c1, c2 = st.columns(2)
    with c1: show(px.line(bd, x="batch_year", y="placement_rate", color="branch", markers=True, title="Placement rate by branch (%)").update_xaxes(dtick=1))
    with c2: show(px.imshow(bd.pivot(index="branch", columns="batch_year", values="yoy_pp"), text_auto=".1f", color_continuous_scale="RdYlGn",
                           color_continuous_midpoint=0, aspect="auto", title="Year-over-year change (pp)"))
    st.markdown("#### Placement risk register")
    risk = A.placement_risk(G, so)
    table(risk.round(1))
    st.caption("Flags: 3+ year slope ≤ -3 pp/yr · latest ≥10 pp below institute · YoY drop ≥ 8 pp · one recruiter ≥ 30% of the branch's offers.")
    sr = A.stage_risk(G)
    if len(sr):
        piv = sr.pivot(index="branch", columns="stage", values="delta_pp")
        show(px.imshow(piv, text_auto=".1f", color_continuous_scale="RdYlGn", color_continuous_midpoint=0, aspect="auto",
                       title="Latest-year stage conversion vs prior-year average (pp) - red = deterioration"), 320)
    table(bd.round(2))
    dl(bd, "department_benchmark", "dl_bd"); dl(risk, "placement_risk", "dl_risk")

# ------------------------------------------------------------------ Skill gap
with tabs[5]:
    sg = A.skill_gap(G, so)
    m = sg.melt("skill", ["demand_pct", "supply_pct"], "side", "pct")
    c1, c2 = st.columns(2)
    with c1: show(px.bar(m, x="skill", y="pct", color="side", barmode="group", title="Recruiter demand vs student supply (% of openings / eligible students)").update_xaxes(tickangle=35), 420)
    with c2:
        g2 = sg.sort_values("gap_pp")
        show(px.bar(g2, x="gap_pp", y="skill", orientation="h", color="gap_pp", color_continuous_scale="RdYlGn_r", title="Skill gap (demand − supply, pp)"), 420)
    show(px.scatter(sg, x="gap_pp", y="placement_lift_pp", text="skill", size=sg.demand_pct.clip(lower=1), title="Gap vs placement lift: bigger = more demanded"), 400)
    table(sg.round(1))
    dl(sg, "skill_gap", "dl_sg")

# ------------------------------------------------------------------ Internship impact
with tabs[6]:
    it, strat, mh = A.internship_impact(so)
    cols = st.columns(3)
    kpi(cols[0], "Crude odds ratio", f(mh["crude_or"], ".2f", "x"), "any internship vs none")
    kpi(cols[1], "CGPA-adjusted OR (Mantel-Haenszel)", f(mh["mh_or"], ".2f", "x"), f"95% CI {f(mh['ci_lo'], '.2f')}-{f(mh['ci_hi'], '.2f')}")
    kpi(cols[2], "Reading", "Association", "observational data; not a causal effect")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=it.bucket, y=it.placement_rate, error_y=dict(type="data", symmetric=False, array=it.ci_hi - it.placement_rate, arrayminus=it.placement_rate - it.ci_lo)))
        show(fig.update_layout(title="Placement rate by internships (95% Wilson CI)", xaxis_title="internships"))
    with c2:
        show(px.bar(strat.assign(group=strat.any_intern.map({0: "No internship", 1: "≥1 internship"})), x="cgpa_band", y="placement_rate", color="group",
                    barmode="group", title="Within CGPA bands (controls for ability confounding)"))
    table(it.round(1))
    dl(it, "internship_impact", "dl_it")

# ------------------------------------------------------------------ EDA & cohorts
with tabs[7]:
    c1, c2 = st.columns(2)
    with c1: show(px.imshow(A.correlation(so).round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Spearman correlation (placement pool)"), 440)
    with c2: show(px.line(A.cohort_curve(G, so), x="season_week", y="cum_placed_pct", color="batch_year", title="Cohort curve: cumulative % of pool placed by season week"), 440)
    cc = A.cgpa_cohort(so)
    c3, c4 = st.columns(2)
    with c3: show(px.imshow(cc.pivot(index="cgpa_band", columns="batch_year", values="placement_rate"), text_auto=".0f", color_continuous_scale="Tealgrn", aspect="auto", title="Placement % - CGPA band × batch"))
    with c4: show(px.box(pool, x=pool.placed.map({0: "Not placed", 1: "Placed"}), y="cgpa", points=False, title="CGPA by outcome"))
    st.markdown("#### Outlier detection (IQR & modified z-score)")
    table(A.outlier_report(so).round(2))
    yo = pool.groupby("batch_year").agg(pool=("student_id", "size"), placed=("placed", "sum"), median_lpa=("best_offer", "median")).reset_index()
    yo["placement_rate"] = 100 * yo.placed / yo.pool
    yo["yoy_rate_pp"] = yo.placement_rate.diff(); yo["yoy_median_pct"] = 100 * yo.median_lpa.pct_change()
    st.markdown("#### Year-over-year comparison"); table(yo.round(2))
    dl(so[so.in_pool].drop(columns=["in_pool"]), "student_level_outcomes_SYNTHETIC", "dl_so")

# ------------------------------------------------------------------ What-if
with tabs[8]:
    st.markdown("Scenario analysis on the **currently filtered** data. Levers are applied to a simulated funnel; results are shown as the observed baseline "
                "plus the simulated change (not a forecast).")
    c = st.columns(4)
    e = c[0].slider("More eligible students (pp of cohort)", 0.0, 15.0, 0.0, .5)
    s_ = c[1].slider("Close skill-match gap (%)", 0, 100, 0, 5)
    v = c[2].slider("Interview → offer conversion (+%)", 0, 50, 0, 5)
    o = c[3].slider("Recruiter openings (+%)", 0, 50, 0, 5)
    P = get_whatif((tuple(years), tuple(branches), tuple(comps), tuple(inds), tuple(sal)), G, so)
    res = W.compare(P, W.Scenario(e, s_, v, o), n_sims=60)
    k = st.columns(4)
    kpi(k[0], "Placement rate", f(res["projected_rate"], ".1f", "%"), f"{res['delta_rate_pp']:+.1f} pp vs observed {res['actual_rate']:.1f}%")
    kpi(k[1], "Students placed", f"{res['projected_placed']:,.0f}", f"{res['delta_placed']:+,.0f} vs {res['actual_placed']:,}")
    kpi(k[2], "Offers", f"{res['actual_offers'] + res['delta_offers']:,.0f}", f"{res['delta_offers']:+,.0f}")
    kpi(k[3], "Eligible pool", f"{res['projected_pool']:,}", f"{res['projected_pool'] - res['actual_pool']:+,}")
    sens = W.lever_sensitivity(P)
    show(px.bar(sens, x="delta_placed_students", y="lever", orientation="h", title="Which lever moves placements most? (extra students placed, vs baseline)"), 320)
    st.caption("Reading the result: eligibility growth adds students but they are marginal candidates competing for the same seats; "
               "skills/conversion levers help existing applicants; extra openings only help when seats - not candidates - are the bottleneck.")

# ------------------------------------------------------------------ Predictive model
with tabs[9]:
    mdl = get_model(mt)
    if not mdl:
        st.info("Not enough data for validation.")
    else:
        st.markdown(f"Predicts whether a placement-seeking eligible student gets an offer. **Temporal validation:** train on {mdl['train_years']} "
                    f"(n={mdl['n_train']:,}), test on {mdl['test_year']} (n={mdl['n_test']:,}). Uses the full dataset, independent of sidebar filters. "
                    "Features exclude any application/offer information (no leakage).")
        table(mdl["results"].round(3))
        st.caption(f"Baseline Brier score (predict base rate): {mdl['baseline_brier']:.3f}. Because the data is synthetic, high accuracy reflects the relationships "
                   "built into the generator, not real-world predictability.")
        show(px.bar(mdl["importance"], x="auc_drop", y="feature", orientation="h", error_x="std", title=f"Permutation importance on test year ({mdl['best']})"), 320)
        st.markdown("#### Estimate a profile")
        c = st.columns(5)
        pr = pd.DataFrame([dict(cgpa=c[0].slider("CGPA", 6.0, 10.0, 7.5, .1), internships=c[1].slider("Internships", 0, 4, 1),
                                certifications=c[2].slider("Certifications", 0, 6, 1), projects=c[3].slider("Projects", 0, 6, 2),
                                n_skills=c[4].slider("Skills held", 0, 15, 4), branch=st.selectbox("Branch", fa.branches))])
        st.metric("Estimated placement probability (logistic model, synthetic data)", f"{100 * mdl['predictor'].predict_proba(pr)[0, 1]:.0f}%")

# ------------------------------------------------------------------ SQL lab & data
with tabs[10]:
    qs = load_queries()
    name = st.selectbox("Named query (runs on the full database, ignores sidebar filters)", list(qs))
    st.code(qs[name], language="sql")
    out = run_query(name)
    table(out)
    dl(out, name, "dl_sql")
    st.markdown("#### Data quality report (cleaning log)")
    dq = pd.read_csv(PROC / "data_quality_report.csv") if (PROC / "data_quality_report.csv").exists() else pd.DataFrame()
    table(dq)
    st.markdown("#### Downloads (all SYNTHETIC)")
    d = st.columns(3)
    with d[0]: dl(G.students, "students_filtered_SYNTHETIC", "dl_stu")
    with d[1]: dl(G.apps, "applications_filtered_SYNTHETIC", "dl_app")
    with d[2]: dl(G.drives, "drives_filtered_SYNTHETIC", "dl_drv")
