# TalentPulse - Campus Hiring Intelligence & Predictive Analytics Platform

> ⚠️ **ALL DATA IS SYNTHETIC.** Every student, recruiter, drive and offer is produced by a seeded simulator for a **fictional "XYZ College"**.
> Nothing here is an actual placement statistic, and the findings below describe the *simulator's* built-in behaviour (seed 42), not any real institution.

An analytics platform that investigates placement outcomes, recruiter behaviour, salary patterns, employability and skill-market mismatch, with an
8-module analytical layer, a SQL layer (CTEs, joins, window functions), a validated ML model and a dark SaaS-style Streamlit dashboard.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py            # generate -> clean -> validate -> SQLite (data/talentpulse.db)
python -m pytest -q               # 16 tests (generation, cleaning, validation, SQL, analytics, filters, simulator, app)
streamlit run app.py              # http://localhost:8501  (the pipeline also auto-runs on first launch)
```
`python run_pipeline.py --seed 7` regenerates a different but equally reproducible world.

## Architecture
```
generate_data.py --raw CSV--> clean.py --> validate.py --> db.py (SQLite) --> filters.py (SQL joins + one filter engine)
                                                                                  |
                       analytics.py | whatif.py | model.py | insights.py  <--------+--> app.py (Streamlit + Plotly)
```
| Path | Role |
|---|---|
| `talentpulse/generate_data.py` | Seeded simulator: students, skills, companies, drives, applications; injects data-quality defects |
| `talentpulse/clean.py` / `validate.py` | Documented cleaning log; 19 integrity/range/business-rule checks (pipeline aborts if any fail) |
| `talentpulse/db.py`, `sql/queries.sql` | SQLite build + 7 named analytical queries (also browsable in the dashboard's *SQL Lab*) |
| `talentpulse/filters.py` | Loads data via SQL joins; **one** `apply_filters` used by every chart/metric/download |
| `talentpulse/analytics.py` | Funnel, salary, benchmarking, risk, internship impact, skill gap, recruiters, cohorts, correlation, outliers |
| `talentpulse/whatif.py` | Seat-limited Monte-Carlo scenario simulator |
| `talentpulse/model.py` | Logistic regression + gradient boosting with temporal validation |
| `tests/` | pytest suite incl. headless Streamlit `AppTest` |

## Synthetic-data assumptions (all configurable in `generate_data.py`)
1. **Cohorts:** 5 batches (2020-2024), ~760 students each (≈3,760 total); 6 branches (CSE .22, IT .15, ECE .20, EEE .15, ME .17, CE .11).
2. **Latent ability** `a ~ N(0,1)` drives CGPA (`branch mean + 0.75a + N(0,0.4)`, clipped 4.5-9.95), skills (logit shift 0.5a), internships (Poisson, λ rises with a and year),
   certifications/projects (Poisson) and backlogs (lower for high a). This is how CGPA, skills and internships become *correlated* - and why internship effects must be adjusted for confounding.
3. **Skills:** 15 skills with branch-specific base probabilities (e.g. CSE: Python .70, CAD .02; ME: CAD .65). Python/SQL/ML/Cloud/Data-Analytics supply drifts up 0.10 logit per year.
4. **Eligibility:** CGPA ≥ 6.0 and zero active backlogs; 8% of eligible students do not seek campus placement (higher studies etc.).
5. **Recruiters:** 48 fictional companies (6 per industry × 8 industries; tiers 1/2/3). Visit probability = 0.72 × industry-year multiplier (e.g. IT Services dips in 2023, FinTech grows 2020→2024, Core Engineering declines).
   Growth roles (Data Scientist, Cloud/DevOps, Data Analyst) get more weight over time. 13 roles, each with a skill pool, base salary and allowed branches.
6. **Drives:** openings ~ Poisson(tier base × multiplier)+1, minimum CGPA by tier, season week by tier (tier-1 early), 3-4 required skills.
7. **Application process (per drive, in season order):** apply prob = (0.10 + 0.65 × skill-match) × student appetite; shortlist = logistic(match, CGPA, internships, certs, projects, tier);
   interview = logistic(match, CGPA, tier); selection = top `min(openings, 55% of interviewees)` by score. **Dream-offer policy:** after an offer, a student may only apply to a *better* tier.
8. **Salary (LPA):** role base × tier multiplier (2.2/1.35/1.0) × 4%/yr inflation × lognormal(0, 0.12) × small CGPA bonus; 1.2% "super-dream" offers get an extra ×lognormal(0.8, 0.2) (genuine outliers).
9. **Injected defects (for cleaning practice):** 2% missing CGPA, 8 impossible CGPAs (0, 11.2, 78.5, 82), 3% missing certifications, ~1.2% inconsistent branch labels, 12 duplicate students,
   6 offers keyed in rupees instead of LPA, 1.5% selected offers with missing amount. Treatments are logged in `data/processed/data_quality_report.csv` and in the dashboard.

## Data dictionary
**students** `student_id`, `batch_year`, `branch`, `cgpa` (0-10), `internships` (0-4), `certifications` (0-6), `projects` (0-6), `active_backlogs`, `seeking_placement` (0/1), `eligible` (0/1), `cgpa_imputed` (0/1, added by cleaning)
**student_skills** `student_id`, `skill` · **companies** `company_id`, `company_name`, `industry`, `tier` (1 = premium)
**drives** `drive_id`, `company_id`, `batch_year`, `role`, `min_cgpa`, `openings`, `season_week` (1-27), `allowed_branches` (pipe-separated) · **drive_skills** `drive_id`, `skill`
**applications** `app_id`, `student_id`, `drive_id`, `shortlisted`, `interviewed`, `selected` (0/1 funnel flags), `offer_lpa` (only if selected), `offer_imputed`

## Methodology & metric definitions
* **Pool** = eligible AND seeking placement. **Placement rate** = pool students with ≥1 offer ÷ pool. **Final package** = a student's highest offer.
* **Funnel (student level):** eligible → applied → shortlisted → interviewed → selected; plus an application-level Sankey with drop-off nodes.
* **Salary intelligence:** percentiles, P90/P10, Gini, top-10% value share, Tukey (1.5×IQR) outliers, MAD-based modified z-score; genuine outliers are *kept* and flagged.
* **Skill gap** = opening-weighted share of drives requiring the skill − share of eligible students holding it; also placement lift (with vs without).
* **Internship impact:** placement by internship count (Wilson CIs), stratified by CGPA band, crude vs **Mantel-Haenszel CGPA-adjusted odds ratio** (Robins-Breslow-Greenland CI). Observational association only.
* **Risk register:** flags for ≥3-yr slope ≤ −3 pp/yr, ≥10 pp below the institute, ≥8 pp YoY drop, ≥30% single-recruiter concentration; stage risk compares the latest year's conversion with the prior-year average (≤ −5 pp = flag).
* **Cohorts:** cumulative % of each batch placed by season week; placement by CGPA band × batch. **Correlation:** Spearman.
* **What-if simulator:** levers = more eligible students, closing the skill-match gap, interview→offer uplift, more openings. Each application's win probability is
  P(shortlist)·P(interview)·P(selected | interviewed), with the last term capped by seat supply (openings ÷ expected interviewees). Stage models are logistic fits on the filtered data.
  Results are reported as **observed baseline + simulated change** (common random numbers), because the simulator ignores stop-applying-after-offer behaviour and would otherwise understate absolute levels.
  It is scenario analysis, not a forecast, and does not model salaries.
* **Predictive model:** P(placed) from CGPA, internships, certifications, projects, #skills, branch. Train 2020-2023 / test 2024 (temporal split), 5-fold CV on train, AUC, average precision, Brier vs base-rate baseline,
  permutation importance. No application/offer features (no leakage). Because the data is synthetic, the high AUC reflects the generator's built-in relationships.
* **Filters** (year, branch, company, industry, salary): year/branch filter students; company/industry filter applications and drives; salary removes offers outside the range;
  a drive stays in demand analytics if it allows ≥1 selected branch. Under a company/industry filter, "placed" means "holds an offer from the selected recruiters".

## SQL (`sql/queries.sql`)
| Query | Techniques |
|---|---|
| `branch_year_placement` | CTE chain, LEFT JOIN, `LAG` (YoY pp), `RANK() OVER (PARTITION BY year)` |
| `funnel_by_branch` | CTE, conditional aggregation, student-level funnel |
| `stage_conversion_by_year` | CTE, `LAG` on interview-stage conversion |
| `salary_by_industry` | `ROW_NUMBER`/`COUNT() OVER` to compute median & P90 (SQLite has no `PERCENTILE_CONT`) |
| `recruiter_trend` | 3-table join, `LAG`, running `SUM() OVER`, share-of-year window |
| `internship_impact` | correlated `EXISTS`, CGPA bands × internship bucket |
| `skill_gap` | two CTEs with scalar sub-queries, `RANK() OVER` |

## Findings on the seed-42 synthetic world (what the dashboard surfaces; recomputed live for any filter)
* **Overall:** 2,627 placement-seeking eligible students, **73.2%** placed; median package **9.1 LPA** vs mean 10.4 (right-skewed, skew 1.36).
* **Branch gap:** 2024 placement ranges from **88.4% (CSE) to 64.2% (ME)**, a 24 pp spread; ME is the only branch flagged (≥10 pp below the institute). EEE improves from 48.9% (2020) to 68.8% (2024).
* **Hiring cycle:** IT Services/Product Software offers fall sharply in 2023-24 while FinTech offers rise from 50 (2020) to 119 (2024); CSE/IT dip in 2023 (CSE 92.0 → 87.0, IT 88.6 → 80.3).
* **Salary inequality:** Gini 0.25; top 10% of placed students hold **20%** of package value; P90/P10 = 3.1x; 77 Tukey outliers retained and flagged.
* **Internships:** placement rises from **57.9% (0 internships) to 97.2% (3+)**; crude OR 3.9, **CGPA-adjusted OR 3.0 (95% CI 2.4-3.7)** - confounding by ability inflates the naive estimate by ~30%.
* **Skill gaps:** biggest shortfalls are Communication (+16 pp demand over supply), Excel (+12.5) and SQL (+7.0); Java/C++ are oversupplied relative to demand. Skills with demand > supply also show +21 to +25 pp placement lift.
* **Funnel:** the interview→offer step is the tightest (~32-40% by year) and swings most; CE's latest interview→offer conversion fell 6.1 pp vs its prior average (stage alert).
* **Simulator:** raising interview→offer conversion by 20% or closing 25% of the skill-match gap each add roughly +70-90 placed students; adding eligible students mostly adds marginal candidates competing for fixed seats.

## Limitations (say these in interviews)
Synthetic data encodes the relationships I chose, so "findings" validate the pipeline, not the real world. Internship results are associations. The simulator is a scenario tool, anchored to the observed baseline.
Placement here is binary per campus season (no offer-acceptance/decline modelling), and salary is not part of the simulator.

## Interview talking points
Business Analyst: funnel diagnostics, recruiter concentration (HHI), seat-fill vs candidate-supply bottlenecks. Data Analyst: cleaning log, outlier policy (flag, don't delete), SQL window functions.
Product Analyst: stage-conversion monitoring, risk register, what-if levers ranked by impact. Data Scientist: leakage-free features, temporal validation, confounder-adjusted effect (Mantel-Haenszel).
