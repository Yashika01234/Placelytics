import sqlite3
import numpy as np
import pandas as pd
import pytest
from talentpulse import analytics as A, whatif as W
from talentpulse.clean import clean
from talentpulse.config import YEARS
from talentpulse.db import build_db, load_queries, run_query
from talentpulse.filters import Filters, apply_filters, full_filters, load_frames
from talentpulse.generate_data import generate
from talentpulse.validate import validate


@pytest.fixture(scope="module")
def raw():
    return generate(42)


@pytest.fixture(scope="module")
def tables(raw):
    return clean(raw)


@pytest.fixture(scope="module")
def F(tables, tmp_path_factory):
    p = tmp_path_factory.mktemp("db") / "t.db"
    build_db(tables[0], p)
    return load_frames(p), p


# ---------- generation
def test_reproducible(raw):
    b = generate(42)
    pd.testing.assert_frame_equal(raw["applications"], b["applications"])
    pd.testing.assert_frame_equal(raw["students"], b["students"])
    assert not generate(7)["students"].cgpa.equals(raw["students"].cgpa)


def test_size_and_years(raw):
    assert 3000 <= raw["students"].student_id.nunique() <= 5000
    assert sorted(raw["students"].batch_year.unique()) == YEARS


def test_dirt_present_in_raw(raw):
    s, a = raw["students"], raw["applications"]
    assert s.cgpa.isna().any() and (s.cgpa > 10).any() and s.student_id.duplicated().any() and a.offer_lpa.max() > 1000


# ---------- cleaning + validation
def test_cleaning_fixes_everything(tables):
    t, rep = tables
    assert validate(t).passed.all()
    assert t["students"].cgpa.between(0.01, 10).all() and t["students"].branch.isin(["CSE", "IT", "ECE", "EEE", "ME", "CE"]).all()
    assert t["applications"].offer_lpa.max() < 150
    assert rep.rows_affected.sum() > 0


def test_validator_catches_corruption(tables):
    t = {k: v.copy() for k, v in tables[0].items()}
    t["applications"].loc[t["applications"].selected == 0, "interviewed"] = 1
    t["applications"].loc[0, "student_id"] = "NOPE"
    bad = validate(t)
    assert not bad.passed.all() and (~bad.passed).sum() >= 2


# ---------- SQL
def test_all_sql_queries_run(F):
    con = sqlite3.connect(F[1])
    for q in load_queries():
        assert len(run_query(q, con)) > 0, q
    con.close()


def test_sql_matches_pandas(F):
    fr, p = F
    so = A.student_outcomes(apply_filters(fr, full_filters(fr)))
    sql = run_query("branch_year_placement", sqlite3.connect(p))
    py = A.dept_benchmark(so)
    m = sql.merge(py, on=["branch", "batch_year"])
    assert len(m) == len(sql) and np.allclose(m.placement_pct, m.placement_rate, atol=0.06)


# ---------- analytics
def test_gini_known_values():
    assert A.gini([5, 5, 5, 5]) == pytest.approx(0, abs=1e-9)
    assert A.gini([0, 0, 0, 10]) == pytest.approx(0.75)


def test_mantel_haenszel_hand_example():
    rows = []
    for band in ["a", "b"]:  # identical 2x2 per stratum: a=20,b=10,c=10,d=20 -> OR = 4
        rows += [dict(cgpa_band=band, any_intern=1, placed=1)] * 20 + [dict(cgpa_band=band, any_intern=1, placed=0)] * 10
        rows += [dict(cgpa_band=band, any_intern=0, placed=1)] * 10 + [dict(cgpa_band=band, any_intern=0, placed=0)] * 20
    r = A.mantel_haenszel(pd.DataFrame(rows))
    assert r["mh_or"] == pytest.approx(4.0) and r["crude_or"] == pytest.approx(4.0) and r["ci_lo"] < 4 < r["ci_hi"]


def test_funnel_monotonic_and_consistent(F):
    G = apply_filters(F[0], full_filters(F[0]))
    fn = A.funnel(A.student_outcomes(G))
    assert (fn.students.diff().dropna() <= 0).all()
    k = A.kpis(G, A.student_outcomes(G))
    assert k["placed"] == fn.students.iloc[-1] and k["offers"] == G.apps.selected.sum()


def test_filters_partition_total(F):
    fr = F[0]
    full = full_filters(fr)
    tot = A.kpis(apply_filters(fr, full), A.student_outcomes(apply_filters(fr, full)))
    placed = 0
    for y in full.years:
        f = Filters([y], full.branches, full.companies, full.industries, full.salary)
        G = apply_filters(fr, f)
        placed += A.kpis(G, A.student_outcomes(G))["placed"]
    assert placed == tot["placed"]


def test_branch_and_salary_filters(F):
    fr, full = F[0], full_filters(F[0])
    G = apply_filters(fr, Filters(full.years, ["CSE"], full.companies, full.industries, full.salary))
    assert set(G.students.branch) == {"CSE"} and set(G.apps.branch) == {"CSE"}
    G2 = apply_filters(fr, Filters(full.years, full.branches, full.companies, full.industries, (10.0, 12.0)))
    o = G2.apps[G2.apps.selected == 1].offer_lpa
    assert o.between(10, 12).all() and len(o) > 0


def test_company_filter_limits_offers(F):
    fr, full = F[0], full_filters(F[0])
    c = fr.companies.company_name.iloc[0]
    G = apply_filters(fr, Filters(full.years, full.branches, [c], full.industries, full.salary))
    assert set(G.apps.company_name) <= {c} and set(G.drives.company_name) == {c}


def test_risk_and_skill_gap_run(F):
    G = apply_filters(F[0], full_filters(F[0])); so = A.student_outcomes(G)
    assert {"risk_level", "flags"} <= set(A.placement_risk(G, so).columns)
    sg = A.skill_gap(G, so)
    assert len(sg) == 15 and sg.demand_pct.between(0, 100).all() and sg.supply_pct.between(0, 100).all()


# ---------- what-if
def test_whatif_monotonic(F):
    G = apply_filters(F[0], full_filters(F[0])); so = A.student_outcomes(G)
    P = W.prepare(G, so)
    assert W.compare(P, W.Scenario(), 20)["delta_placed"] == pytest.approx(0)
    for sc in (W.Scenario(skill_pp=30), W.Scenario(conversion_pct=30), W.Scenario(openings_pct=30), W.Scenario(eligibility_pp=5)):
        assert W.compare(P, sc, 20)["delta_placed"] > 0
