"""Data cleaning: every treatment is logged to a data-quality report."""
import numpy as np
import pandas as pd


def clean(raw):
    rep = []
    s, a = raw["students"].copy(), raw["applications"].copy()
    n = len(s)
    s = s.drop_duplicates("student_id", keep="first").reset_index(drop=True)
    rep.append(("Duplicate student rows", "students", n - len(s), "Dropped; first occurrence kept"))
    norm = s.branch.astype(str).str.replace(".", "", regex=False).str.strip().str.upper()
    rep.append(("Inconsistent branch labels", "students", int((norm != s.branch).sum()), "Normalised to upper-case codes"))
    s["branch"] = norm
    bad = (s.cgpa <= 0) | (s.cgpa > 10)
    rep.append(("Impossible CGPA (<=0 or >10)", "students", int(bad.sum()), "Set to missing, then imputed"))
    s.loc[bad, "cgpa"] = np.nan
    rep.append(("Missing CGPA (incl. above)", "students", int(s.cgpa.isna().sum()), "Imputed with branch x batch median; flagged cgpa_imputed"))
    s["cgpa_imputed"] = s.cgpa.isna().astype(int)
    s["cgpa"] = s.cgpa.fillna(s.groupby(["branch", "batch_year"]).cgpa.transform("median")).fillna(s.cgpa.median()).round(2)
    rep.append(("Missing certifications", "students", int(s.certifications.isna().sum()), "Imputed with branch median"))
    s["certifications"] = s.certifications.fillna(s.groupby("branch").certifications.transform("median")).round().astype(int)

    unit = a.offer_lpa > 1000
    rep.append(("Offer entered in rupees not LPA (>1000)", "applications", int(unit.sum()), "Divided by 1e5"))
    a.loc[unit, "offer_lpa"] = a.loc[unit, "offer_lpa"] / 1e5
    key = raw["drives"].merge(raw["companies"], on="company_id")[["drive_id", "role", "tier"]]
    a2 = a.merge(key, on="drive_id", how="left")
    miss = ((a2.selected == 1) & a2.offer_lpa.isna()).values
    med = a2[(a2.selected == 1) & a2.offer_lpa.notna()].groupby(["role", "tier"]).offer_lpa.median()
    fill = np.asarray(a2.set_index(["role", "tier"]).index.map(med), float)
    a.loc[miss, "offer_lpa"] = fill[miss]
    a.loc[miss & a.offer_lpa.isna().values, "offer_lpa"] = a.offer_lpa.median()
    rep.append(("Selected but offer amount missing", "applications", int(miss.sum()), "Imputed with role x tier median; flagged offer_imputed"))
    a["offer_imputed"] = miss.astype(int)
    return {**raw, "students": s, "applications": a}, pd.DataFrame(rep, columns=["issue", "table", "rows_affected", "treatment"])
