"""Data validation checks -> DataFrame(check, passed)."""
import pandas as pd
from .config import YEARS


def validate(t) -> pd.DataFrame:
    s, a, d, c = t["students"], t["applications"], t["drives"], t["companies"]
    ss, ds = t["student_skills"], t["drive_skills"]
    sel = a[a.selected == 1]
    per_drive = sel.groupby("drive_id").size().reindex(d.drive_id, fill_value=0).values
    ap = a.merge(s[["student_id", "batch_year", "eligible", "seeking_placement"]], on="student_id").merge(
        d[["drive_id", "batch_year"]], on="drive_id", suffixes=("_s", "_d"))
    checks = [
        ("student_id unique", s.student_id.is_unique),
        ("drive_id unique", d.drive_id.is_unique),
        ("app_id unique", a.app_id.is_unique),
        ("students between 3000 and 5000", 3000 <= len(s) <= 5000),
        ("5 academic years", sorted(s.batch_year.unique()) == YEARS),
        ("FK applications->students", a.student_id.isin(s.student_id).all()),
        ("FK applications->drives", a.drive_id.isin(d.drive_id).all()),
        ("FK drives->companies", d.company_id.isin(c.company_id).all()),
        ("FK student_skills->students", ss.student_id.isin(s.student_id).all()),
        ("FK drive_skills->drives", ds.drive_id.isin(d.drive_id).all()),
        ("CGPA in (0,10] and non-null", s.cgpa.notna().all() and s.cgpa.between(0.01, 10).all()),
        ("no negative counts", (s[["internships", "certifications", "projects", "active_backlogs"]] >= 0).all().all()),
        ("funnel monotonic (sel<=int<=short)", ((a.selected <= a.interviewed) & (a.interviewed <= a.shortlisted)).all()),
        ("offer present iff selected", (a.offer_lpa.notna() == (a.selected == 1)).all()),
        ("offers in (0,150] LPA", sel.offer_lpa.between(0.1, 150).all()),
        ("selected <= openings per drive", (per_drive <= d.openings.values).all()),
        ("applicants are eligible & seeking", ((ap.eligible == 1) & (ap.seeking_placement == 1)).all()),
        ("application year matches drive year", (ap.batch_year_s == ap.batch_year_d).all()),
        ("eligible implies CGPA>=6 (non-imputed)", ((s.eligible == 0) | (s.cgpa >= 6) | (s["cgpa_imputed"] == 1)).all()),
    ]
    out = pd.DataFrame(checks, columns=["check", "passed"])
    out["passed"] = out.passed.astype(bool)
    return out
