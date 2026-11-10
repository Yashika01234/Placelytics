"""Data loading (via SQL joins) and the single filter engine used by every dashboard view."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .db import connect
from .config import DB_PATH


@dataclass
class Frames:
    students: pd.DataFrame
    apps: pd.DataFrame
    drives: pd.DataFrame
    student_skills: pd.DataFrame
    drive_skills: pd.DataFrame
    companies: pd.DataFrame


@dataclass
class Filters:
    years: list
    branches: list
    companies: list
    industries: list
    salary: tuple = (0.0, 1e9)


def load_frames(path=DB_PATH) -> Frames:
    con = connect(path)
    try:
        q = pd.read_sql
        students = q("SELECT * FROM students", con)
        drives = q("""SELECT d.*, c.company_name, c.industry, c.tier FROM drives d
                      JOIN companies c ON c.company_id = d.company_id""", con)
        apps = q("""SELECT a.*, d.batch_year, d.role, d.openings, d.season_week, d.company_id, c.company_name,
                           c.industry, c.tier, s.branch
                    FROM applications a JOIN drives d ON d.drive_id = a.drive_id
                    JOIN companies c ON c.company_id = d.company_id
                    JOIN students s ON s.student_id = a.student_id""", con)
        return Frames(students, apps, drives, q("SELECT * FROM student_skills", con),
                      q("SELECT * FROM drive_skills", con), q("SELECT * FROM companies", con))
    finally:
        con.close()


def full_filters(F: Frames) -> Filters:
    o = F.apps.loc[F.apps.selected == 1, "offer_lpa"]
    return Filters(sorted(F.students.batch_year.unique()), sorted(F.students.branch.unique()),
                   sorted(F.companies.company_name), sorted(F.companies.industry.unique()),
                   (float(np.floor(o.min())), float(np.ceil(o.max()))))


def apply_filters(F: Frames, f: Filters) -> Frames:
    """Semantics (documented in README):
    * year/branch filter students; company/industry filter applications and drives.
    * salary range removes *offers* outside the range (the whole application row is dropped).
    * a drive is kept for demand analytics if it matches year/company/industry and allows >=1 selected branch.
    """
    st = F.students[F.students.batch_year.isin(f.years) & F.students.branch.isin(f.branches)]
    a = F.apps
    m = a.batch_year.isin(f.years) & a.branch.isin(f.branches) & a.company_name.isin(f.companies) & a.industry.isin(f.industries)
    out_of_range = (a.selected == 1) & ~a.offer_lpa.between(f.salary[0], f.salary[1])
    ap = a[m & ~out_of_range]
    br = set(f.branches)
    dm = F.drives.batch_year.isin(f.years) & F.drives.company_name.isin(f.companies) & F.drives.industry.isin(f.industries) \
        & F.drives.allowed_branches.map(lambda x: bool(br & set(x.split("|"))))
    dr = F.drives[dm]
    return Frames(st, ap, dr, F.student_skills[F.student_skills.student_id.isin(st.student_id)],
                  F.drive_skills[F.drive_skills.drive_id.isin(dr.drive_id)], F.companies)
