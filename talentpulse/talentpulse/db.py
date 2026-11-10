import re
import sqlite3
import pandas as pd
from .config import DB_PATH, SQL_PATH

INDEXES = ["CREATE INDEX ix_app_student ON applications(student_id)", "CREATE INDEX ix_app_drive ON applications(drive_id)",
           "CREATE INDEX ix_drive_company ON drives(company_id)", "CREATE INDEX ix_ss ON student_skills(student_id)",
           "CREATE INDEX ix_ds ON drive_skills(drive_id)"]


def build_db(tables, path=DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(path)
    for name, df in tables.items():
        df.to_sql(name, con, index=False)
    for q in INDEXES:
        con.execute(q)
    con.commit()
    con.close()


def connect(path=DB_PATH):
    return sqlite3.connect(path)


def load_queries(path=SQL_PATH) -> dict[str, str]:
    parts = re.split(r"^-- name:\s*(\w+)\s*$", path.read_text(), flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_query(name, con=None):
    own = con is None
    con = con or connect()
    try:
        return pd.read_sql(load_queries()[name], con)
    finally:
        if own:
            con.close()
