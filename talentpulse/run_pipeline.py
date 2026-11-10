"""generate -> clean -> validate -> persist (CSV + SQLite).   Usage: python run_pipeline.py [--seed 42]"""
import argparse
from talentpulse.config import PROC, SEED
from talentpulse.generate_data import generate, write_raw
from talentpulse.clean import clean
from talentpulse.validate import validate
from talentpulse.db import build_db


def run_pipeline(seed=SEED, verbose=True):
    raw = generate(seed)
    write_raw(raw)
    tables, report = clean(raw)
    checks = validate(tables)
    if not checks.passed.all():
        raise RuntimeError("Validation failed:\n" + checks[~checks.passed].to_string())
    PROC.mkdir(parents=True, exist_ok=True)
    for k, df in tables.items():
        df.to_csv(PROC / f"{k}.csv", index=False)
    report.to_csv(PROC / "data_quality_report.csv", index=False)
    build_db({**tables, "dq_report": report})
    if verbose:
        print(f"SYNTHETIC data built (seed={seed}): " + ", ".join(f"{k}={len(v):,}" for k, v in tables.items()))
        print(report.to_string(index=False))
        print(f"{int(checks.passed.sum())}/{len(checks)} validation checks passed")
    return tables, report, checks


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    run_pipeline(p.parse_args().seed)
