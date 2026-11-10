"""Optional predictive model: P(placed | student profile) with temporal validation (train <= year N-1, test = year N)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["cgpa", "internships", "certifications", "projects", "n_skills"]
CAT = ["branch"]


def _pipe(clf):
    pre = ColumnTransformer([("n", StandardScaler(), NUM), ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("clf", clf)])


def models():
    return {"Logistic Regression": _pipe(LogisticRegression(max_iter=1000)),
            "Gradient Boosting": _pipe(GradientBoostingClassifier(random_state=0, n_estimators=150, max_depth=3, learning_rate=.05))}


def evaluate(so: pd.DataFrame, seed: int = 0) -> dict:
    """`so` = student_outcomes of the FULL data. Features contain no application/offer information (no leakage)."""
    p = so[so.in_pool].copy()
    last = p.batch_year.max()
    tr, te = p[p.batch_year < last], p[p.batch_year == last]
    if tr.placed.nunique() < 2 or te.placed.nunique() < 2:
        return {}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows, fitted = [], {}
    for name, m in models().items():
        cvauc = cross_val_score(m, tr[NUM + CAT], tr.placed, cv=cv, scoring="roc_auc")
        m.fit(tr[NUM + CAT], tr.placed)
        pr = m.predict_proba(te[NUM + CAT])[:, 1]
        rows.append(dict(model=name, cv_auc_mean=cvauc.mean(), cv_auc_std=cvauc.std(), test_auc=roc_auc_score(te.placed, pr),
                         test_avg_precision=average_precision_score(te.placed, pr), test_brier=brier_score_loss(te.placed, pr)))
        fitted[name] = (m, pr)
    res = pd.DataFrame(rows)
    best = res.sort_values("test_auc", ascending=False).model.iloc[0]
    imp = permutation_importance(fitted[best][0], te[NUM + CAT], te.placed, scoring="roc_auc", n_repeats=10, random_state=seed)
    importance = pd.DataFrame({"feature": NUM + CAT, "auc_drop": imp.importances_mean, "std": imp.importances_std}).sort_values("auc_drop")
    base_brier = brier_score_loss(te.placed, np.full(len(te), tr.placed.mean()))
    final = _pipe(LogisticRegression(max_iter=1000)).fit(p[NUM + CAT], p.placed)
    return dict(results=res, importance=importance, best=best, train_years=f"{p.batch_year.min()}-{last - 1}", test_year=int(last),
                n_train=len(tr), n_test=len(te), baseline_brier=base_brier, predictor=final)
