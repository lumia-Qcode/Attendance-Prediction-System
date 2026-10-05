"""Absence ("holiday") prediction.

Model: hierarchical empirical-Bayes Beta-Binomial per student, with
  * exponential recency weighting (recent behaviour counts more),
  * a pooled day-of-week effect (e.g. Mondays are worse for everybody),
  * shrinkage towards the class average so one bad week does not label a student.
Predictive intervals come from simulating from the posterior. Pure numpy/pandas - no external services.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

__all__ = ["working_days", "Fit", "fit", "predict", "backtest"]


def working_days(start: date, end: date, off_weekdays: set[int], holidays: set[date] | None = None) -> list[date]:
    holidays = holidays or set()
    d, out = start, []
    while d <= end:
        if d.weekday() not in off_weekdays and d not in holidays:
            out.append(d)
        d += timedelta(days=1)
    return out


@dataclass
class Fit:
    prior_mean: float
    kappa: float
    weekday_factor: np.ndarray           # (7,)
    halflife: float
    students: list[str]
    A: np.ndarray                        # weighted absences
    N: np.ndarray                        # weighted exposure
    n_days: np.ndarray                   # raw observed days
    raw_abs: np.ndarray                  # raw absences


def fit(long: pd.DataFrame, halflife: float = 14.0) -> Fit:
    if long.empty:
        raise ValueError("No observed attendance days to learn from.")
    df = long.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["absent"] = 1 - df["present"]
    t_end = df["date"].max()
    df["w"] = 0.5 ** ((t_end - df["date"]).dt.days / halflife)
    overall = df["absent"].mean()
    # pooled weekday factor (shrunk to 1 with 10 pseudo-observations)
    df["wd"] = df["date"].dt.weekday
    f = np.ones(7)
    for wd, g in df.groupby("wd"):
        n = len(g)
        rate = (g["absent"].sum() + 10 * overall) / (n + 10)
        f[wd] = rate / max(overall, 1e-6)
    # remove weekday effect before estimating per-student rates
    df["exp_abs"] = overall * f[df["wd"].to_numpy()]
    g = df.groupby("student")
    students = list(g.groups.keys())
    A = g.apply(lambda x: (x["absent"] * x["w"]).sum(), include_groups=False).reindex(students).to_numpy()
    N = g["w"].sum().reindex(students).to_numpy()
    n_days = g.size().reindex(students).to_numpy()
    raw = g["absent"].sum().reindex(students).to_numpy()
    # method-of-moments Beta prior across students
    rates = raw / np.maximum(n_days, 1)
    mu = float(np.clip(rates.mean(), 0.01, 0.9))
    var_between = max(rates.var(ddof=1) if len(rates) > 1 else 0.0, 1e-6)
    var_binom = np.mean(mu * (1 - mu) / np.maximum(n_days, 1))
    extra = max(var_between - var_binom, 1e-4)
    kappa = float(np.clip(mu * (1 - mu) / extra - 1, 2.0, 40.0))
    return Fit(mu, kappa, f, halflife, students, A, N, n_days, raw)


def predict(m: Fit, future_days: list[date], n_sims: int = 4000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    S = len(m.students)
    a = m.prior_mean * m.kappa + m.A
    b = (1 - m.prior_mean) * m.kappa + (m.N - m.A)
    wd = np.array([d.weekday() for d in future_days])
    fac = m.weekday_factor[wd] if len(wd) else np.zeros(0)
    p_mean = a / (a + b)
    exp_abs = np.array([np.clip(p_mean[i] * fac, 0, 0.95).sum() for i in range(S)])
    p_draw = rng.beta(a[:, None], b[:, None], size=(S, n_sims))          # (S, sims)
    sims = np.zeros((S, n_sims))
    for f_ in fac:
        sims += rng.random((S, n_sims)) < np.clip(p_draw * f_, 0, 0.95)
    out = pd.DataFrame({
        "student": m.students,
        "observed_days": m.n_days,
        "observed_absences": m.raw_abs,
        "absence_rate_so_far": np.round(m.raw_abs / np.maximum(m.n_days, 1), 3),
        "future_working_days": len(future_days),
        "predicted_holidays": np.round(exp_abs, 1),
        "predicted_holidays_int": np.round(exp_abs).astype(int),
        "low_10pct": np.percentile(sims, 10, axis=1).astype(int),
        "high_90pct": np.percentile(sims, 90, axis=1).astype(int),
        "prob_3plus_holidays": np.round((sims >= 3).mean(axis=1), 2),
    })
    out["predicted_total_absences"] = out["observed_absences"] + out["predicted_holidays"]
    out["risk"] = pd.cut(out["predicted_holidays"] / max(len(future_days), 1),
                         [-1, 0.08, 0.2, 2], labels=["low", "medium", "high"])
    return out.sort_values("predicted_holidays", ascending=False).reset_index(drop=True)


def backtest(long: pd.DataFrame, train_frac: float = 0.65, halflife: float = 14.0) -> dict:
    """Train on the first part of the observed days, predict the absences in the rest."""
    days = sorted(long["date"].unique())
    if len(days) < 8:
        return {}
    k = max(int(len(days) * train_frac), 4)
    tr_days, te_days = days[:k], days[k:]
    tr, te = long[long["date"].isin(tr_days)], long[long["date"].isin(te_days)]
    m = fit(tr, halflife)
    pr = predict(m, [pd.Timestamp(d).date() for d in te_days], n_sims=500)
    truth = (1 - te).groupby("student")["present"].sum() if False else te.assign(a=1 - te["present"]).groupby("student")["a"].sum()
    pr = pr.set_index("student")
    truth = truth.reindex(pr.index).fillna(0)
    glob = tr.assign(a=1 - tr["present"])["a"].mean() * len(te_days)
    raw = (pr["observed_absences"] / pr["observed_days"].clip(lower=1)) * len(te_days)
    mae = lambda x: float(np.mean(np.abs(x - truth)))
    return {"train_days": len(tr_days), "test_days": len(te_days), "students": len(pr),
            "mae_model": mae(pr["predicted_holidays"]), "mae_class_average": mae(pd.Series(glob, index=pr.index)),
            "mae_raw_rate": mae(raw)}
