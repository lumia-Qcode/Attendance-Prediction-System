from datetime import date, timedelta
import numpy as np, pandas as pd
from attendance import predict as M


def _synthetic(n_students=12, n_days=20, seed=0):
    rng = np.random.default_rng(seed)
    rates = rng.beta(2, 10, n_students)
    days = M.working_days(date(2026, 9, 1), date(2026, 9, 30), {4, 5, 6})[:n_days]
    rows = [(f"S{i}", d, int(rng.random() > rates[i])) for i in range(n_students) for d in days]
    return pd.DataFrame(rows, columns=["student", "date", "present"]), rates


def test_working_days_skip_weekly_off_and_holidays():
    d = M.working_days(date(2026, 10, 1), date(2026, 10, 31), {4, 5, 6}, {date(2026, 10, 12)})
    assert all(x.weekday() < 4 for x in d) and date(2026, 10, 12) not in d


def test_predictions_are_sane_and_ordered_by_risk():
    long, rates = _synthetic()
    m = M.fit(long)
    fut = M.working_days(date(2026, 10, 1), date(2026, 11, 30), {4, 5, 6})
    p = M.predict(m, fut)
    assert len(p) == 12 and (p["predicted_holidays"] >= 0).all()
    assert (p["predicted_holidays"] <= len(fut)).all()
    assert p["low_10pct"].le(p["high_90pct"]).all()
    assert p["predicted_holidays"].is_monotonic_decreasing


def test_always_present_student_gets_lowest_prediction():
    long, _ = _synthetic()
    long.loc[long.student == "S0", "present"] = 1
    long.loc[long.student == "S1", "present"] = 0
    p = M.predict(M.fit(long), M.working_days(date(2026, 10, 1), date(2026, 11, 30), {4, 5, 6})).set_index("student")
    assert p.loc["S0", "predicted_holidays"] < p.loc["S1", "predicted_holidays"]
