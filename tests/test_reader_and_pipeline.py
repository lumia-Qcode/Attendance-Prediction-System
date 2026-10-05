"""Regression tests for the cases reported by the user: dots must not count as 'P'; dropped students are excluded."""
import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

import attendance as A

PDF = os.environ.get("SAMPLE_PDF", "/mnt/user-data/uploads/Attendance_Register.pdf")
NAMES = ["Abdul Rehman", "Abdul Wahab", "Adeel Ahmad", "Adeel Bin Asghar", "Ahtasham Ali", "Ali Maaz", "Ali Suffyan",
         "Amna Nawaz", "Aneeq Ahmad", "Arif Ullah Khan", "Asad Ali", "Asad Ullah", "Asim Javed", "Aziz Ahmad",
         "Haseeb Ahmad", "Huma Aslam", "Hussnain Ali", "Hussnain Dawood", "Iqra Jahangir", "Khawaja Subhan",
         "Lumia Noman", "M. Shehzad", "Majid Yaseen", "Minahil Aftab", "Muazam Ali", "Muhammad Abdullah (553)",
         "Muhammad Abdullah (463)", "Muhammad Ans", "Nadir Ahmad", "Rizwan Riaz", "Syed M. Qasim Jamil",
         "Tahir Raza Shah", "Talha"]
DOTS = {"Adeel Ahmad": [28], "Huma Aslam": [29], "Khawaja Subhan": [24], "M. Shehzad": [29],
        "Majid Yaseen": [14, 29, 30]}


@pytest.fixture(scope="module")
def sept():
    if not os.path.exists(PDF):
        pytest.skip("sample PDF not available")
    imgs = A.load_images(open(PDF, "rb").read(), "register.pdf")
    pages = A.analyse_pages(imgs[3:8])
    return A.consensus(A.group_pages(pages)[0])


def test_rows_and_dropped_students(sept):
    assert sept.n_rows == 33
    assert list(np.where(sept.struck)[0] + 1) == [3, 11, 15, 16, 18, 22, 26, 33]


def test_pink_columns_are_off_days(sept):
    assert list(np.where(sept.off_cols)[0] + 1) == [4, 5, 6, 11, 12, 13, 18, 19, 20, 25, 26, 27]


@pytest.mark.parametrize("name,days", DOTS.items())
def test_dots_are_absent(sept, name, days):
    r = NAMES.index(name)
    for d in days:
        assert not sept.present[r, d - 1], f"{name} day {d}: a dot was read as P"


def test_clear_p_marks_still_found(sept):
    r = NAMES.index("Abdul Rehman")
    assert sept.present[r, [0, 1, 2, 7, 8, 9]].all()
    assert sept.present.sum() > 420


def test_prediction_uses_only_active_students(sept):
    wide = A.sheet_table(NAMES, sept, 2026, 9)
    off = {date(2026, 9, d) for d in range(1, 31) if date(2026, 9, d).weekday() in (4, 5, 6)}
    long = A.long_format(wide, off, date(2026, 9, 30))
    dropped = set(wide.loc[wide.status == "dropped", "student"])
    assert dropped and not (set(long["student"]) & dropped)
    pred = A.predict_holidays(A.fit_model(long), A.working_days(date(2026, 10, 1), date(2026, 11, 30), {4, 5, 6}))
    assert set(pred["student"]) == set(wide.loc[wide.status == "active", "student"])


def test_csv_roundtrip(tmp_path):
    wide = pd.DataFrame({"student": ["A", "B"], "status": ["active", "dropped"],
                         "2026-09-01": ["P", ""], "2026-09-02": ["", "P"]})
    p = tmp_path / "a.csv"
    A.export_attendance_csv(wide, p)
    back = A.import_attendance_csv(p)
    assert back.equals(wide)
    assert list(A.active_only(back)["student"]) == ["A"]
