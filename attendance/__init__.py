"""Attendance sheet digitiser + absence predictor (classical CV + statistics; no LLM, no API keys).

Typical use
-----------
>>> from attendance import (load_images, read_sheet, sheet_table, long_format, fit_model, predict_holidays,
...                         working_days, export_attendance_csv)
>>> img = load_images(open("register.pdf", "rb").read(), "register.pdf")[0]
>>> res = read_sheet(img)                                  # P / absent per cell, off-days, crossed-out rows
>>> wide = sheet_table(names, res, 2026, 9)                # one row per student, one column per date
>>> long = long_format(wide, off_dates)                    # ACTIVE students only
>>> pred = predict_holidays(fit_model(long), working_days(date(2026, 10, 1), date(2026, 11, 30), {4, 5, 6}))
"""
from .reader import (N_LEFT, N_RIGHT, HalfLayout, SheetResult, load_images, normalise_size, read_sheet)
from .pipeline import (PageInfo, active_only, analyse_pages, consensus, group_pages, infer_off_weekdays,
                       long_format, looks_like_same_sheet, result_overlap, sheet_table)
from .predict import Fit, backtest, working_days
from .predict import fit as fit_model          # aliases: a bare `predict` / `fit` would shadow the submodule
from .predict import predict as predict_holidays
from .names import match_roster, ocr_name
from .tableio import export_attendance_csv, export_predictions_csv, import_attendance_csv

__version__ = "1.1.0"
__all__ = [
    # reading
    "load_images", "normalise_size", "read_sheet", "SheetResult", "HalfLayout", "N_LEFT", "N_RIGHT",
    # pipeline
    "PageInfo", "analyse_pages", "group_pages", "consensus", "sheet_table", "active_only", "long_format",
    "infer_off_weekdays", "result_overlap", "looks_like_same_sheet",
    # prediction
    "Fit", "fit_model", "predict_holidays", "backtest", "working_days",
    # names
    "ocr_name", "match_roster",
    # csv in / out
    "export_attendance_csv", "import_attendance_csv", "export_predictions_csv",
]
