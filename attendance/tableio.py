"""CSV import / export for attendance tables and predictions."""
from __future__ import annotations

import io
from datetime import date
from os import PathLike
from typing import BinaryIO

import pandas as pd

__all__ = ["export_attendance_csv", "import_attendance_csv", "export_predictions_csv"]


def export_attendance_csv(wide: pd.DataFrame, path: str | PathLike | None = None) -> str:
    """Write the wide table (student, status, one column per ISO date; 'P' or empty). Returns the CSV text."""
    text = wide.to_csv(index=False)
    if path is not None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(text)
    return text


def import_attendance_csv(src: str | PathLike | BinaryIO | bytes) -> pd.DataFrame:
    """Read a CSV produced by `export_attendance_csv` (or hand-made in the same shape).

    Accepts: path, file object or raw bytes. Columns: `student`, optional `status` (default 'active'),
    then date columns named YYYY-MM-DD. Anything other than P/p in a date cell is read as absent.
    """
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    df = pd.read_csv(src, dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    if "student" not in low:
        raise ValueError("CSV needs a 'student' column")
    df = df.rename(columns={low["student"]: "student"})
    if "status" in low:
        df = df.rename(columns={low["status"]: "status"})
    else:
        df["status"] = "active"
    date_cols = []
    for c in df.columns:
        if c in ("student", "status"):
            continue
        try:
            date.fromisoformat(c)
            date_cols.append(c)
        except ValueError:
            pass
    if not date_cols:
        raise ValueError("CSV needs date columns named YYYY-MM-DD")
    for c in date_cols:
        df[c] = df[c].str.strip().str.upper().where(df[c].str.strip().str.upper().eq("P"), "")
    df["student"] = df["student"].str.strip()
    df["status"] = df["status"].str.strip().str.lower().replace("", "active")
    return df[["student", "status", *sorted(date_cols)]]


def export_predictions_csv(pred: pd.DataFrame, path: str | PathLike | None = None) -> str:
    text = pred.to_csv(index=False)
    if path is not None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write(text)
    return text
