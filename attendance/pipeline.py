"""Glue between the image reader and the rest: page analysis, duplicate grouping, consensus, CSV tables."""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from . import reader

__all__ = ["PageInfo", "analyse_pages", "group_pages", "consensus", "sheet_table",
           "active_only", "long_format", "infer_off_weekdays", "result_overlap", "looks_like_same_sheet"]


@dataclass
class PageInfo:
    index: int                              # 1-based page number
    result: reader.SheetResult | None
    error: str | None = None

    @property
    def n_present(self) -> int:
        return int(self.result.present.sum()) if self.result else 0


def _read_one(args):
    i, img, layout = args
    try:
        return PageInfo(i, reader.read_sheet(img, layout=layout))
    except Exception as e:                       # blank / non-register pages
        return PageInfo(i, None, str(e))


def analyse_pages(images, layout: str = "auto", workers: int | None = None) -> list[PageInfo]:
    """Read every page (in parallel: OpenCV releases the GIL, so threads scale across CPU cores)."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    jobs_ = [(i, img, layout) for i, img in enumerate(images, 1)]
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    if workers == 1 or len(jobs_) == 1:
        return [_read_one(j) for j in jobs_]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_read_one, jobs_))


def result_overlap(a: reader.SheetResult, b: reader.SheetResult) -> float:
    """Jaccard overlap of the 'P' marks of two readings (0 when they cannot be the same register page).

    Compared on the rows both captures have, so a photo that lost its last row still matches.
    """
    if a.n_days_read != b.n_days_read:
        return 0.0
    k = min(a.n_rows, b.n_rows)
    pa, pb = a.present[:k], b.present[:k]
    union = (pa | pb).sum()
    if union < 15:
        return 0.0
    return float((pa & pb).sum() / union)


def looks_like_same_sheet(a: reader.SheetResult, b: reader.SheetResult) -> bool:
    """True when two readings are probably two photos of the same register page.

    Strong overlap of the P marks, or - for a poorer capture whose rows drifted - the same highlighted
    off-day columns (a month-specific fingerprint) plus a moderate overlap.
    """
    ov = result_overlap(a, b)
    if ov > 0.6:
        return True
    same_off = a.n_days_read == b.n_days_read and bool((a.off_cols == b.off_cols).all()) and int(a.off_cols.sum()) >= 4
    return same_off and ov > 0.4


def _similar(a: reader.SheetResult, b: reader.SheetResult) -> bool:
    return a.n_rows == b.n_rows and result_overlap(a, b) > 0.75


def group_pages(pages: list[PageInfo], min_marks: int = 15) -> list[list[PageInfo]]:
    """Group captures of the same register page (people often photograph a page several times)."""
    groups: list[list[PageInfo]] = []
    for p in pages:
        if not p.result or p.n_present < min_marks:
            continue
        for g in groups:
            if _similar(g[0].result, p.result):
                g.append(p)
                break
        else:
            groups.append([p])
    return groups


def consensus(group: list[PageInfo]) -> reader.SheetResult:
    """Majority vote over repeated captures of one register page."""
    base = group[0].result
    if len(group) == 1:
        return base
    pres = np.mean([g.result.present for g in group], axis=0) >= 0.5
    struck = np.mean([g.result.struck for g in group], axis=0) >= 0.5
    off = np.mean([g.result.off_cols for g in group], axis=0) >= 0.5
    # keep the capture that agrees most with the consensus for names / debug picture
    best = max(group, key=lambda g: (g.result.present == pres).mean()).result
    unc = (np.mean([g.result.uncertain for g in group], axis=0) >= 0.5) & ~pres
    score = np.mean([g.result.cell_score for g in group], axis=0)
    return reader.SheetResult(best.n_rows, pres, off, struck, best.cell_ink, best.name_crops, best.debug,
                              best.layout, best.n_days_read, best.warnings, unc, score)


def sheet_table(names: list[str], res: reader.SheetResult, year: int, month: int,
                status: list[str] | None = None) -> pd.DataFrame:
    """Wide table: one row per student, one column per calendar day ('P' or '')."""
    ndays = calendar.monthrange(year, month)[1]
    rows = []
    for r in range(res.n_rows):
        row = {"student": names[r] if r < len(names) else f"Student {r + 1}",
               "status": (status[r] if status else ("dropped" if res.struck[r] else "active"))}
        for d in range(1, ndays + 1):
            row[date(year, month, d).isoformat()] = "P" if (d <= 31 and res.present[r, d - 1]) else ""
        rows.append(row)
    return pd.DataFrame(rows)


def active_only(wide: pd.DataFrame) -> pd.DataFrame:
    """Keep only students whose status is exactly 'active' (dropped-out / crossed-out rows are excluded)."""
    st = wide["status"].astype(str).str.strip().str.lower()
    return wide[st == "active"].copy()


def long_format(wide: pd.DataFrame, off_dates: set[date], last_observed: date | None = None) -> pd.DataFrame:
    """Wide -> long rows (student, date, present) for ACTIVE students on days when the class was held.

    * dropped-out students never enter the data the model learns from;
    * a working day counts as 'held' when at least one active student is marked 'P'
      (nobody present -> no class that day, so it is not an absence for anyone);
    * a mark counts as present only if it is a 'P' / 'p'.
    """
    date_cols = [c for c in wide.columns if c not in ("student", "status")]
    act = active_only(wide)
    recs = []
    for c in date_cols:
        d = date.fromisoformat(c)
        if d in off_dates or (last_observed and d > last_observed):
            continue
        marks = act[c].astype(str).str.strip().str.upper().eq("P")
        if not marks.any():
            continue
        for s, m in zip(act["student"], marks):
            recs.append((s, d, int(m)))
    return pd.DataFrame(recs, columns=["student", "date", "present"])


def infer_off_weekdays(off_cols: np.ndarray, year: int, month: int) -> list[int]:
    """Weekdays (Mon=0) that are highlighted as off in >=60% of their occurrences."""
    ndays = calendar.monthrange(year, month)[1]
    hit, tot = np.zeros(7), np.zeros(7)
    for d in range(1, min(ndays, 31) + 1):
        wd = date(year, month, d).weekday()
        tot[wd] += 1
        hit[wd] += bool(off_cols[d - 1])
    return [w for w in range(7) if tot[w] and hit[w] / tot[w] >= 0.6]
