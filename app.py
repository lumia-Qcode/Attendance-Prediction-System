"""Flask front-end: photo / PDF of a handwritten attendance register -> CSV -> predicted holidays.

Runs fully offline (OpenCV + Tesseract + numpy). No API keys, no LLM.
Start with:   python app.py        (then open http://127.0.0.1:5000)
"""
from __future__ import annotations

import dataclasses
import os
import re
import calendar
from datetime import date, timedelta

import cv2
import numpy as np
import pandas as pd
from flask import Flask, abort, redirect, render_template, request, send_file, url_for

import attendance as A
import jobs

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
ALLOWED = {".pdf", ".jpg", ".jpeg", ".png"}
MIN_MARKS = 15                      # a page with fewer 'P' marks is treated as an empty template


def month_range(start: date, end: date) -> list[tuple[int, int]]:
    out, (y, m) = [], (start.year, start.month)
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_label(y: int, m: int) -> str:
    return f"{calendar.month_name[m]} {y}"


def unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        k = n.lower()
        seen[k] = seen.get(k, 0) + 1
        out.append(n if seen[k] == 1 else f"{n} ({seen[k]})")
    return out


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024
    app.jinja_env.globals.update(month_label=month_label, WEEKDAYS=WEEKDAYS)

    def fail(msg: str, code: int = 400, back: str | None = None):
        return render_template("error.html", message=msg, back=back), code

    # ------------------------------------------------------------------ 1. upload
    @app.get("/")
    def index():
        return render_template("index.html", start="2026-09-01", end="2026-11-30", layout="auto")

    @app.post("/upload")
    def upload():
        files = [f for f in request.files.getlist("files") if f and f.filename]
        form = dict(start=request.form.get("start", ""), end=request.form.get("end", ""),
                    layout=request.form.get("layout", "auto"))

        def again(msg):
            return render_template("index.html", error=msg, **form), 400

        try:
            s_start, s_end = date.fromisoformat(form["start"]), date.fromisoformat(form["end"])
        except ValueError:
            return again("Please enter valid session start and end dates.")
        if s_end < s_start:
            return again("The session end date is before the start date.")
        if form["layout"] not in ("auto", "spread", "single"):
            form["layout"] = "auto"
        if not files:
            return again("Choose at least one file (PDF, JPG or PNG).")

        jobs.cleanup()
        images = []
        for f in files:
            if os.path.splitext(f.filename.lower())[1] not in ALLOWED:
                return again(f"'{f.filename}' is not a PDF / JPG / PNG file.")
            try:
                images += A.load_images(f.read(), f.filename)
            except Exception as e:
                return again(f"Could not open '{f.filename}': {e}")

        pages = A.analyse_pages(images, form["layout"])
        groups = A.group_pages(pages, MIN_MARKS)
        if not groups:
            return again("No filled attendance pages were found. Try another 'Photo layout' option or a clearer photo.")

        roster = jobs.load_roster()
        months = month_range(s_start, s_end)
        gdata, next_month = [], 0
        for g in groups:
            res = A.consensus(g)
            full_page = res.n_days_read == 31 and sum(x.n_present for x in g) / len(g) >= 60
            twin = any(d["default"] and A.looks_like_same_sheet(res, d["res"]) for d in gdata)
            default = None
            if full_page and not twin and next_month < len(months):   # complete month pages get months in order
                default = months[next_month]
                next_month += 1
            guesses = []
            if default:                                      # OCR names only for sheets that will be used
                for c in res.name_crops:
                    guess = A.ocr_name(c)
                    guesses.append(A.match_roster(guess, roster) or guess)
            gdata.append({"pages": [p.index for p in g], "res": res, "guesses": guesses, "default": default, "twin": twin})

        jid = jobs.new_id()
        jobs.save(jid, {"start": s_start, "end": s_end, "groups": gdata,
                        "pages": [{"index": p.index, "rows": p.result.n_rows if p.result else 0,
                                   "layout": p.result.layout if p.result else "-", "marks": p.n_present,
                                   "note": p.error or ("empty / template" if p.n_present < MIN_MARKS else "")}
                                  for p in pages]})
        return redirect(url_for("review", jid=jid))

    # ------------------------------------------------------------------ 2. review
    @app.get("/job/<jid>")
    def review(jid):
        job = jobs.load(jid)
        if not job:
            return fail("This upload has expired or does not exist. Please upload the sheet again.", 404, url_for("index"))
        months = month_range(job["start"], job["end"])
        glist, off_default = [], None
        for gi, g in enumerate(job["groups"]):
            res = g["res"]
            if off_default is None and g["default"]:
                off_default = A.infer_off_weekdays(res.off_cols, *g["default"])
            names = g["guesses"] or [""] * res.n_rows
            unc = res.uncertain if res.uncertain is not None else np.zeros_like(res.present)
            rows = [{"r": r, "n": r + 1, "name": names[r], "status": "dropped" if res.struck[r] else "active",
                     "marks": [bool(x) for x in res.present[r]], "unc": [bool(x) for x in unc[r]]}
                    for r in range(res.n_rows)]
            doubtful = [f"row {r + 1} day {d + 1}" for r, d in zip(*np.where(unc & ~res.struck[:, None]))]
            glist.append({"i": gi, "pages": ", ".join(map(str, g["pages"])), "n_rows": res.n_rows, "rows": rows,
                          "off": [bool(x) for x in res.off_cols], "doubtful": doubtful,
                          "twin": g.get("twin", False),
                          "default": f"{g['default'][0]}-{g['default'][1]:02d}" if g["default"] else "skip"})
        if off_default is None:
            off_default = A.infer_off_weekdays(job["groups"][0]["res"].off_cols, months[0][0], months[0][1])
        return render_template("review.html", jid=jid, job=job, groups=glist, months=months,
                               off_default=off_default or [5, 6])

    @app.get("/job/<jid>/overlay/<int:gi>.jpg")
    def overlay(jid, gi):
        job = jobs.load(jid)
        if not job or gi >= len(job["groups"]):
            abort(404)
        ok, buf = cv2.imencode(".jpg", job["groups"][gi]["res"].debug, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return app.response_class(buf.tobytes(), mimetype="image/jpeg")

    @app.get("/job/<jid>/crop/<int:gi>/<int:r>.png")
    def crop(jid, gi, r):
        job = jobs.load(jid)
        if not job or gi >= len(job["groups"]) or r >= job["groups"][gi]["res"].n_rows:
            abort(404)
        ok, buf = cv2.imencode(".png", job["groups"][gi]["res"].name_crops[r])
        return app.response_class(buf.tobytes(), mimetype="image/png")

    # ------------------------------------------------------------------ 3. predict
    @app.post("/job/<jid>/predict")
    def predict(jid):
        job = jobs.load(jid)
        if not job:
            return fail("This upload has expired. Please upload the sheet again.", 404, url_for("index"))
        back = url_for("review", jid=jid)
        f = request.form
        tables, used, crop_map = [], set(), {}
        for gi, g in enumerate(job["groups"]):
            v = f.get(f"month_{gi}", "skip")
            if v == "skip":
                continue
            try:
                y, mo = (int(x) for x in v.split("-"))
            except ValueError:
                return fail("Invalid month selection.", back=back)
            if (y, mo) in used:
                return fail(f"{month_label(y, mo)} is selected for two different sheets. Pick each month once.", back=back)
            used.add((y, mo))
            res = g["res"]
            names = unique_names([f.get(f"name_{gi}_{r}", "").strip() or f"Student {r + 1}" for r in range(res.n_rows)])
            for r, n in enumerate(names):                    # unnamed rows: remember the handwriting crop to show in results
                if not f.get(f"name_{gi}_{r}", "").strip() and n not in crop_map:
                    crop_map[n] = (gi, r)
            status = ["dropped" if f.get(f"status_{gi}_{r}") == "dropped" else "active" for r in range(res.n_rows)]
            pres = np.zeros((res.n_rows, 31), bool)
            for r in range(res.n_rows):
                for d in range(31):
                    pres[r, d] = f"p_{gi}_{r}_{d + 1}" in f          # ticked box = 'P'; anything else = absent
            tables.append(A.sheet_table(names, dataclasses.replace(res, present=pres), y, mo, status))
        if not tables:
            return fail("Pick a month for at least one sheet.", back=back)

        full = tables[0]
        for t in tables[1:]:
            full = full.merge(t, on="student", how="outer", suffixes=("", "_dup"))
            full["status"] = np.where(full[["status", "status_dup"]].eq("dropped").any(axis=1), "dropped", "active")
            full = full.drop(columns=[c for c in full.columns if c.endswith("_dup")])
        dcols = sorted(c for c in full.columns if c not in ("student", "status"))
        full = full[["student", "status"] + dcols].fillna("")

        off = {int(x) for x in f.getlist("off") if x.isdigit() and 0 <= int(x) <= 6}
        hol = set()
        for line in f.get("holidays", "").replace(",", "\n").splitlines():
            try:
                hol.add(date.fromisoformat(line.strip()))
            except ValueError:
                pass
        try:
            halflife = min(max(float(f.get("halflife", 14)), 7), 60)
        except ValueError:
            halflife = 14.0

        act = A.active_only(full)
        marked = [c for c in dcols if act[c].astype(str).str.upper().eq("P").any()]
        if not marked:
            return fail("No active student has a 'P' on any day - nothing to learn from.", back=back)
        observed_end = date.fromisoformat(max(marked))
        off_dates = {date.fromisoformat(c) for c in dcols if date.fromisoformat(c).weekday() in off} | hol
        long = A.long_format(full, off_dates, observed_end)
        if long.empty:
            return fail("No class days were found among the active students.", back=back)
        future = A.working_days(max(observed_end, job["start"]) + timedelta(days=1), job["end"], off, hol)
        if not future:
            return fail("There are no working days left in the session after the last attendance date.", back=back)

        model = A.fit_model(long, halflife)
        pred = A.predict_holidays(model, future)
        bt = A.backtest(long, halflife=halflife)

        full.to_csv(jobs.path(jid, "attendance.csv"), index=False)
        pred.to_csv(jobs.path(jid, "predictions.csv"), index=False)
        jobs.save_roster(jobs.load_roster() + [n for n in full["student"] if not n.startswith("Student ")])
        job["result"] = {
            "future_first": future[0], "future_last": future[-1], "n_future": len(future),
            "n_days": int(long["date"].nunique()), "n_students": int(long["student"].nunique()),
            "n_dropped": int((full["status"] == "dropped").sum()), "observed_end": observed_end,
            "prior": float(model.prior_mean),
            "weekday": ", ".join(f"{WEEKDAYS[i]} {model.weekday_factor[i]:.2f}" for i in range(7) if i not in off),
            "bt": bt, "pred": pred.to_dict("records"), "crops": crop_map}
        jobs.save(jid, job)
        return redirect(url_for("results", jid=jid))

    # ------------------------------------------------------------------ 4. results + downloads
    @app.get("/job/<jid>/results")
    def results(jid):
        job = jobs.load(jid)
        if not job or "result" not in job:
            return fail("No results yet for this upload.", 404, url_for("index"))
        rs = job["result"]
        top = max([r["predicted_holidays"] for r in rs["pred"]] + [1])
        for r in rs["pred"]:
            r["width"] = round(100 * r["predicted_holidays"] / top)
            gr = rs.get("crops", {}).get(r["student"])
            r["crop"] = url_for("crop", jid=jid, gi=gr[0], r=gr[1]) if gr else None
        n_unnamed = sum(1 for r in rs["pred"] if r["crop"])
        return render_template("results.html", jid=jid, rs=rs, end=job["end"], n_unnamed=n_unnamed)

    @app.get("/job/<jid>/download/<kind>.csv")
    def download(jid, kind):
        if kind not in ("attendance", "predictions") or not jobs.valid(jid):
            abort(404)
        p = jobs.path(jid, f"{kind}.csv")
        if not os.path.exists(p):
            abort(404)
        return send_file(p, as_attachment=True, download_name=f"{kind}.csv", mimetype="text/csv")

    return app


app = create_app()

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 5000))
    print(f"Attendance app running on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    serve(app, host=os.environ.get("HOST", "127.0.0.1"), port=port, threads=4)