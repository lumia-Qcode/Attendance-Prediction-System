"""End-to-end test of the Flask app with the real sample register (skipped if the PDF is missing)."""
import io
import os
import re
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import app  # noqa: E402

PDF = os.environ.get("SAMPLE_PDF", "sample/Attendance_Register.pdf")


@pytest.fixture(scope="module")
def client():
    return app.test_client()


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Upload the attendance register" in r.data


def test_rejects_bad_upload(client):
    r = client.post("/upload", data={"start": "2026-09-01", "end": "2026-11-30"})
    assert r.status_code == 400 and b"Choose at least one file" in r.data
    r = client.post("/upload", data={"start": "2026-09-01", "end": "2026-11-30",
                                     "files": (io.BytesIO(b"x"), "notes.txt")}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_job_id_is_validated(client):
    assert client.get("/job/../../etc/passwd").status_code == 404
    assert client.get("/job/zzzz/download/attendance.csv").status_code == 404


@pytest.mark.skipif(not os.path.exists(PDF), reason="sample PDF not available")
def test_full_flow(client):
    t = time.time()
    r = client.post("/upload", data={"start": "2026-09-01", "end": "2026-11-30", "layout": "auto",
                                     "files": (open(PDF, "rb"), "Attendance_Register.pdf")},
                    content_type="multipart/form-data")
    print("upload seconds:", round(time.time() - t, 1))
    assert r.status_code == 302
    jid = re.search(r"/job/([0-9a-f]{16})", r.headers["Location"]).group(1)
    page = client.get(f"/job/{jid}")
    assert page.status_code == 200 and b"Review" in page.data
    assert client.get(f"/job/{jid}/overlay/0.jpg").status_code == 200
    assert client.get(f"/job/{jid}/crop/0/0.png").status_code == 200
    # submit the form exactly as the browser would: read defaults back out of the rendered HTML
    html = page.data.decode()
    form = {}
    for m in re.finditer(r'<select name="(month_\d+)">.*?</select>', html, re.S):
        sel = re.search(r'<option value="([^"]+)" selected', m.group(0))
        form[m.group(1)] = sel.group(1) if sel else "skip"
    for m in re.finditer(r'<input class="nm" type="text" name="(name_\d+_\d+)" value="([^"]*)"', html):
        form[m.group(1)] = "Student " + m.group(1).split("_")[-1]       # fixed names -> deterministic test
    for m in re.finditer(r'<select class="st" name="(status_\d+_\d+)">.*?</select>', html, re.S):
        sel = re.search(r'<option value="([^"]+)" selected', m.group(0))
        form[m.group(1)] = sel.group(1)
    for m in re.finditer(r'name="(p_\d+_\d+_\d+)" checked', html):
        form[m.group(1)] = "on"
    form["off"] = ["4", "5", "6"]
    assigned = [v for k, v in form.items() if k.startswith("month_") and v != "skip"]
    assert assigned == ["2026-09"], assigned            # the September spread only; duplicates / templates skipped
    r = client.post(f"/job/{jid}/predict", data=form)
    assert r.status_code == 302, r.data[:400]
    res = client.get(f"/job/{jid}/results")
    assert res.status_code == 200 and b"Predicted holidays" in res.data
    csv = client.get(f"/job/{jid}/download/predictions.csv")
    assert csv.status_code == 200 and csv.data.startswith(b"student,")
    att = client.get(f"/job/{jid}/download/attendance.csv")
    assert att.status_code == 200
    import pandas as pd
    df = pd.read_csv(io.BytesIO(att.data))
    assert len(df) == 33 and df["status"].value_counts().to_dict() == {"active": 25, "dropped": 8}
    pred = pd.read_csv(io.BytesIO(csv.data))
    assert len(pred) == 25                              # only active students are predicted
    assert pred["future_working_days"].iloc[0] == 34
