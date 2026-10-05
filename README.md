# Attendance Digitiser & Holiday Predictor (Flask)

Photo / scan / PDF of a handwritten attendance register → CSV → predicted holidays (absences) per student.
**No API keys, no LLM, no Docker.** Python + Flask + OpenCV + Tesseract (optional) + numpy/pandas.

## Folder structure
```
attendance-predictor/
├── app.py                  Flask app (routes: upload → review → predict → results → downloads)
├── jobs.py                 tiny on-disk store for uploads (instance/, auto-deleted after 24 h)
├── requirements.txt        Python libraries
├── Procfile                start command for hosts that read one (gunicorn)
├── .gitignore
├── README.md
├── attendance/             the reusable module
│   ├── __init__.py         public API (__all__)
│   ├── reader.py           image/PDF -> P / absent grid (OpenCV, no ML)
│   ├── pipeline.py         page grouping, majority vote, tables, active-only filter
│   ├── predict.py          holiday prediction model (numpy/pandas)
│   ├── names.py            name OCR (Tesseract) + roster matching
│   └── tableio.py          CSV import / export
├── templates/              HTML pages (base, index, review, results, error)
├── static/                 style.css, app.js
├── tests/                  pytest tests
└── sample/                 put your own register PDF here (git-ignored)
```

## Run it locally
1. Install Python 3.10+ (python.org). Windows: tick "Add Python to PATH".
2. (Optional, for guessing names) install Tesseract: Windows - UB Mannheim installer, then add its folder to PATH;
   macOS `brew install tesseract`; Ubuntu `sudo apt install tesseract-ocr`.
3. In a terminal, inside this folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate          (Windows)      source .venv/bin/activate   (macOS / Linux)
   pip install -r requirements.txt
   python app.py
   ```
4. Open http://127.0.0.1:5000

## Tests
```
pip install pytest
pytest tests/test_predict.py                    # model only
SAMPLE_PDF=sample/Attendance_Register.pdf pytest tests    # everything (Windows: set SAMPLE_PDF=... first)
```

## Put it on GitHub
```
git init
git add .
git commit -m "Attendance digitiser (Flask)"
git branch -M main
git remote add origin https://github.com/<you>/attendance-predictor.git
git push -u origin main
```
(Create the empty repo on github.com first. When asked for a password use a Personal Access Token.)

## Deploy without Docker
Any host that runs Python web apps works. The start command is
`gunicorn app:app --workers 1 --threads 4 --timeout 300` (already in `Procfile`).
* **Render** (render.com): New → Web Service → connect the repo → Runtime *Python 3* → Build `pip install -r requirements.txt`
  → Start `gunicorn app:app --workers 1 --threads 4 --timeout 300`.
* **PythonAnywhere**: upload/clone the repo, make a virtualenv, `pip install -r requirements.txt`, create a Flask web app
  pointing at `app.py` (`app` object).
Notes: reading a big PDF takes ~3 s per page per CPU core, so use a host with a few hundred MB of RAM and a long request
timeout. Tesseract may not be installed on a host; the app still works (names come up blank - type them).
The app stores uploads under `instance/` and deletes them after 24 h; restrict who can open the site, it handles student names.

## How it reads the sheet
Rows crossed out with a pen line = dropped-out students (ignored for prediction). Pink highlighter columns = off days.
A day is **present only if a 'P' / 'p'-shaped stroke is found**; dots, specks and a neighbouring row's letter are absent.
Doubtful marks are shown in yellow for you to confirm.
