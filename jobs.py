"""Tiny on-disk job store so uploads survive between requests (and between server workers).

Each upload gets a random id and a folder  instance/jobs/<id>/  holding the analysed sheets and, later,
the generated CSV files.  Folders older than 24 h are deleted automatically (the data is about people).
"""
from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import time
import uuid

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instance")
JOBS = os.path.join(BASE, "jobs")
ROSTER = os.path.join(BASE, "roster.json")
_ID = re.compile(r"^[0-9a-f]{16}$")


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def valid(jid: str) -> bool:
    return bool(_ID.match(jid or ""))


def folder(jid: str) -> str:
    if not valid(jid):
        raise ValueError("bad job id")
    return os.path.join(JOBS, jid)


def path(jid: str, name: str) -> str:
    return os.path.join(folder(jid), name)


def save(jid: str, payload: dict) -> None:
    os.makedirs(folder(jid), exist_ok=True)
    tmp = path(jid, "job.pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path(jid, "job.pkl"))


def load(jid: str) -> dict | None:
    try:
        with open(path(jid, "job.pkl"), "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, ValueError):
        return None


def cleanup(max_age_hours: float = 24) -> None:
    if not os.path.isdir(JOBS):
        return
    limit = time.time() - max_age_hours * 3600
    for name in os.listdir(JOBS):
        p = os.path.join(JOBS, name)
        try:
            if os.path.isdir(p) and os.path.getmtime(p) < limit:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


# ---- roster: confirmed student names, used to auto-correct OCR guesses in later months
def load_roster() -> list[str]:
    try:
        with open(ROSTER, encoding="utf-8") as f:
            return list(json.load(f))
    except (FileNotFoundError, ValueError):
        return []


def save_roster(names: list[str]) -> None:
    os.makedirs(BASE, exist_ok=True)
    with open(ROSTER, "w", encoding="utf-8") as f:
        json.dump(sorted(set(names)), f, ensure_ascii=False, indent=1)
