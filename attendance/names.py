"""Student-name handling: local Tesseract OCR guess + fuzzy matching to a roster (no cloud, no LLM).

Handwriting is the weak spot of classical OCR, so the app always lets the user confirm / fix names
once; confirmed names are saved to a roster so later months are matched automatically.
"""
from __future__ import annotations

import re
import cv2
import numpy as np

try:
    import pytesseract
    from rapidfuzz import fuzz, process
except Exception:  # pragma: no cover
    pytesseract = None
    fuzz = process = None

__all__ = ["ocr_name", "match_roster"]


def _prep(crop: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    b = crop.astype(np.int16)
    neutral = ((b[:, :, 2] - b[:, :, 0]) < 28)
    bg = cv2.medianBlur(g, 31)
    dark = np.clip(bg.astype(np.float32) - g.astype(np.float32), 0, 255) * neutral
    ink = (dark > 38).astype(np.uint8) * 255
    # remove long horizontal pen/strike lines and ruled lines
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.55 * ink.shape[1]), 1)))
    ink = cv2.subtract(ink, cv2.dilate(lines, np.ones((3, 3), np.uint8)))
    ink = cv2.dilate(ink, np.ones((2, 2), np.uint8))
    out = 255 - ink
    out = cv2.resize(out, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(out, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def ocr_name(crop: np.ndarray) -> str:
    if pytesseract is None:
        return ""
    try:
        txt = pytesseract.image_to_string(_prep(crop), config="--psm 7 --oem 1")
    except Exception:
        return ""
    txt = re.sub(r"[^A-Za-z .'()0-9-]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def match_roster(guess: str, roster: list[str], cutoff: int = 70) -> str | None:
    if not guess or not roster or process is None:
        return None
    hit = process.extractOne(guess, roster, scorer=fuzz.token_sort_ratio, score_cutoff=cutoff)
    return hit[0] if hit else None
