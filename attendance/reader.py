"""Read a photographed / scanned handwritten attendance register with classical computer vision.

Pipeline (no ML models, no external APIs):
  1. split the photo into the two register pages (left: names + days 1-12, right: days 13-31)
  2. de-skew each page (shear search on the ruled lines)
  3. fit a regular lattice to the printed ruled lines -> cell boxes
  4. a cell is 'present' when it holds a pen stroke big enough to be a 'P' (not a dot / grid line)
  5. pink highlighter columns -> off days; long pen line through a row -> student dropped out
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

N_LEFT, N_RIGHT = 12, 19          # printed day columns on the left / right page (days 1-12 / 13-31)

__all__ = ["load_images", "normalise_size", "read_sheet", "SheetResult", "HalfLayout",
           "N_LEFT", "N_RIGHT", "MIN_P_HEIGHT", "MIN_P_WIDTH", "MIN_P_AREA"]


# --------------------------------------------------------------------------- loading
def load_images(data: bytes, filename: str) -> list[np.ndarray]:
    """Return a list of BGR images (one per PDF page, or a single image)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        import pymupdf
        doc = pymupdf.open(stream=data, filetype="pdf")
        pages = []
        for page in doc:
            img = None
            imgs = page.get_images(full=True)
            if imgs:  # scanned PDFs: take the embedded photo at native resolution
                xref = max(imgs, key=lambda i: i[2] * i[3])[0]
                raw = doc.extract_image(xref)["image"]
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                zoom = 3000 / max(page.rect.width, page.rect.height)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
                img = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3),
                                   cv2.COLOR_RGB2BGR)
            pages.append(img)
        return pages
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return [img]


def normalise_size(img: np.ndarray, target_w: int = 3000) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > 1.3 * target_w or max(h, w) < 0.6 * target_w:
        s = target_w / max(h, w)
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return img


# --------------------------------------------------------------------------- helpers
def _masks(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(g, 51)
    dark = np.clip(bg.astype(np.float32) - g.astype(np.float32), 0, 255)
    b = img.astype(np.int16)
    neutral = (b[:, :, 2] - b[:, :, 0]) < 28          # not red/pink  -> pen ink, grey grid lines
    return dark, dark * neutral


def _shear_x(a, s):
    h, w = a.shape[:2]
    M = np.float32([[1, s, -s * h / 2], [0, 1, 0]])
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0 if a.ndim == 2 else (235, 235, 235))


def _shear_y(a, t):
    h, w = a.shape[:2]
    M = np.float32([[1, 0, 0], [t, 1, -t * w / 2]])
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0 if a.ndim == 2 else (235, 235, 235))


def _score_shear(mask, fn, axis, s):
    p = fn(mask, s).sum(axis=axis).astype(np.float64)
    return float((np.diff(p) ** 2).sum())


def _best_shear(mask, fn, axis, rng=0.03, step=0.0005, down=1, coarse=3, ntop=3):
    """Shear angle that makes the ruled lines sharpest.

    Coarse-to-fine (same answer as sweeping every step at full resolution, several times faster):
    sweep every `coarse`-th step (optionally on a down-scaled mask), then refine around the best `ntop`.
    """
    small = cv2.resize(mask, None, fx=1 / down, fy=1 / down, interpolation=cv2.INTER_AREA) if down > 1 else mask
    grid = np.arange(-rng, rng + 1e-9, step * coarse)
    cs = [_score_shear(small, fn, axis, s) for s in grid]
    top = sorted(range(len(grid)), key=lambda i: -cs[i])[:ntop]
    cands = {round(float(grid[i] + d * step), 6) for i in top for d in range(-coarse, coarse + 1)}
    cands = [c for c in cands if abs(c) <= rng + 1e-9]
    return max(cands, key=lambda s: _score_shear(mask, fn, axis, s))


def _fit_lattice(profile, n, lo, hi, dmin, dmax, prefer="high"):
    """Find x0, d such that profile is high at x0 + k*d, k=0..n.

    Several shifted lattices can score almost the same (e.g. one that swallows the Rank column);
    among near-best candidates keep the right-most ('high', left page: date block touches the page
    edge) or left-most ('low', right page) one.
    """
    ks = np.arange(n + 1)
    cands = []
    for d in np.arange(dmin, dmax, 0.25):
        x0s = np.arange(lo, max(lo + 1, hi - n * d), 1.0)
        idx = np.clip(np.round(x0s[:, None] + d * ks[None, :]).astype(int), 0, len(profile) - 1)
        sc = profile[idx].mean(axis=1)
        j = int(np.argmax(sc))
        cands.append((float(sc[j]), float(x0s[j]), float(d)))
    top = max(c[0] for c in cands)
    good = [c for c in cands if c[0] >= 0.9 * top]
    pick = max(good, key=lambda c: c[1]) if prefer == "high" else min(good, key=lambda c: c[1])
    return pick[1], pick[2]


@dataclass
class HalfLayout:
    img: np.ndarray                 # de-skewed colour crop of this page
    xs: np.ndarray                  # column boundaries (n+1)
    ys: np.ndarray                  # row boundaries (n_rows+1)
    dark: np.ndarray
    ink: np.ndarray                 # cleaned pen-ink mask


@dataclass
class SheetResult:
    n_rows: int
    present: np.ndarray             # (n_rows, 31) bool   True = 'P' seen
    off_cols: np.ndarray            # (31,) bool          pink highlighter columns
    struck: np.ndarray              # (n_rows,) bool      row crossed out -> dropped out
    cell_ink: np.ndarray            # (n_rows, 31) float  ink pixels per cell (confidence)
    name_crops: list = field(default_factory=list)
    debug: np.ndarray | None = None
    layout: str = "spread"
    n_days_read: int = 31
    warnings: list = field(default_factory=list)
    uncertain: np.ndarray | None = None   # (n_rows, 31) bool  mark that is P-like but doubtful (check by eye)
    cell_score: np.ndarray | None = None  # (n_rows, 31) float  tallest stroke in the cell / row height


# --------------------------------------------------------------------------- page analysis
def _find_gutter(img):
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gm = gaussian_filter1d(g[int(.15 * h):int(.85 * h)].mean(axis=0), 3)
    mid = np.arange(int(.4 * w), int(.6 * w))
    x = int(mid[np.argmin(gm[mid])])
    contrast = np.median(gm) - gm[x]
    return x, contrast


def _prep_half(crop, n_cols, is_left, W_full):
    """De-skew a page and fit its column lattice. Returns (img, x0, d)."""
    h, w = crop.shape[:2]
    dark, dn = _masks(crop)
    m = (dn > 12).astype(np.uint8)
    mv = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120)))
    mv = mv[int(.12 * h):int(.88 * h)] * 255
    sx = _best_shear(mv, _shear_x, 0, down=4, coarse=2)
    img = _shear_x(crop, sx)
    dark, dn = _masks(img)
    mh = cv2.morphologyEx((dark > 12).astype(np.uint8), cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1)))
    mh = mh[:, int(.35 * w):int(.95 * w)] * 255 if is_left else mh[:, int(.05 * w):int(.8 * w)] * 255
    sy = _best_shear(mh, _shear_y, 1, rng=0.02, step=0.0004, coarse=2, ntop=5)
    img = _shear_y(img, sy)
    dark, dn = _masks(img)
    t = cv2.morphologyEx((dn > 12).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 200)))
    vp = gaussian_filter1d(t[int(.1 * h):int(.9 * h)].sum(axis=0).astype(np.float64), 1.2)
    lo = int(.3 * w) if is_left else 0
    x0, d = _fit_lattice(vp, n_cols, lo, w - 5, .028 * w, .052 * w, "high" if is_left else "low")
    return img, x0, d


def _row_lattice(img, x_from, x_to):
    """Row boundaries from the ruled lines. Returns (y_body_top, pitch)."""
    h, w = img.shape[:2]
    dark, _ = _masks(img)
    rowp = dark[:, int(x_from):int(x_to)].mean(axis=1)
    hp = gaussian_filter1d(rowp - gaussian_filter1d(rowp, 6), 1.2)
    seg = hp[int(.12 * h):int(.95 * h)]
    seg = seg - seg.mean()
    ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
    ac /= max(ac[0], 1e-9)
    lo, hi = int(.014 * h), int(.04 * h)
    cands = [(d, ac[d]) for d in range(lo, hi) if ac[d] >= ac[d - 1] and ac[d] >= ac[d + 1]]
    if not cands:
        raise ValueError("Could not find the ruled rows - is this a register page?")
    top = max(c for _, c in cands)
    d0 = min(d for d, c in cands if c >= 0.85 * top)
    # body top = the red double rule that closes the printed header
    b = img.astype(np.int16)
    red = np.clip((b[:, :, 2] - b[:, :, 0]) - 20, 0, 255).astype(np.float32)
    rr = red[:, : max(int(x_from) - 5, 10)].mean(axis=1)
    rr = gaussian_filter1d(rr, 1.0)
    thr_r = max(rr[: int(.3 * h)].mean() + 1.5 * rr[: int(.3 * h)].std(), 1.0)
    pk = [y for y in range(3, int(.22 * h)) if rr[y] > rr[y - 1] and rr[y] >= rr[y + 1] and rr[y] > thr_r]
    if pk:
        ytop = pk[-1]            # lowest red rule in the header zone = rule closing the header
    else:
        pk2 = [y for y in range(3, int(.3 * h)) if hp[y] > hp[y - 1] and hp[y] >= hp[y + 1] and hp[y] > 0.8]
        ytop = pk2[0] if pk2 else int(.1 * h)
    best = (-1e9, ytop, d0)
    for d in np.arange(d0 * .97, d0 * 1.03, .1):
        for y0 in range(ytop - 5, ytop + 6):
            idx = np.round(y0 + d * np.arange(0, 28)).astype(int)
            idx = idx[idx < h]
            sc = hp[idx].mean()
            if sc > best[0]:
                best = (sc, y0, d)
    return float(best[1]), float(best[2])


def _clean_ink(img):
    """Pen-ink mask: dark, not pink, with the printed grid lines removed."""
    dark, dn = _masks(img)
    ink = (dn > 38).astype(np.uint8)
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 55)))
    lines |= cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1)))
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    return ink & (1 - lines), dark, ink


def _pink_fraction(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (((hh >= 150) | (hh <= 6)) & (ss > 40) & (vv > 140)).astype(np.uint8)


def _otsu_log(v):
    v = np.log1p(v[np.isfinite(v)])
    if v.size < 10 or v.max() - v.min() < 1:
        return float(np.expm1(v.max())) if v.size else 1.0
    hist, edges = np.histogram(v, bins=60)
    p = hist / hist.sum()
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p); mu = np.cumsum(p * centres); mu_t = mu[-1]
    sb = (mu_t * w0 - mu) ** 2 / np.maximum(w0 * (1 - w0), 1e-9)
    return float(np.expm1(centres[int(np.argmax(sb))]))


# A real 'P' (upper- or lower-case) is a tall pen stroke with a bowl. Dots, specks and the tail of a
# neighbouring row's letter are not.  Thresholds are fractions of the row height / column width so they
# work at any photo resolution.
MIN_P_HEIGHT = 0.30        # stroke height / row height
MIN_P_WIDTH = 0.14         # stroke width / column width  (rejects a bare vertical tick)
MIN_P_AREA = 0.012         # ink area / (row height * column width)
DOUBT_HEIGHT = 0.40        # P-like but below this height -> flagged for a human look
CLIP_KEEP_HEIGHT = 0.45    # a stroke cut by a row line is only trusted if what is left in this row is this tall
DOT_HEIGHT = 0.16          # smaller than this = speck / dot, never a mark


def _detect_marks(hl: "HalfLayout"):
    """Per-cell decision from connected pen strokes.

    Every stroke (connected blob of ink, grid lines already removed) is assigned to the cell that holds
    its centre, so the foot of a letter written low in the row above never leaks into this cell.
    Returns (ink_sum, score, present, uncertain), each shaped (n_rows, n_cols).
    """
    nr, nc = len(hl.ys) - 1, len(hl.xs) - 1
    ink_sum = np.zeros((nr, nc)); score = np.zeros((nr, nc))
    present = np.zeros((nr, nc), bool); unsure = np.zeros((nr, nc), bool)
    if nr <= 0:
        return ink_sum, score, present, unsure
    ink = hl.ink.astype(np.uint8)
    # re-join strokes that the grid-line removal cut where a letter crosses a ruling
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    ys, xs = hl.ys, hl.xs
    # cut along every row boundary so letters in neighbouring rows can never fuse into one stroke
    for y in ys:
        y = int(round(y))
        ink[max(y - 1, 0):y + 2, :] = 0
    n, _, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    for i in range(1, n):
        cx, cy = cen[i]
        c = int(np.searchsorted(xs, cx) - 1)
        r = int(np.searchsorted(ys, cy) - 1)
        if not (0 <= c < nc and 0 <= r < nr):
            continue
        ph, pw = ys[r + 1] - ys[r], xs[c + 1] - xs[c]
        h, w, area = st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_AREA]
        ink_sum[r, c] += area
        hr, wr, ar = h / ph, w / pw, area / (ph * pw)
        top, bottom = st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_TOP] + h
        f = (cy - ys[r]) / ph
        clipped = ((bottom >= ys[r + 1] - 3 and f > 0.70) or (top <= ys[r] + 3 and f < 0.30)) and hr < CLIP_KEEP_HEIGHT
        if clipped:                        # foot / head of a letter that belongs to the row above or below
            if hr >= DOT_HEIGHT:
                unsure[r, c] = True
            continue
        if hr > score[r, c]:
            score[r, c] = hr
        if hr >= MIN_P_HEIGHT and wr >= MIN_P_WIDTH and ar >= MIN_P_AREA and wr <= 1.1 and hr <= 1.6:
            present[r, c] = True
            if hr < DOUBT_HEIGHT:
                unsure[r, c] = True
        elif hr >= DOT_HEIGHT and (ar >= MIN_P_AREA * .6):
            unsure[r, c] = True            # something was written but it does not look like a P
    unsure &= ~present | unsure           # (kept explicit: doubtful marks are never silently 'present')
    unsure[present & (score >= DOUBT_HEIGHT)] = False
    return ink_sum, score, present, unsure


def read_sheet(img: np.ndarray, layout: str = "auto", n_rows: int | None = None,
               name_x: tuple[float, float] | None = None) -> SheetResult:
    img = normalise_size(img)
    H, W = img.shape[:2]
    warnings: list[str] = []
    gx, contrast = _find_gutter(img)
    if layout == "auto":
        layout = "spread" if W > 1.15 * H else "single"
    if layout == "spread":
        left_crop, right_crop = img[:, :gx - 6], img[:, gx + 6:]
    else:
        left_crop, right_crop = img, None

    limg, lx0, ld = _prep_half(left_crop, N_LEFT, True, W)
    lh, lw = limg.shape[:2]
    lxs = lx0 + ld * np.arange(N_LEFT + 1)
    ytop, pitch = _row_lattice(limg, lxs[0], lxs[-1])

    nx0 = int((name_x[0] if name_x else 0.10) * lw)
    nx1 = int(lxs[0] - 2.4 * ld)
    lcl_ink, ldark, lraw = _clean_ink(limg)

    max_rows = int((lh * .97 - ytop) / pitch)
    name_ink = []
    for k in range(max_rows):
        a, b = int(ytop + (k + .12) * pitch), int(ytop + (k + .88) * pitch)
        name_ink.append(float(lraw[a:b, nx0:nx1].sum()))
    thr = 0.012 * (nx1 - nx0) * pitch * .76
    filled = [v > thr for v in name_ink]
    nr, gap = 0, 0
    for k, f in enumerate(filled):
        if f:
            nr, gap = k + 1, 0
        else:
            gap += 1
            if gap >= 2 and nr:
                break
    if n_rows:
        nr = n_rows
    if nr == 0:
        raise ValueError("No handwritten rows found")
    lys = ytop + pitch * np.arange(nr + 1)

    halves = [HalfLayout(limg, lxs, lys, ldark, lcl_ink)]
    if right_crop is not None:
        rimg, rx0, rd = _prep_half(right_crop, N_RIGHT, False, W)
        rxs = rx0 + rd * np.arange(N_RIGHT + 1)
        rytop, rpitch = _row_lattice(rimg, rxs[0], rxs[-1])
        rys = rytop + rpitch * np.arange(nr + 1)
        rcl, rdark, _ = _clean_ink(rimg)
        halves.append(HalfLayout(rimg, rxs, rys, rdark, rcl))
    ncols = N_LEFT + (N_RIGHT if right_crop is not None else 0)

    cell_ink = np.zeros((nr, 31), float)
    cell_score = np.zeros((nr, 31), float)
    present = np.zeros((nr, 31), bool)
    uncertain = np.zeros((nr, 31), bool)
    off = np.zeros(31, bool)
    ch_off = 0
    for hl in halves:
        n = len(hl.xs) - 1
        pink = _pink_fraction(hl.img)
        for c in range(n):
            xa, xb = int(hl.xs[c]), int(hl.xs[c + 1])
            ya, yb = int(hl.ys[0]), int(hl.ys[-1])
            col_pink = pink[ya:yb, xa:xb].mean() if yb > ya and xb > xa else 0.0
            off[ch_off + c] = col_pink > 0.12
        ink_sum, score, ok, unsure = _detect_marks(hl)
        cell_ink[:, ch_off:ch_off + n] = ink_sum
        cell_score[:, ch_off:ch_off + n] = score
        present[:, ch_off:ch_off + n] = ok
        uncertain[:, ch_off:ch_off + n] = unsure
        ch_off += n
    present[:, off] = False
    uncertain[:, off] = False
    present[:, ncols:] = False
    uncertain[:, ncols:] = False

    struck = np.zeros(nr, bool)
    span0, span1 = nx0, int(lxs[-1])
    for r in range(nr):
        a, b = int(lys[r] + .15 * pitch), int(lys[r + 1] - .1 * pitch)
        band = lraw[a:b, span0:span1]
        band = cv2.dilate(band, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
        run = cv2.morphologyEx(band, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(.45 * (span1 - span0)), 1)))
        struck[r] = run.any()
    crops = []
    for r in range(nr):
        a, b = int(lys[r]), int(lys[r + 1])
        crops.append(limg[max(a - 4, 0):b + 2, nx0:nx1].copy())

    dbg = _draw_debug(halves, present, off, struck, uncertain)
    return SheetResult(nr, present, off, struck, cell_ink, crops, dbg, layout, ncols, warnings,
                       uncertain, cell_score)


def _draw_debug(halves, present, off, struck, unc=None):
    tiles = []
    off_i = 0
    for hl in halves:
        ov = hl.img.copy()
        n = len(hl.xs) - 1
        nr = len(hl.ys) - 1
        for c in range(n):
            xa, xb = int(hl.xs[c]), int(hl.xs[c + 1])
            for r in range(nr):
                ya, yb = int(hl.ys[r]), int(hl.ys[r + 1])
                col = (0, 170, 0) if present[r, off_i + c] else (0, 0, 220)
                if unc is not None and unc[r, off_i + c]:
                    col = (0, 215, 255)
                if off[off_i + c]:
                    col = (160, 160, 160)
                if struck[r]:
                    col = (0, 140, 255)
                cv2.rectangle(ov, (xa + 3, ya + 3), (xb - 3, yb - 3), col, 1 if col == (0, 0, 220) else 2)
        off_i += n
        tiles.append(ov)
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    out = np.hstack(tiles)
    s = 1800 / out.shape[1]
    return cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
