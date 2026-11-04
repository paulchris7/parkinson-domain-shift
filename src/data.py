"""Raw-file parsing and signal preprocessing for SisFall and Daphnet."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from . import config as C

SISFALL_NAME = re.compile(r"^(?P<act>[DF]\d{2})_(?P<subj>S[AE]\d{2})_(?P<trial>R\d{2})\.txt$")


# --------------------------------------------------------------------------- #
# SisFall
# --------------------------------------------------------------------------- #
def parse_sisfall_file(path):
    """Return an (n, 9) float array of raw counts. Malformed rows become NaN.

    A handful of SisFall files have truncated last lines or stray characters,
    so the file is parsed line by line instead of trusting np.loadtxt.
    """
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            vals = np.full(9, np.nan)
            for k, p in enumerate(parts[:9]):
                try:
                    vals[k] = float(p)
                except ValueError:
                    pass
            rows.append(vals)
    return np.asarray(rows, dtype=np.float64)


def index_sisfall(root):
    """List every trial file with its parsed metadata."""
    recs = []
    for p in sorted(Path(root).rglob("*.txt")):
        m = SISFALL_NAME.match(p.name)
        if m is None:
            continue
        d = m.groupdict()
        recs.append({
            "path": str(p), "file": p.name, "activity": d["act"], "subject": d["subj"],
            "trial": d["trial"], "group": "Elderly" if d["subj"].startswith("SE") else "Young",
            "is_fall": d["act"].startswith("F"),
        })
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Daphnet
# --------------------------------------------------------------------------- #
DAPHNET_COLS = ["t_ms",
                "ankle_fwd", "ankle_vert", "ankle_lat",
                "thigh_fwd", "thigh_vert", "thigh_lat",
                "trunk_fwd", "trunk_vert", "trunk_lat",
                "annotation"]


def parse_daphnet_file(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=DAPHNET_COLS, engine="python")
    return df


def index_daphnet(root):
    recs = []
    for p in sorted(Path(root).rglob("S*R*.txt")):
        m = re.match(r"^(S\d{2})(R\d{2})\.txt$", p.name)
        if m is None:
            continue
        recs.append({"path": str(p), "file": p.name, "subject": m.group(1), "run": m.group(2)})
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# Generic signal operations
# --------------------------------------------------------------------------- #
def fill_missing(x):
    """Linear interpolation inside the signal, nearest value at the edges."""
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.shape[0]
    idx = np.arange(n)
    for c in range(x.shape[1]):
        col = x[:, c]
        bad = ~np.isfinite(col)
        if bad.all():
            return None
        if bad.any():
            col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
    return x


def resample_to(x, fs_in, fs_out=C.FS):
    """Polyphase resampling; the built-in FIR filter handles anti-aliasing."""
    if fs_in == fs_out:
        return x
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    up, down = int(fs_out) // g, int(fs_in) // g
    return signal.resample_poly(x, up, down, axis=0, padtype="line")


def to_frame(acc, mapping):
    """Reorder/flip a 3-axis accelerometer into [V, AP, ML] and append |a|.

    mapping: dict with keys V, AP, ML -> (column index, sign)
    """
    out = np.stack([mapping[k][1] * acc[:, mapping[k][0]] for k in ("V", "AP", "ML")], axis=1)
    mag = np.linalg.norm(out, axis=1, keepdims=True)
    return np.concatenate([out, mag], axis=1).astype(np.float32)


def sliding_windows(n, win=C.WIN, stride=C.FINE_STRIDE):
    if n < win:
        return np.empty(0, dtype=int)
    return np.arange(0, n - win + 1, stride)


def fall_window_labels(starts, peak_idx, fs=C.FS, win=C.WIN):
    """+1 fall, 0 non-fall, -1 ambiguous (dropped).

    The event is the interval peak +/- FALL_HALF_EVENT seconds. A window is a fall
    window when it overlaps that interval by at least FALL_MIN_OVERLAP seconds,
    which keeps the impact roughly centred. Windows touching the event only
    partially are ambiguous and left out rather than forced into either class.
    """
    ev0 = peak_idx - C.FALL_HALF_EVENT * fs
    ev1 = peak_idx + C.FALL_HALF_EVENT * fs
    w0 = starts.astype(float)
    w1 = w0 + win
    ov = np.clip(np.minimum(w1, ev1) - np.maximum(w0, ev0), 0, None) / fs
    lab = np.full(len(starts), -1, dtype=int)
    lab[ov >= C.FALL_MIN_OVERLAP] = 1
    lab[ov == 0] = 0
    return lab


def contiguous_segments(mask):
    """(start, stop) pairs of True runs in a boolean array."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
