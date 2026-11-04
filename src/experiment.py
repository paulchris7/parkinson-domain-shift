"""Small helpers that load the processed data the same way in every notebook."""
import json

import numpy as np
import pandas as pd
import torch

from . import config as C
from .transforms import RobustScaler, scale_by_group


def load_scaler(name):
    s = json.load(open(C.DATA_PROCESSED / "scalers.json"))[name]
    sc = RobustScaler()
    sc.center_ = np.array(s["center"], dtype=np.float32)
    sc.scale_ = np.array(s["scale"], dtype=np.float32)
    return sc


def load_all():
    torch.set_num_threads(2)
    X_sis = np.load(C.DATA_PROCESSED / "sisfall_X_g.npy")
    X_dap = np.load(C.DATA_PROCESSED / "daphnet_X_g.npy")
    meta_sis = pd.read_csv(C.DATA_PROCESSED / "sisfall_meta.csv")
    meta_dap = pd.read_csv(C.DATA_PROCESSED / "daphnet_meta.csv")
    return X_sis, meta_sis, X_dap, meta_dap


def sisfall_splits(X_sis, meta_sis, src_scaler):
    out = {}
    for split in ("train", "val", "test"):
        m = (meta_sis.split == split).values
        out[split] = {
            "X_g": X_sis[m],
            "X": src_scaler.transform(X_sis[m]),
            "y": meta_sis.label.values[m].astype(int),
            "meta": meta_sis[m].reset_index(drop=True),
        }
    return out


def daphnet_views(X_dap, meta_dap, src_scaler, tgt_scaler):
    """Three ways to normalise the target domain, from naive to subject-adaptive."""
    return {
        "source statistics": src_scaler.transform(X_dap),
        "domain statistics": tgt_scaler.transform(X_dap),
        "subject statistics": scale_by_group(X_dap, meta_dap.subject.values),
    }
