"""Training and inference loops shared by every architecture."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from .models import build_model
from .transforms import augment_batch

# Training runs on the GPU when one is visible (e.g. Colab); everything else is unchanged.
# Random draws (shuffling, augmentation) always come from a CPU generator, so the batch
# sequence for a given seed does not depend on the device.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_TRAIN = dict(
    epochs=30, batch_size=256, lr=1e-3, weight_decay=1e-4, patience=6,
    warmup_epochs=1, grad_clip=1.0, augment=True, select_metric="val_loss",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _lr_lambda(total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
    return f


@torch.no_grad()
def predict_proba(model, X, batch_size=2048):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(dev)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


@torch.no_grad()
def extract_embeddings(model, X, batch_size=2048):
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size])).to(dev)
        out.append(model.embed(xb).cpu().numpy())
    return np.concatenate(out)


def _score(y, p, metric):
    if len(np.unique(y)) < 2:
        return float("nan")
    return average_precision_score(y, p) if metric == "auprc" else roc_auc_score(y, p)


def train_model(arch, hp, X_train_g, y_train, X_val_scaled, y_val, scaler, cfg=None,
                seed=0, init_state=None, verbose=True, log_fn=None):
    """Train one model.

    X_train_g is kept in g (unscaled) because augmentation has to happen in physical
    units; it is scaled on the fly with `scaler`. Validation data arrives pre-scaled.
    Shuffling and augmentation draw from one seeded generator, so for a given seed
    every architecture sees exactly the same sequence of mini-batches.
    """
    cfg = {**DEFAULT_TRAIN, **(cfg or {})}
    set_seed(seed)
    model = build_model(arch, hp)
    if init_state is not None:
        model.load_state_dict(init_state)
    model.to(DEVICE)

    gen = torch.Generator().manual_seed(seed)
    center = torch.tensor(scaler.center_)
    scale = torch.tensor(scaler.scale_)
    Xt = torch.from_numpy(np.ascontiguousarray(X_train_g, dtype=np.float32))
    yt = torch.from_numpy(y_train.astype(np.float32))

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = int(np.ceil(len(Xt) / cfg["batch_size"]))
    total = steps_per_epoch * cfg["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(total, steps_per_epoch * cfg["warmup_epochs"]))

    best, best_state, best_epoch, wait = -np.inf, None, -1, 0
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Xt), generator=gen)
        run_loss = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * cfg["batch_size"]:(b + 1) * cfg["batch_size"]]
            xb, yb = Xt[idx], yt[idx]
            if cfg["augment"]:
                xb = augment_batch(xb, center, scale, generator=gen)
            else:
                xb = (xb - center) / scale
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            loss = criterion(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            run_loss += loss.item() * len(idx)

        p_val = predict_proba(model, X_val_scaled)
        val_loss = nn.functional.binary_cross_entropy(
            torch.from_numpy(np.clip(p_val, 1e-7, 1 - 1e-7)),
            torch.from_numpy(y_val.astype(np.float32))).item()
        # Higher is better for every criterion; validation loss is negated. On SisFall the
        # validation AUPRC saturates at 1.0 within a couple of epochs, so the (unweighted)
        # cross-entropy is the only criterion that keeps ranking checkpoints.
        score = -val_loss if cfg["select_metric"] == "val_loss" else _score(y_val, p_val, cfg["select_metric"])
        rec = dict(epoch=epoch + 1, train_loss=run_loss / len(Xt), val_loss=val_loss,
                   val_auprc=_score(y_val, p_val, "auprc"), val_auroc=_score(y_val, p_val, "auroc"),
                   seconds=time.time() - t0)
        history.append(rec)
        msg = (f"[{arch} s{seed}] ep {epoch + 1:02d} loss {rec['train_loss']:.4f} "
               f"val_auprc {rec['val_auprc']:.4f} val_auroc {rec['val_auroc']:.4f} ({rec['seconds']:.0f}s)")
        if verbose:
            print(msg, flush=True)
        if log_fn:
            log_fn(msg)

        if score > best + 1e-4:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best, best_epoch, wait = score, epoch + 1, 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "best_val_score": best}
