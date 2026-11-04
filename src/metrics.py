"""Evaluation metrics, threshold selection and distribution-shift measures."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve)
from sklearn.model_selection import GroupKFold, StratifiedKFold


def fbeta_threshold(y, p, beta=2.0):
    """Threshold maximising F-beta on the given (validation) data.

    beta=2 weighs recall four times as much as precision, which matches the clinical
    priority of not missing events while still penalising a detector that fires on
    everything.
    """
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f = (1 + beta ** 2) * prec * rec / np.maximum(beta ** 2 * prec + rec, 1e-12)
    return float(thr[np.nanargmax(f)])


def recall_at_fpr(y, p, fpr_target=0.10):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(fpr_target, fpr, tpr))


def binary_metrics(y, p, thr):
    y = np.asarray(y).astype(int)
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    f2 = 5 * prec * rec / max(4 * prec + rec, 1e-12)
    both = len(np.unique(y)) == 2
    return {
        "auroc": roc_auc_score(y, p) if both else np.nan,
        "auprc": average_precision_score(y, p) if both else np.nan,
        "recall": rec, "specificity": spec, "fpr": 1 - spec, "precision": prec,
        "f1": f1, "f2": f2, "balanced_acc": 0.5 * (rec + spec),
        "accuracy": (tp + tn) / len(y),
        "recall_at_10fpr": recall_at_fpr(y, p, 0.10) if both else np.nan,
        "threshold": thr, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "prevalence": float(y.mean()),
    }


def bootstrap_auroc(y, p, groups=None, n_boot=1000, seed=0):
    """95% CI for AUROC. If groups are given, subjects are resampled (cluster
    bootstrap), since windows from one subject are far from independent."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    if groups is None:
        n = len(y)
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            if len(np.unique(y[i])) == 2:
                vals.append(roc_auc_score(y[i], p[i]))
    else:
        groups = np.asarray(groups)
        ug = np.unique(groups)
        idx_by_g = {g: np.where(groups == g)[0] for g in ug}
        for _ in range(n_boot):
            gs = rng.choice(ug, len(ug), replace=True)
            i = np.concatenate([idx_by_g[g] for g in gs])
            if len(np.unique(y[i])) == 2:
                vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mmd_rbf(A, B, n=2000, seed=0):
    """Unbiased squared MMD with an RBF kernel (median heuristic bandwidth)."""
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    Z = np.vstack([A, B])
    sq = (Z ** 2).sum(1)
    D = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
    D = np.maximum(D, 0)
    sigma2 = np.median(D[D > 0])
    K = np.exp(-D / sigma2)
    m = len(A)
    Kaa, Kbb, Kab = K[:m, :m], K[m:, m:], K[:m, m:]
    np.fill_diagonal(Kaa, 0)
    np.fill_diagonal(Kbb, 0)
    nb = len(B)
    return float(Kaa.sum() / (m * (m - 1)) + Kbb.sum() / (nb * (nb - 1)) - 2 * Kab.mean())


def domain_classifier_auc(A, B, seed=0, n=4000):
    """How well a linear probe separates the two domains (5-fold AUROC).
    0.5 means the representations are indistinguishable."""
    rng = np.random.default_rng(seed)
    A = A[rng.choice(len(A), min(n, len(A)), replace=False)]
    B = B[rng.choice(len(B), min(n, len(B)), replace=False)]
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = (X - mu) / sd
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))
