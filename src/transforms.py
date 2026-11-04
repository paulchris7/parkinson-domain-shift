"""Scaling and augmentation. Everything operates on windows shaped (N, T, 4) in g."""
import numpy as np
import torch


class RobustScaler:
    """Per-channel (x - median) / IQR.

    Fall impacts reach 10+ g while the bulk of the data sits near 1 g, so mean/std
    would be dominated by a few windows. Median and IQR are not.
    """

    def __init__(self):
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        q25, q50, q75 = np.percentile(flat, [25, 50, 75], axis=0)
        self.center_ = q50.astype(np.float32)
        self.scale_ = np.maximum(q75 - q25, 1e-3).astype(np.float32)
        return self

    def transform(self, X):
        return ((X - self.center_) / self.scale_).astype(np.float32)

    def state(self):
        return {"center": self.center_.tolist(), "scale": self.scale_.tolist()}


def scale_by_group(X, groups):
    """Fit a separate RobustScaler on each group (e.g. each subject) and apply it."""
    out = np.empty_like(X, dtype=np.float32)
    for g in np.unique(groups):
        m = groups == g
        out[m] = RobustScaler().fit(X[m]).transform(X[m])
    return out


def augment_batch(x, scaler_center, scaler_scale, generator=None,
                  max_yaw_deg=20.0, scale_range=(0.9, 1.1), jitter_g=0.01, p_flip_ml=0.5):
    """Physically plausible augmentation applied in g before scaling.

    x: (B, T, 4) tensor in g. Returns a scaled tensor ready for the network.
    - rotation about the vertical axis: sensor yaw on the belt is never exact;
    - ML sign flip: left/right mounting is not standardised across datasets;
    - amplitude scaling and small jitter: inter-sensor gain and noise differences.
    The magnitude channel is recomputed so the four channels stay consistent.
    """
    B = x.shape[0]
    dev = x.device
    g = generator
    v, ap, ml = x[..., 0], x[..., 1], x[..., 2]
    theta = (torch.rand(B, 1, device=dev, generator=g) * 2 - 1) * np.deg2rad(max_yaw_deg)
    c, s = torch.cos(theta), torch.sin(theta)
    ap2 = c * ap - s * ml
    ml2 = s * ap + c * ml
    flip = (torch.rand(B, 1, device=dev, generator=g) < p_flip_ml).float() * -2 + 1
    ml2 = ml2 * flip
    sc = torch.empty(B, 1, device=dev).uniform_(*scale_range, generator=g)
    xyz = torch.stack([v, ap2, ml2], dim=-1) * sc.unsqueeze(-1)
    xyz = xyz + torch.randn(xyz.shape, device=dev, generator=g) * jitter_g
    mag = xyz.norm(dim=-1, keepdim=True)
    out = torch.cat([xyz, mag], dim=-1)
    return (out - scaler_center) / scaler_scale
