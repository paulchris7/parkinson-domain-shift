"""The four architectures compared in the study.

All models take (B, T, C) windows and return one logit per window. `embed` returns
the representation just before the classifier, which is what the shift analysis uses.
"""
import math

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    def __init__(self, in_ch=4, channels=(32, 64, 128), kernel=7, dropout=0.3):
        super().__init__()
        layers, c_prev = [], in_ch
        for i, c in enumerate(channels):
            k = max(3, kernel - 2 * i)  # wide first kernel, narrower deeper ones
            layers += [nn.Conv1d(c_prev, c, k, padding=k // 2, bias=False),
                       nn.BatchNorm1d(c), nn.ReLU(inplace=True)]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool1d(2))
            c_prev = c
        self.features = nn.Sequential(*layers)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(c_prev, 1)
        self.emb_dim = c_prev

    def embed(self, x):
        h = self.features(x.transpose(1, 2))
        # global average pooling: the impact can sit anywhere in the window
        return h.mean(dim=-1)

    def forward(self, x):
        return self.head(self.drop(self.embed(x))).squeeze(-1)


class RNNClassifier(nn.Module):
    def __init__(self, cell="lstm", in_ch=4, hidden=64, layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(in_ch, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        d = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * d, 1)
        self.emb_dim = 2 * d

    def embed(self, x):
        out, _ = self.rnn(x)
        # last step keeps the recurrent summary; the temporal mean guards against
        # the event being early in the window and partly forgotten
        return torch.cat([out[:, -1], out.mean(dim=1)], dim=-1)

    def forward(self, x):
        return self.head(self.drop(self.embed(x))).squeeze(-1)


class PatchTransformer(nn.Module):
    """Encoder-only Transformer over non-overlapping temporal patches (PatchTST-style
    tokenisation, channel-mixing embedding) with a learnable [CLS] token."""

    def __init__(self, in_ch=4, seq_len=128, patch=16, d_model=64, heads=4, layers=3,
                 ff_mult=2, dropout=0.1):
        super().__init__()
        assert seq_len % patch == 0
        self.patch = patch
        self.n_tokens = seq_len // patch
        self.proj = nn.Linear(patch * in_ch, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens + 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * ff_mult, dropout,
                                         batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)
        self.emb_dim = d_model
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def embed(self, x):
        B, T, C = x.shape
        tok = x.reshape(B, self.n_tokens, self.patch * C)
        tok = self.proj(tok)
        tok = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1) + self.pos
        h = self.encoder(tok)
        return self.norm(h[:, 0])

    def forward(self, x):
        return self.head(self.drop(self.embed(x))).squeeze(-1)


def build_model(arch, hp=None, in_ch=4, seq_len=128):
    hp = dict(hp or {})
    if arch == "CNN-1D":
        ch = hp.get("width", 32)
        return CNN1D(in_ch, channels=(ch, 2 * ch, 4 * ch), kernel=hp.get("kernel", 7),
                     dropout=hp.get("dropout", 0.3))
    if arch in ("LSTM", "GRU"):
        return RNNClassifier(arch.lower(), in_ch, hidden=hp.get("hidden", 64),
                             layers=hp.get("layers", 2), dropout=hp.get("dropout", 0.3),
                             bidirectional=hp.get("bidirectional", False))
    if arch == "Transformer":
        return PatchTransformer(in_ch, seq_len, patch=hp.get("patch", 16),
                                d_model=hp.get("d_model", 64), heads=hp.get("heads", 4),
                                layers=hp.get("layers", 3), dropout=hp.get("dropout", 0.1))
    raise ValueError(arch)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
