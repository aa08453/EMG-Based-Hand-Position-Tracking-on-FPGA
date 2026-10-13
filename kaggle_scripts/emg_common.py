"""
emg_common.py
=============
Shared library for the Kaggle notebooks in this folder (NinaPro DB2, sEMG -> finger joint angles,
RoFormer-style model targeted at a Zynq-7020 HLS implementation).

Attach this file to a Kaggle notebook as a *Utility Script* (File > Add utility script) or upload it
as a dataset; the notebooks search both locations.

Sections
  1. Config
  2. Signal processing and dataset preparation (leakage-free split, causal normalisation)
  3. Datasets (windows, consecutive-window sequences, ordered streams)
  4. Models (RoFormer variants, dual absolute/delta heads, previous-angle conditioning)
  5. Fixed-point fake quantisation (QAT matching ap_fixed<W,I>)
  6. Training and window-level evaluation
  7. Sequential evaluation: delta integration, Kalman fusion, rest anchoring
  8. Numpy reference forward pass (float and fixed-point) and HLS export
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, sosfilt, sosfiltfilt
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

# =====================================================================================
# 1. CONFIG
# =====================================================================================

JOINT_NAMES = ['T_CMC_ABD', 'T_MCP_FLEX', 'WRIST_ABD', 'I_MCP_FLEX', 'I_MCP_ABD',
               'M_MCP_FLEX', 'R_MCP_FLEX', 'R_PIP_FLEX', 'P_MCP_FLEX', 'P_PIP_FLEX']


@dataclass
class Config:
    # ---- paths (Kaggle defaults) ----
    base_path: str = '/kaggle/input/datasets/quddusikashaf/ninapro-db2'
    processed_dir: str = '/kaggle/working/db2_processed'
    save_dir: str = '/kaggle/working/outputs'
    export_dir: str = '/kaggle/working/hls_export'

    # ---- data ----
    subjects: Optional[List[str]] = None     # None -> first n_subjects found under base_path
    n_subjects: int = 3
    exercises: Sequence[int] = (1, 2)        # DB2 files E1, E2 (E3 has force data)
    fs: int = 2000
    seq_len: int = 200                       # window samples (100 ms). Must equal HLS SEQ_LEN
    stride: int = 50                         # 25 ms hop. Deltas are per stride.
    num_channels: int = 12
    num_angles: int = 10
    glove_cols: Sequence[int] = (1, 2, 4, 5, 7, 8, 11, 12, 15, 16)
    train_reps: Sequence[int] = (1, 3, 4, 6)  # NinaPro convention
    test_reps: Sequence[int] = (2, 5)
    bandpass: Tuple[float, float] = (5.0, 500.0)
    filter_order: int = 4
    causal_filter: bool = True               # True = what the firmware can actually do
    mu: int = 255
    emg_scale_percentile: float = 99.9       # per-channel scale from TRAIN samples only
    balance_rest: bool = True

    # ---- model ----
    variant: str = 'roformer'                # roformer | patch | noattn | causal
    d_model: int = 128
    d_ff: int = 256
    num_heads: int = 8
    num_layers: int = 1
    conv_kernel: int = 3
    conv_stride: int = 1
    conv_padding: int = 1
    patch_stride: int = 8                    # used by patch/noattn/causal presets -> 25 tokens
    norm: str = 'layernorm'                  # layernorm | rmsnorm
    use_attention: bool = True
    causal: bool = False
    pooling: str = 'mean'                    # mean | last
    dropout: float = 0.1
    heads: str = 'both'                      # absolute | delta | both
    cond_prev: bool = True                   # feed previous angle estimate into the head

    # ---- training ----
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 40
    patience: int = 8
    seed: int = 42
    lambda_abs: float = 1.0
    lambda_delta: float = 1.0
    loss_norm: bool = True                   # divide the absolute error by the per-joint train variance so that
                                             # abs and (standardised) delta terms are both on a 1-R^2 scale
    select_metric: str = 'val_abs'           # checkpoint selection / early stop: val_abs | val_delta | val_loss
    delta_detach: bool = False               # train the delta head on detached trunk features (linear probe)
    lambda_traj: float = 1.0                 # loss on integrated trajectory (unrolled mode)
    unroll_steps: int = 0                    # 0 = plain windows; K>1 = integration in the loop
    feedback_noise_std: float = 0.03         # noise on fed-back estimate during training
    fuse_alpha: float = 0.1                  # training-time complementary-filter gain
    num_workers: int = 2

    # ---- quantisation (ap_fixed<W,I>) ----
    qat_epochs: int = 0                      # 0 disables QAT fine-tuning
    qat_lr: float = 5e-5
    act_fmt: Tuple[int, int] = (10, 3)       # current HLS data_t (norm outputs, pooled feature, head outputs, linear inputs)
    resid_fmt: Tuple[int, int] = (16, 8)     # residual stream: branch outputs and pre-norm sums (FFN outputs reach |100|+)
    weight_fmt: Tuple[int, int] = (10, 3)
    rope_fmt: Tuple[int, int] = (8, 2)
    export_weight_type: str = 'data_t'       # change to 'weight_t' after adding a typedef in types.h

    # ---- logging ----
    project_name: str = 'ninapro_db2_roformer'
    run_name: str = 'run'
    use_wandb: bool = True

    def apply_variant(self):
        v = self.variant
        if v == 'roformer':
            self.conv_kernel, self.conv_stride, self.conv_padding = 3, 1, 1
            self.use_attention, self.causal = True, False
        elif v == 'patch':
            self.conv_kernel = self.conv_stride = self.patch_stride
            self.conv_padding = 0
            self.use_attention, self.causal = True, False
        elif v == 'noattn':
            self.conv_kernel = self.conv_stride = self.patch_stride
            self.conv_padding = 0
            self.use_attention = False
        elif v == 'causal':
            self.conv_kernel = self.conv_stride = self.patch_stride
            self.conv_padding = 0
            self.use_attention, self.causal = True, True
            self.norm = 'rmsnorm'
            self.pooling = 'last'
        else:
            raise ValueError(f'unknown variant {v}')
        return self

    @property
    def num_tokens(self) -> int:
        return (self.seq_len + 2 * self.conv_padding - self.conv_kernel) // self.conv_stride + 1

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.num_heads == 0
        return self.d_model // self.num_heads

    def to_dict(self):
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d

    @classmethod
    def from_dict(cls, d):
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, tuple(v) if isinstance(getattr(c, k), tuple) else v)
        return c


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_wandb(cfg: Config, name: Optional[str] = None):
    """W&B via Kaggle Secrets (secret name: wandb_key) or WANDB_API_KEY. Never hard-code keys."""
    if not cfg.use_wandb:
        return None
    try:
        import wandb
        key = os.environ.get('WANDB_API_KEY')
        if key is None:
            try:
                from kaggle_secrets import UserSecretsClient
                key = UserSecretsClient().get_secret('wandb_key')
            except Exception:
                key = None
        if key is None:
            print('[wandb] no key found, logging disabled')
            return None
        wandb.login(key=key)
        return wandb.init(project=cfg.project_name, name=name or cfg.run_name,
                          config=cfg.to_dict(), reinit='finish_previous')
    except Exception as e:  # pragma: no cover
        print(f'[wandb] disabled: {e}')
        return None


def wandb_log(run, d: dict):
    if run is not None:
        run.log(d)


# =====================================================================================
# 2. SIGNAL PROCESSING AND DATASET PREPARATION
# =====================================================================================

def list_subjects(cfg: Config) -> List[str]:
    if cfg.subjects:
        return list(cfg.subjects)
    subs = sorted(f for f in os.listdir(cfg.base_path) if f.startswith('DB2'))
    return subs[:cfg.n_subjects]


def bandpass_sos(cfg: Config):
    nyq = 0.5 * cfg.fs
    return butter(cfg.filter_order, [cfg.bandpass[0] / nyq, cfg.bandpass[1] / nyq],
                  btype='band', output='sos')


def filter_emg(emg: np.ndarray, cfg: Config) -> np.ndarray:
    """Bandpass then rectify. causal_filter=True uses a one-directional IIR (deployable)."""
    sos = bandpass_sos(cfg)
    y = sosfilt(sos, emg, axis=0) if cfg.causal_filter else sosfiltfilt(sos, emg, axis=0)
    return np.abs(y).astype(np.float32)


def mu_law(x: np.ndarray, mu: int) -> np.ndarray:
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def load_subject_raw(cfg: Config, subject: str):
    """Returns per-file lists of (emg, glove, restimulus, rerepetition)."""
    paths = sorted(glob.glob(f'{cfg.base_path}/{subject}/{subject}/*.mat'))
    if not paths:
        paths = sorted(glob.glob(f'{cfg.base_path}/{subject}/*.mat'))
    if not paths:
        raise FileNotFoundError(f'no .mat files for {subject} under {cfg.base_path}')
    out = []
    for p in paths:
        ex = int(os.path.basename(p).split('_E')[1][0]) if '_E' in os.path.basename(p) else None
        if ex is not None and ex not in cfg.exercises:
            continue
        d = scio.loadmat(p)
        rerep = d['rerepetition'] if 'rerepetition' in d else d['repetition']
        out.append((d['emg'].astype(np.float32), d['glove'].astype(np.float32),
                    d['restimulus'].astype(np.int32).squeeze(), rerep.astype(np.int32).squeeze()))
    return out


def fill_rep_labels(rep: np.ndarray) -> np.ndarray:
    """Rest samples carry rerepetition 0. Forward-fill from the preceding movement so rest
    windows follow the repetition they belong to (then back-fill the leading rest)."""
    rep = rep.copy()
    last = 0
    for i in range(len(rep)):
        if rep[i] == 0:
            rep[i] = last
        else:
            last = rep[i]
    first_nz = rep[rep > 0]
    if len(first_nz):
        rep[rep == 0] = first_nz[0]
    return rep


def compute_norm_stats(rect_emg_train: np.ndarray, glove_train: np.ndarray, cfg: Config) -> dict:
    scale = np.percentile(rect_emg_train, cfg.emg_scale_percentile, axis=0) + 1e-8
    gmin = glove_train.min(axis=0)
    gmax = glove_train.max(axis=0)
    return {'emg_scale': scale.astype(np.float32), 'glove_min': gmin.astype(np.float32),
            'glove_max': gmax.astype(np.float32), 'mu': cfg.mu}


def normalise_emg(rect_emg: np.ndarray, stats: dict, cfg: Config) -> np.ndarray:
    x = np.clip(rect_emg / stats['emg_scale'], 0.0, 1.0)
    return mu_law(x, cfg.mu).astype(np.float32)


def normalise_glove(glove: np.ndarray, stats: dict) -> np.ndarray:
    return ((glove - stats['glove_min']) / (stats['glove_max'] - stats['glove_min'] + 1e-8)).astype(np.float32)


def process_subject(cfg: Config, subject: str, verbose: bool = True) -> str:
    """Builds <processed_dir>/<subject>/data.npz with a repetition-based split and
    normalisation constants computed from the training repetitions only."""
    files = load_subject_raw(cfg, subject)
    cols = list(cfg.glove_cols)

    # pass 1: filter, collect train-sample statistics
    filt, train_masks = [], []
    for emg, glove, restim, rerep in files:
        rect = filter_emg(emg, cfg)
        rep = fill_rep_labels(rerep)
        is_train = np.isin(rep, cfg.train_reps)
        filt.append((rect, glove[:, cols], restim, rep))
        train_masks.append(is_train)
    rect_train = np.concatenate([f[0][m] for f, m in zip(filt, train_masks)])
    glove_train = np.concatenate([f[1][m] for f, m in zip(filt, train_masks)])
    stats = compute_norm_stats(rect_train, glove_train, cfg)

    # pass 2: normalise and window per file (segment)
    X, A, Aprev, Dstride, Dwin, R, REP, SEG, T, SPLIT, HASPREV = ([] for _ in range(11))
    for seg_id, (rect, glove, restim, rep) in enumerate(filt):
        emg_n = normalise_emg(rect, stats, cfg)
        glove_n = normalise_glove(glove, stats)
        is_test = np.isin(rep, cfg.test_reps)
        n = len(emg_n)
        L, S = cfg.seq_len, cfg.stride
        prev_angle = None
        for i in range(0, n - L + 1, S):
            last = i + L - 1
            n_test = int(is_test[i:i + L].sum())
            if n_test not in (0, L):
                prev_angle = None          # window straddles train/test boundary: drop it
                continue
            a = glove_n[last]
            if prev_angle is None:
                ap, hp = a.copy(), 0
            else:
                ap, hp = prev_angle, 1
            X.append(emg_n[i:i + L]); A.append(a); Aprev.append(ap)
            Dstride.append(a - ap); Dwin.append(a - glove_n[i]); R.append(restim[last])
            REP.append(rep[last]); SEG.append(seg_id); T.append(i); SPLIT.append(1 if n_test == L else 0)
            HASPREV.append(hp)
            prev_angle = a

    out_dir = os.path.join(cfg.processed_dir, subject)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'data.npz')
    np.savez_compressed(path, sEMG=np.asarray(X, np.float32), angles=np.asarray(A, np.float32),
                        angles_prev=np.asarray(Aprev, np.float32), delta_stride=np.asarray(Dstride, np.float32),
                        delta_window=np.asarray(Dwin, np.float32), restim=np.asarray(R, np.int32),
                        rep=np.asarray(REP, np.int32), segment=np.asarray(SEG, np.int32),
                        t_index=np.asarray(T, np.int64), split=np.asarray(SPLIT, np.int8),
                        has_prev=np.asarray(HASPREV, np.int8))
    with open(os.path.join(out_dir, 'norm_stats.json'), 'w') as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in stats.items()}, f, indent=2)
    if verbose:
        sp = np.asarray(SPLIT)
        print(f'{subject}: {len(X)} windows (train {int((sp == 0).sum())}, test {int((sp == 1).sum())}) -> {path}')
    return path


def load_norm_stats(npz_path: str) -> dict:
    with open(os.path.join(os.path.dirname(npz_path), 'norm_stats.json')) as f:
        d = json.load(f)
    return {k: (np.asarray(v, np.float32) if isinstance(v, list) else v) for k, v in d.items()}


def delta_scale_from_train(npz_files: Sequence[str]) -> np.ndarray:
    """Per-joint std of the per-stride delta over training windows (used to standardise the delta head)."""
    ds = []
    for f in npz_files:
        d = np.load(f)
        m = (d['split'] == 0) & (d['has_prev'] == 1)
        ds.append(d['delta_stride'][m])
    return np.concatenate(ds).std(axis=0).astype(np.float32) + 1e-6


def angle_var_from_train(npz_files: Sequence[str]) -> np.ndarray:
    """Per-joint variance of the absolute angle over training windows (normalises the absolute loss term)."""
    a = np.concatenate([np.load(f)['angles'][np.load(f)['split'] == 0] for f in npz_files])
    return a.var(axis=0).astype(np.float32) + 1e-6


def rest_pose_from_train(npz_files: Sequence[str]) -> np.ndarray:
    poses = []
    for f in npz_files:
        d = np.load(f)
        m = (d['split'] == 0) & (d['restim'] == 0)
        poses.append(d['angles'][m])
    return np.concatenate(poses).mean(axis=0).astype(np.float32)


# =====================================================================================
# 3. DATASETS
# =====================================================================================

class WindowDataset(Dataset):
    """Independent windows. Rest windows are capped (train only) to the median movement-class count."""

    def __init__(self, npz_files: Sequence[str], split: str, cfg: Config, balance_rest: Optional[bool] = None):
        want = 0 if split == 'train' else 1
        balance = cfg.balance_rest if balance_rest is None else balance_rest
        rng = np.random.RandomState(cfg.seed)
        xs, a, ap, d, hp = [], [], [], [], []
        for f in npz_files:
            dd = np.load(f)
            idx = np.where(dd['split'] == want)[0]
            if balance and split == 'train':
                restim = dd['restim'][idx]
                counts = [int((restim == c).sum()) for c in np.unique(restim) if c != 0]
                cap = int(np.median(counts)) if counts else len(idx)
                rest = idx[restim == 0]
                if len(rest) > cap:
                    keep = np.sort(rng.choice(rest, cap, replace=False))
                    idx = np.sort(np.concatenate([idx[restim != 0], keep]))
            xs.append(dd['sEMG'][idx]); a.append(dd['angles'][idx]); ap.append(dd['angles_prev'][idx])
            d.append(dd['delta_stride'][idx]); hp.append(dd['has_prev'][idx])
        self.x = np.concatenate(xs); self.angles = np.concatenate(a); self.angles_prev = np.concatenate(ap)
        self.delta = np.concatenate(d); self.has_prev = np.concatenate(hp)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {'x': torch.from_numpy(np.ascontiguousarray(self.x[i].T)),   # [C, T]
                'angle': torch.from_numpy(self.angles[i]),
                'angle_prev': torch.from_numpy(self.angles_prev[i]),
                'delta': torch.from_numpy(self.delta[i]),
                'has_prev': torch.tensor(float(self.has_prev[i]))}


class SequenceDataset(Dataset):
    """K consecutive windows (same segment, consecutive strides, same split) for integration-in-the-loop."""

    def __init__(self, npz_files: Sequence[str], split: str, cfg: Config, K: int):
        want = 0 if split == 'train' else 1
        self.K = K
        self.data, self.starts = [], []
        for fi, f in enumerate(npz_files):
            dd = np.load(f)
            x, a, ap, d = dd['sEMG'], dd['angles'], dd['angles_prev'], dd['delta_stride']
            seg, t, sp, hp = dd['segment'], dd['t_index'], dd['split'], dd['has_prev']
            self.data.append((x, a, ap, d))
            n = len(x)
            for i in range(n - K + 1):
                j = i + K - 1
                if seg[i] != seg[j] or (t[j] - t[i]) != cfg.stride * (K - 1):
                    continue
                if (sp[i:j + 1] != want).any() or (hp[i + 1:j + 1] == 0).any():
                    continue
                self.starts.append((fi, i))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, k):
        fi, i = self.starts[k]
        x, a, ap, d = self.data[fi]
        sl = slice(i, i + self.K)
        return {'x': torch.from_numpy(np.ascontiguousarray(x[sl].transpose(0, 2, 1))),  # [K, C, T]
                'angle': torch.from_numpy(a[sl]), 'angle_prev': torch.from_numpy(ap[sl]),
                'delta': torch.from_numpy(d[sl])}


def ordered_runs(npz_file: str, split: str) -> List[dict]:
    """Time-ordered contiguous runs of windows for sequential evaluation (no shuffling, no balancing)."""
    want = 0 if split == 'train' else 1
    dd = np.load(npz_file)
    idx = np.where(dd['split'] == want)[0]
    runs, cur = [], [idx[0]] if len(idx) else []
    for p, q in zip(idx[:-1], idx[1:]):
        if dd['segment'][p] == dd['segment'][q] and dd['has_prev'][q] == 1 and q == p + 1:
            cur.append(q)
        else:
            runs.append(cur); cur = [q]
    if cur:
        runs.append(cur)
    out = []
    for r in runs:
        r = np.asarray(r)
        out.append({k: dd[k][r] for k in ['sEMG', 'angles', 'angles_prev', 'delta_stride', 'restim', 'rep', 't_index']})
    return out


# =====================================================================================
# 4. MODELS
# =====================================================================================

class RotaryEmbedding(nn.Module):
    """cos/sin tables for RoPE applied on the full d_model vector (matches the HLS, which rotates
    before the Q/K projections). Half-dim symmetry: emb = cat(freqs, freqs)."""

    def __init__(self, dim: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(cfg: Config):
    return nn.LayerNorm(cfg.d_model) if cfg.norm == 'layernorm' else RMSNorm(cfg.d_model)


class Attention(nn.Module):
    """Explicit Q/K/V/O linears (weights [out,in]) so the export maps 1:1 onto the HLS arrays."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.h, self.dh, self.causal = cfg.num_heads, cfg.head_dim, cfg.causal
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.k = nn.Linear(cfg.d_model, cfg.d_model)
        self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, x_rope):
        B, T, D = x.shape
        q = self.q(x_rope).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x_rope).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(mask, float('-inf'))
        p = self.drop(scores.softmax(-1))
        ctx = (p @ v).transpose(1, 2).reshape(B, T, D)
        return self.o(ctx)


class EncoderLayer(nn.Module):
    """Post-norm block, identical structure to the HLS: x = N1(x + Attn(x)); x = N2(x + FFN(x))."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.use_attention = cfg.use_attention
        if cfg.use_attention:
            self.attn = Attention(cfg)
            self.norm1 = make_norm(cfg)
        self.ff1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.ff2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.norm2 = make_norm(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        q = lambda t: _qa(self, t)
        qr = lambda t: _qa(self, t, 'resid_fmt')
        if self.use_attention:
            o = qr(self.drop(self.attn(x, q(apply_rope(x, cos, sin)))))
            x = q(self.norm1(qr(x + o)))
        f = qr(self.drop(self.ff2(self.drop(F.relu(self.ff1(x))))))
        x = q(self.norm2(qr(x + f)))
        return x


class EMGRoFormer(nn.Module):
    """Conv/patch embedding -> N encoder layers -> pooling -> absolute and/or delta heads.
    Delta head outputs are standardised; `delta_scale` (per joint std from the training set) maps them
    back to normalised-angle units. With cond_prev the previous angle estimate is projected and added
    to the pooled feature, so the model can learn to correct its own accumulated error."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Conv1d(cfg.num_channels, cfg.d_model, cfg.conv_kernel, cfg.conv_stride, cfg.conv_padding)
        self.rope = RotaryEmbedding(cfg.d_model)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.drop = nn.Dropout(cfg.dropout)
        self.cond_prev = cfg.cond_prev
        if cfg.cond_prev:
            self.prev_proj = nn.Linear(cfg.num_angles, cfg.d_model)
        if cfg.heads in ('absolute', 'both'):
            self.fc_abs = nn.Linear(cfg.d_model, cfg.num_angles)
        if cfg.heads in ('delta', 'both'):
            self.fc_delta = nn.Linear(cfg.d_model, cfg.num_angles)
        self.register_buffer('delta_scale', torch.ones(cfg.num_angles))
        self.register_buffer('angle_var', torch.ones(cfg.num_angles))   # per-joint train variance (loss normalisation)
        self.delta_detach = getattr(cfg, 'delta_detach', False)

    def set_delta_scale(self, scale: np.ndarray):
        self.delta_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))

    def set_angle_var(self, var: np.ndarray):
        self.angle_var.copy_(torch.as_tensor(var, dtype=torch.float32))

    def features(self, x):
        q = lambda t: _qa(self, t)
        x = self.drop(q(self.embed(x).transpose(1, 2)))    # [B, T, D]
        cos, sin = self.rope(x.shape[1])
        for layer in self.layers:
            x = layer(x, cos, sin)
        return q(x.mean(1) if self.cfg.pooling == 'mean' else x[:, -1])

    def forward(self, x, prev: Optional[torch.Tensor] = None):
        q = lambda t: _qa(self, t)
        f = self.features(x)
        if self.cond_prev:
            if prev is None:
                raise ValueError('cond_prev=True requires prev angles')
            f = q(f + self.prev_proj(q(prev)))
        out = {}
        if hasattr(self, 'fc_abs'):
            out['abs'] = q(self.fc_abs(f))
        if hasattr(self, 'fc_delta'):
            fd = f.detach() if (self.training and self.delta_detach) else f
            out['delta_std'] = q(self.fc_delta(fd))
            out['delta'] = out['delta_std'] * self.delta_scale
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def estimate_macs(cfg: Config) -> dict:
    """Rough MAC count per inference, mirroring the HLS loop structure."""
    T, D, Fh, H = cfg.num_tokens, cfg.d_model, cfg.d_ff, cfg.num_heads
    conv = T * D * cfg.num_channels * cfg.conv_kernel
    attn = (4 * T * D * D + 2 * H * T * T * cfg.head_dim) if cfg.use_attention else 0
    ffn = 2 * T * D * Fh
    head = D * cfg.num_angles * (2 if cfg.heads == 'both' else 1) + (cfg.num_angles * D if cfg.cond_prev else 0)
    total = conv + cfg.num_layers * (attn + ffn) + head
    return {'tokens': T, 'conv': conv, 'attention': attn * cfg.num_layers, 'ffn': ffn * cfg.num_layers,
            'head': head, 'total': total, 'ms_at_1MAC_per_cycle_100MHz': total / 1e5}


# =====================================================================================
# 5. FIXED-POINT FAKE QUANTISATION (ap_fixed<W,I> with round-to-nearest, saturate; STE backward)
# =====================================================================================

class _FixedPointSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, I):
        frac = W - I
        s = 2.0 ** frac
        lo, hi = -(2.0 ** (I - 1)), 2.0 ** (I - 1) - 1.0 / s
        return torch.clamp(torch.round(x * s) / s, lo, hi)

    @staticmethod
    def backward(ctx, g):
        return g, None, None


def fq(x, fmt: Tuple[int, int]):
    return _FixedPointSTE.apply(x, int(fmt[0]), int(fmt[1]))


def _qa(module, t, which: str = 'act_fmt'):
    """Fake-quantise an activation if QAT formats were attached to the module (see convert_to_qat).
    which='act_fmt' for data_t points, 'resid_fmt' for the residual stream."""
    fmt = getattr(module, which, None)
    return fq(t, fmt) if fmt is not None else t


class QLinear(nn.Linear):
    def __init__(self, lin: nn.Linear, w_fmt, a_fmt):
        super().__init__(lin.in_features, lin.out_features, lin.bias is not None)
        self.load_state_dict(lin.state_dict())
        self.w_fmt, self.a_fmt = w_fmt, a_fmt

    def forward(self, x):
        b = fq(self.bias, self.w_fmt) if self.bias is not None else None
        return F.linear(fq(x, self.a_fmt), fq(self.weight, self.w_fmt), b)


class QConv1d(nn.Conv1d):
    def __init__(self, conv: nn.Conv1d, w_fmt, a_fmt):
        super().__init__(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                         conv.padding, bias=conv.bias is not None)
        self.load_state_dict(conv.state_dict())
        self.w_fmt, self.a_fmt = w_fmt, a_fmt

    def forward(self, x):
        b = fq(self.bias, self.w_fmt) if self.bias is not None else None
        return F.conv1d(fq(x, self.a_fmt), fq(self.weight, self.w_fmt), b, self.stride, self.padding)


def convert_to_qat(model: nn.Module, cfg: Config) -> nn.Module:
    """Replace every Linear/Conv1d with a fake-quantised twin using cfg.weight_fmt / cfg.act_fmt, and
    fake-quantise activations at the same points as the HLS / numpy_forward (conv output, branch outputs,
    residual sums, norm outputs, pooled feature, head outputs). Without the latter, the residual stream can
    silently grow beyond the ap_fixed integer range during training."""
    if isinstance(model, (EMGRoFormer, EncoderLayer)):
        model.act_fmt = tuple(cfg.act_fmt)
        model.resid_fmt = tuple(cfg.resid_fmt)
    for name, child in list(model.named_children()):
        if isinstance(child, QLinear) or isinstance(child, QConv1d):
            continue
        if isinstance(child, nn.Linear):
            setattr(model, name, QLinear(child, cfg.weight_fmt, cfg.act_fmt))
        elif isinstance(child, nn.Conv1d):
            setattr(model, name, QConv1d(child, cfg.weight_fmt, cfg.act_fmt))
        else:
            convert_to_qat(child, cfg)
    return model


def quantise_array(a: np.ndarray, fmt: Tuple[int, int], rounding: str = 'round') -> np.ndarray:
    W, I = fmt
    s = 2.0 ** (W - I)
    lo, hi = -(2.0 ** (I - 1)), 2.0 ** (I - 1) - 1.0 / s
    q = np.round(a * s) if rounding == 'round' else np.floor(a * s)
    return np.clip(q / s, lo, hi)


# =====================================================================================
# 6. TRAINING AND WINDOW-LEVEL EVALUATION
# =====================================================================================

def _loss_windows(model: EMGRoFormer, batch, cfg: Config, device, train: bool):
    x = batch['x'].to(device); a = batch['angle'].to(device)
    ap = batch['angle_prev'].to(device); d = batch['delta'].to(device); hp = batch['has_prev'].to(device)
    prev = ap + torch.randn_like(ap) * cfg.feedback_noise_std if (train and cfg.cond_prev) else ap
    out = model(x, prev if cfg.cond_prev else None)
    loss, logs = 0.0, {}
    nv = model.angle_var if getattr(cfg, 'loss_norm', True) else torch.ones_like(model.angle_var)
    if 'abs' in out:
        l = (((out['abs'] - a) ** 2) / nv).mean(); loss = loss + cfg.lambda_abs * l; logs['abs'] = l.item()
        logs['abs_rmse'] = float(F.mse_loss(out['abs'].detach(), a).sqrt())
    if 'delta' in out:
        w = hp.unsqueeze(1)
        l = ((out['delta_std'] - d / model.delta_scale) ** 2 * w).sum() / (w.sum() * d.shape[1] + 1e-8)
        loss = loss + cfg.lambda_delta * l; logs['delta'] = l.item()
    return loss, logs, out


def _loss_unrolled(model: EMGRoFormer, batch, cfg: Config, device, train: bool):
    """Integration in the loop: est_{t} = fuse(est_{t-1} + delta_t, abs_t); loss on the trajectory."""
    x = batch['x'].to(device); a = batch['angle'].to(device); ap = batch['angle_prev'].to(device)
    d = batch['delta'].to(device)
    B, K = x.shape[:2]
    est = ap[:, 0]
    nv = model.angle_var if getattr(cfg, 'loss_norm', True) else torch.ones_like(model.angle_var)
    nmse = lambda u, v: (((u - v) ** 2) / nv).mean()
    loss, l_traj, l_abs, l_delta = 0.0, 0.0, 0.0, 0.0
    for t in range(K):
        prev_in = est + torch.randn_like(est) * cfg.feedback_noise_std if train else est
        out = model(x[:, t], prev_in if cfg.cond_prev else None)
        new = est
        if 'delta' in out:
            new = est + out['delta']
            l_delta = l_delta + F.mse_loss(out['delta_std'], d[:, t] / model.delta_scale)
        if 'abs' in out:
            l_abs = l_abs + nmse(out['abs'], a[:, t])
            new = (1 - cfg.fuse_alpha) * new + cfg.fuse_alpha * out['abs'] if 'delta' in out else out['abs']
        l_traj = l_traj + nmse(new, a[:, t])
        est = new
    loss = (cfg.lambda_traj * l_traj + cfg.lambda_abs * l_abs + cfg.lambda_delta * l_delta) / K
    return loss, {'traj': float(l_traj.detach()) / K, 'abs': float(getattr(l_abs, 'detach', lambda: l_abs)()) / K,
                  'delta': float(getattr(l_delta, 'detach', lambda: l_delta)()) / K}, None


def run_epoch(model, loader, cfg, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    fn = _loss_unrolled if cfg.unroll_steps > 1 else _loss_windows
    tot, n, agg = 0.0, 0, {}
    with torch.set_grad_enabled(train):
        for batch in loader:
            loss, logs, _ = fn(model, batch, cfg, device, train)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot += loss.item(); n += 1
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v
    return tot / max(n, 1), {k: v / max(n, 1) for k, v in agg.items()}


@torch.no_grad()
def predict_windows(model: EMGRoFormer, loader, cfg: Config, device):
    model.eval()
    A, P_abs, P_delta, D = [], [], [], []
    for batch in loader:
        x = batch['x'].to(device); ap = batch['angle_prev'].to(device)
        out = model(x, ap if cfg.cond_prev else None)
        A.append(batch['angle'].numpy()); D.append(batch['delta'].numpy())
        if 'abs' in out: P_abs.append(out['abs'].cpu().numpy())
        if 'delta' in out: P_delta.append(out['delta'].cpu().numpy())
    res = {'angles': np.concatenate(A), 'delta': np.concatenate(D)}
    if P_abs: res['abs'] = np.concatenate(P_abs)
    if P_delta: res['delta_pred'] = np.concatenate(P_delta)
    return res


def window_metrics(res: dict) -> dict:
    m = {}
    if 'abs' in res:
        m['abs_rmse'] = float(np.sqrt(mean_squared_error(res['angles'], res['abs'])))
        m['abs_r2'] = float(r2_score(res['angles'], res['abs']))
    if 'delta_pred' in res:
        m['delta_rmse'] = float(np.sqrt(mean_squared_error(res['delta'], res['delta_pred'])))
        m['delta_r2'] = float(r2_score(res['delta'], res['delta_pred']))
    return m


def make_loaders(cfg: Config, npz_files: Sequence[str]):
    if cfg.unroll_steps > 1:
        tr = SequenceDataset(npz_files, 'train', cfg, cfg.unroll_steps)
        va = SequenceDataset(npz_files, 'test', cfg, cfg.unroll_steps)
    else:
        tr = WindowDataset(npz_files, 'train', cfg)
        va = WindowDataset(npz_files, 'test', cfg, balance_rest=False)
    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    return DataLoader(tr, shuffle=True, drop_last=True, **kw), DataLoader(va, shuffle=False, **kw)


def save_checkpoint(model: EMGRoFormer, cfg: Config, path: str, extra: Optional[dict] = None):
    torch.save({'state_dict': model.state_dict(), 'cfg': cfg.to_dict(), 'extra': extra or {}}, path)


def load_checkpoint(path: str, device=None) -> Tuple[EMGRoFormer, Config, dict]:
    ck = torch.load(path, map_location=device or 'cpu')
    cfg = Config.from_dict(ck['cfg'])
    model = EMGRoFormer(cfg)
    if any(k.endswith('w_fmt') for k in ck['state_dict']) or ck['extra'].get('qat'):
        convert_to_qat(model, cfg)
    model.load_state_dict(ck['state_dict'])
    return model.to(device or 'cpu'), cfg, ck['extra']


def train_model(cfg: Config, npz_files: Sequence[str], run=None, tag: str = '') -> Tuple[EMGRoFormer, dict]:
    """Float training, optional QAT fine-tuning, early stopping on validation loss. Returns best model."""
    seed_everything(cfg.seed)
    device = get_device()
    os.makedirs(cfg.save_dir, exist_ok=True)
    model = EMGRoFormer(cfg).to(device)
    if hasattr(model, 'fc_delta'):
        model.set_delta_scale(delta_scale_from_train(npz_files))
    model.set_angle_var(angle_var_from_train(npz_files))
    train_loader, val_loader = make_loaders(cfg, npz_files)
    sel = getattr(cfg, 'select_metric', 'val_abs')
    sel_key = {'val_abs': 'abs', 'val_delta': 'delta'}.get(sel)
    if sel_key == 'abs' and cfg.heads == 'delta': sel_key = 'delta'
    if sel_key == 'delta' and cfg.heads == 'absolute': sel_key = 'abs'
    print(f'  checkpoint selection on: {"val_" + sel_key if sel_key else "val_loss"}')
    print(f'[{tag or cfg.run_name}] params={count_params(model):,} tokens={cfg.num_tokens} '
          f'train_batches={len(train_loader)} val_batches={len(val_loader)} MACs={estimate_macs(cfg)["total"]:,}')
    ck_path = os.path.join(cfg.save_dir, f'{tag or cfg.run_name}.pt')
    history = []

    def _fit(epochs, lr, phase):
        nonlocal history
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, patience=3)
        best, bad = float('inf'), 0
        for ep in range(epochs):
            t0 = time.time()
            tr_loss, tr_logs = run_epoch(model, train_loader, cfg, device, opt)
            va_loss, va_logs = run_epoch(model, val_loader, cfg, device)
            sched.step(va_loss)
            rec = {'phase': phase, 'epoch': ep + 1, 'train_loss': tr_loss, 'val_loss': va_loss,
                   **{f'train_{k}': v for k, v in tr_logs.items()}, **{f'val_{k}': v for k, v in va_logs.items()},
                   'lr': opt.param_groups[0]['lr'], 'sec': time.time() - t0}
            history.append(rec); wandb_log(run, rec)
            print(f'  {phase} ep {ep + 1:3d} train {tr_loss:.5f} val {va_loss:.5f} ' +
                  ' '.join(f'{k}={v:.4f}' for k, v in va_logs.items()) + f' ({rec["sec"]:.0f}s)')
            score = va_logs.get(sel_key, va_loss) if sel_key else va_loss
            rec['select_score'] = score
            if score < best - 1e-6:
                best, bad = score, 0
                save_checkpoint(model, cfg, ck_path, {'phase': phase, 'val_loss': va_loss, 'select_score': best,
                                                      'select_metric': sel, 'qat': phase == 'qat'})
            else:
                bad += 1
                if bad >= cfg.patience:
                    print(f'  early stop ({phase})'); break
        model.load_state_dict(torch.load(ck_path, map_location=device)['state_dict'])

    _fit(cfg.epochs, cfg.lr, 'float')
    if cfg.qat_epochs > 0:
        convert_to_qat(model, cfg); model.to(device)
        _fit(cfg.qat_epochs, cfg.qat_lr, 'qat')
    return model, {'history': history, 'checkpoint': ck_path}


# =====================================================================================
# 7. SEQUENTIAL EVALUATION, INTEGRATION AND FUSION
# =====================================================================================

@torch.no_grad()
def predict_run(model: EMGRoFormer, run: dict, cfg: Config, device, batch: int = 256,
                feedback: bool = False, fuse_alpha: float = 0.1) -> dict:
    """Ordered predictions for one contiguous run. With feedback=True the model receives its own
    integrated estimate as `prev` (deployment condition); otherwise the true previous angle."""
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(run['sEMG'].transpose(0, 2, 1)))
    n = len(x)
    abs_p = np.zeros((n, cfg.num_angles), np.float32); dl_p = np.zeros_like(abs_p)
    has_abs = hasattr(model, 'fc_abs'); has_delta = hasattr(model, 'fc_delta')
    if not (cfg.cond_prev and feedback):
        prev_all = torch.from_numpy(run['angles_prev'])
        for i in range(0, n, batch):
            out = model(x[i:i + batch].to(device), prev_all[i:i + batch].to(device) if cfg.cond_prev else None)
            if has_abs: abs_p[i:i + batch] = out['abs'].cpu().numpy()
            if has_delta: dl_p[i:i + batch] = out['delta'].cpu().numpy()
    else:
        est = torch.from_numpy(run['angles_prev'][0:1]).to(device)
        for i in range(n):
            out = model(x[i:i + 1].to(device), est)
            if has_abs: abs_p[i] = out['abs'].cpu().numpy()[0]
            if has_delta: dl_p[i] = out['delta'].cpu().numpy()[0]
            new = est + out['delta'] if has_delta else est
            if has_abs:
                new = (1 - fuse_alpha) * new + fuse_alpha * out['abs'] if has_delta else out['abs']
            est = new
    return {'abs': abs_p if has_abs else None, 'delta': dl_p if has_delta else None,
            'angles': run['angles'], 'start': run['angles_prev'][0], 'restim': run['restim'],
            'energy': run['sEMG'].mean(axis=(1, 2))}


def integrate(delta: np.ndarray, start: np.ndarray, leak: float = 0.0, prior: Optional[np.ndarray] = None,
              clamp: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Plain (optionally leaky, clamped) integration of per-stride deltas."""
    est = start.copy(); out = np.zeros_like(delta)
    prior = start if prior is None else prior
    for t in range(len(delta)):
        est = est + delta[t]
        if leak > 0:
            est = (1 - leak) * est + leak * prior
        if clamp is not None:
            est = np.clip(est, *clamp)
        out[t] = est
    return out


def kalman_fuse(delta: np.ndarray, abs_pred: np.ndarray, start: np.ndarray, q: float = 1e-4, r: float = 2e-2,
                energy: Optional[np.ndarray] = None, rest_threshold: Optional[float] = None,
                rest_pose: Optional[np.ndarray] = None, rest_r: float = 1e-3,
                clamp: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """Per-joint scalar Kalman filter. Process model: x_t = x_{t-1} + delta_t (noise q).
    Measurement: absolute head (noise r). Optional second measurement: rest pose when the EMG
    energy is below `rest_threshold` (noise rest_r) -> re-anchoring at rest."""
    x = start.astype(np.float64).copy(); P = np.full_like(x, r)
    out = np.zeros_like(delta, dtype=np.float64)
    for t in range(len(delta)):
        x = x + delta[t]; P = P + q
        K = P / (P + r); x = x + K * (abs_pred[t] - x); P = (1 - K) * P
        if rest_threshold is not None and energy is not None and energy[t] < rest_threshold and rest_pose is not None:
            K2 = P / (P + rest_r); x = x + K2 * (rest_pose - x); P = (1 - K2) * P
        if clamp is not None:
            x = np.clip(x, *clamp)
        out[t] = x
    return out.astype(np.float32)


def fixed_point_integrate(delta: np.ndarray, start: np.ndarray, frac_bits: int, rounding: str = 'floor') -> np.ndarray:
    """Shows why truncation drifts: integrate deltas that are re-quantised every step."""
    s = 2.0 ** frac_bits
    est = np.round(start * s) / s; out = np.zeros_like(delta)
    for t in range(len(delta)):
        v = (est + delta[t]) * s
        est = (np.floor(v) if rounding == 'floor' else np.round(v)) / s
        out[t] = est
    return out


def seq_metrics(est: np.ndarray, truth: np.ndarray) -> dict:
    return {'rmse': float(np.sqrt(mean_squared_error(truth, est))), 'r2': float(r2_score(truth, est)),
            'mae': float(np.abs(truth - est).mean())}


def error_vs_horizon(est: np.ndarray, truth: np.ndarray, bins: Sequence[int] = (1, 5, 10, 20, 40, 80, 160, 320)) -> dict:
    """RMSE as a function of steps since the run started (how fast drift accumulates)."""
    err = ((est - truth) ** 2).mean(axis=1)
    out, lo = {}, 0
    for hi in list(bins) + [10 ** 9]:
        if lo < len(err):
            label = f'{lo}-{hi}' if hi < 10 ** 9 else f'{lo}+'
            out[label] = float(np.sqrt(err[lo:hi].mean()))
        lo = hi
    return out


def evaluate_sequential(model: EMGRoFormer, npz_files: Sequence[str], cfg: Config, device, split: str = 'test',
                        rest_pose: Optional[np.ndarray] = None, rest_threshold: Optional[float] = None,
                        q: float = 1e-4, r: float = 2e-2, feedback: bool = True, init: str = 'truth') -> dict:
    """Runs every ordered run of every subject and reports metrics for several estimators.
    init='truth' seeds integrators/filters with the true angle at the run start (optimistic: a glove reading).
    init='abs'   seeds them with the absolute head's first prediction (deployable, no glove).
    'abs_smoothed' is the Kalman filter with zero process input: it isolates the gain of the DELTA head
    from plain temporal smoothing of the absolute head."""
    est_all: Dict[str, List[np.ndarray]] = {}
    truth_all, horizon_err = [], {}
    for f in npz_files:
        for run in ordered_runs(f, split):
            if len(run['angles']) < 4:
                continue
            p = predict_run(model, run, cfg, device, feedback=feedback)
            if init == 'abs' and p['abs'] is not None:
                p['start'] = p['abs'][0].copy()
            truth = p['angles']; truth_all.append(truth)
            cands = {}
            if p['abs'] is not None:
                cands['abs_only'] = p['abs']
                cands['abs_smoothed'] = kalman_fuse(np.zeros_like(p['abs']), p['abs'], p['start'], q, r, clamp=(-0.1, 1.1))
            if p['delta'] is not None:
                cands['delta_integrated'] = integrate(p['delta'], p['start'])
                cands['delta_leaky_clamped'] = integrate(p['delta'], p['start'], leak=0.01,
                                                         prior=rest_pose, clamp=(-0.1, 1.1))
                cands['delta_fixed_trunc_7b'] = fixed_point_integrate(p['delta'], p['start'], 7, 'floor')
                cands['delta_fixed_round_12b'] = fixed_point_integrate(p['delta'], p['start'], 12, 'round')
            if p['abs'] is not None and p['delta'] is not None:
                cands['kalman'] = kalman_fuse(p['delta'], p['abs'], p['start'], q, r, clamp=(-0.1, 1.1))
                if rest_pose is not None and rest_threshold is not None:
                    cands['kalman_rest_anchor'] = kalman_fuse(p['delta'], p['abs'], p['start'], q, r,
                                                              energy=p['energy'], rest_threshold=rest_threshold,
                                                              rest_pose=rest_pose, clamp=(-0.1, 1.1))
            for k, v in cands.items():
                est_all.setdefault(k, []).append(v)
                horizon_err.setdefault(k, []).append(error_vs_horizon(v, truth))
    truth = np.concatenate(truth_all)
    report = {k: seq_metrics(np.concatenate(v), truth) for k, v in est_all.items()}
    for k, lst in horizon_err.items():
        keys = sorted({kk for d in lst for kk in d}, key=lambda s: int(s.rstrip('+').split('-')[0]))
        report[k]['rmse_vs_horizon'] = {kk: float(np.mean([d[kk] for d in lst if kk in d])) for kk in keys}
    return report


def kalman_sweep(preds: Sequence[dict], qs=(1e-5, 1e-4, 1e-3, 1e-2), rs=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
                 use_delta: bool = True, init: str = 'truth') -> 'object':
    """RMSE grid of the plain Kalman fusion (use_delta=True) or of abs-only smoothing (use_delta=False).
    preds: outputs of predict_run. Returns a pandas DataFrame indexed by q, columns by r."""
    import pandas as pd
    truth = np.concatenate([p['angles'] for p in preds])
    grid = pd.DataFrame(index=[f'q={q:g}' for q in qs], columns=[f'r={r:g}' for r in rs], dtype=float)
    for q in qs:
        for r in rs:
            est = []
            for p in preds:
                start = p['abs'][0] if init == 'abs' else p['start']
                d = p['delta'] if use_delta else np.zeros_like(p['abs'])
                est.append(kalman_fuse(d, p['abs'], start, q, r, clamp=(-0.1, 1.1)))
            grid.loc[f'q={q:g}', f'r={r:g}'] = seq_metrics(np.concatenate(est), truth)['rmse']
    return grid


def rest_energy_threshold(npz_files: Sequence[str], percentile: float = 90.0) -> float:
    """Energy threshold from TRAIN rest windows: mean mu-law amplitude below this => 'at rest'."""
    e = []
    for f in npz_files:
        d = np.load(f)
        m = (d['split'] == 0) & (d['restim'] == 0)
        e.append(d['sEMG'][m].mean(axis=(1, 2)))
    return float(np.percentile(np.concatenate(e), percentile))


# =====================================================================================
# 8. NUMPY REFERENCE FORWARD PASS AND HLS EXPORT
# =====================================================================================

def _np(t):
    return t.detach().cpu().numpy().astype(np.float64)


def extract_weights(model: EMGRoFormer) -> dict:
    """Float weights in the layout the HLS expects ([out][in] row-major, conv [out][in][k])."""
    cfg = model.cfg
    w = {'conv_w': _np(model.embed.weight), 'conv_b': _np(model.embed.bias)}
    for li, layer in enumerate(model.layers):
        p = f'l{li}_'
        if layer.use_attention:
            for n in 'qkvo':
                lin = getattr(layer.attn, n)
                w[p + f'mha_{n}_w'] = _np(lin.weight); w[p + f'mha_{n}_b'] = _np(lin.bias)
            w[p + 'ln1_g'] = _np(layer.norm1.weight)
            w[p + 'ln1_b'] = _np(layer.norm1.bias) if cfg.norm == 'layernorm' else np.zeros(cfg.d_model)
        w[p + 'ff_w1'] = _np(layer.ff1.weight); w[p + 'ff_b1'] = _np(layer.ff1.bias)
        w[p + 'ff_w2'] = _np(layer.ff2.weight); w[p + 'ff_b2'] = _np(layer.ff2.bias)
        w[p + 'ln2_g'] = _np(layer.norm2.weight)
        w[p + 'ln2_b'] = _np(layer.norm2.bias) if cfg.norm == 'layernorm' else np.zeros(cfg.d_model)
    if hasattr(model, 'fc_abs'):
        w['fc_w'] = _np(model.fc_abs.weight); w['fc_b'] = _np(model.fc_abs.bias)
    if hasattr(model, 'fc_delta'):
        # Exported in STANDARDISED units (magnitude ~1): real deltas (~0.005) are below one LSB of
        # ap_fixed<10,3>. The firmware multiplies by DELTA_SCALE (float) before integrating.
        w['fc_delta_w'] = _np(model.fc_delta.weight); w['fc_delta_b'] = _np(model.fc_delta.bias)
        w['delta_scale'] = _np(model.delta_scale)
    if model.cond_prev:
        w['prev_w'] = _np(model.prev_proj.weight); w['prev_b'] = _np(model.prev_proj.bias)
    return w


def rope_tables(num_tokens: int, dim: int, base: float = 10000.0):
    inv = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    fr = np.outer(np.arange(num_tokens, dtype=np.float64), inv)
    return np.cos(fr), np.sin(fr)         # [T, dim/2] (second half identical)


def numpy_forward(w: dict, cfg: Config, x: np.ndarray, prev: Optional[np.ndarray] = None,
                  Q=None, rope_fmt: Optional[Tuple[int, int]] = None, Qr=None) -> dict:
    """Reference forward pass. x: [C, seq_len] channel-major (the HLS input layout).
    Q: optional quantiser applied to weights and activations (fixed-point simulation).
    Returns intermediates named like the HLS debug checkpoints."""
    Q = Q or (lambda a: a)
    Qr = Qr or Q                      # residual-stream quantiser (branch outputs, pre-norm sums)
    W = {k: Q(v) for k, v in w.items()}
    C, L = x.shape
    k, s, p = cfg.conv_kernel, cfg.conv_stride, cfg.conv_padding
    xp = np.pad(Q(x), ((0, 0), (p, p)))
    T = cfg.num_tokens
    conv = np.stack([W['conv_b'] + np.einsum('oik,ik->o', W['conv_w'], xp[:, t * s:t * s + k]) for t in range(T)])
    conv = Q(conv)
    cos_h, sin_h = rope_tables(T, cfg.d_model)
    if rope_fmt is not None:
        cos_h, sin_h = quantise_array(cos_h, rope_fmt), quantise_array(sin_h, rope_fmt)
    cos, sin = np.concatenate([cos_h, cos_h], 1), np.concatenate([sin_h, sin_h], 1)
    out = {'conv1d_out': conv}
    h = conv

    def norm(v, g, b):
        if cfg.norm == 'layernorm':
            m = v.mean(-1, keepdims=True); var = ((v - m) ** 2).mean(-1, keepdims=True)
            return (v - m) / np.sqrt(var + 1e-5) * g + b
        return v / np.sqrt((v ** 2).mean(-1, keepdims=True) + 1e-5) * g

    for li in range(cfg.num_layers):
        pre = f'l{li}_'
        if cfg.use_attention:
            half = cfg.d_model // 2
            rot = np.concatenate([-h[:, half:], h[:, :half]], 1)
            r = Q(h * cos + rot * sin); out[pre + 'rope_out'] = r
            q = r @ W[pre + 'mha_q_w'].T + W[pre + 'mha_q_b']; kk = r @ W[pre + 'mha_k_w'].T + W[pre + 'mha_k_b']
            v = h @ W[pre + 'mha_v_w'].T + W[pre + 'mha_v_b']
            ctx = np.zeros_like(h); hd = cfg.head_dim
            for hh in range(cfg.num_heads):
                sl = slice(hh * hd, (hh + 1) * hd)
                sc = q[:, sl] @ kk[:, sl].T / math.sqrt(hd)
                if cfg.causal:
                    sc = sc + np.triu(np.full((T, T), -1e9), 1)
                sc = sc - sc.max(-1, keepdims=True); e = np.exp(sc); pr = e / e.sum(-1, keepdims=True)
                ctx[:, sl] = pr @ v[:, sl]
            o = Qr(ctx @ W[pre + 'mha_o_w'].T + W[pre + 'mha_o_b'])
            h = Q(norm(Qr(h + o), W[pre + 'ln1_g'], W[pre + 'ln1_b'])); out[pre + 'add_norm1_out'] = h
        hid = np.maximum(Q(h @ W[pre + 'ff_w1'].T + W[pre + 'ff_b1']), 0)
        f = Qr(hid @ W[pre + 'ff_w2'].T + W[pre + 'ff_b2'])
        h = Q(norm(Qr(h + f), W[pre + 'ln2_g'], W[pre + 'ln2_b'])); out[pre + 'add_norm2_out'] = h
    pooled = Q(h.mean(0) if cfg.pooling == 'mean' else h[-1]); out['pool_out'] = pooled
    feat = pooled
    if 'prev_w' in W:
        if prev is None:
            raise ValueError('model is conditioned on prev angles')
        feat = Q(feat + W['prev_w'] @ Q(prev) + W['prev_b']); out['feat_out'] = feat
    if 'fc_w' in W:
        out['full_output'] = Q(W['fc_w'] @ feat + W['fc_b'])
    if 'fc_delta_w' in W:
        out['delta_output'] = Q(W['fc_delta_w'] @ feat + W['fc_delta_b'])       # standardised (HLS output)
        out['delta_real'] = out['delta_output'] * w['delta_scale']               # firmware: x DELTA_SCALE
    return out


def _write_txt(path, arr):
    np.savetxt(path, np.asarray(arr, np.float64).ravel(), fmt='%.8f', newline=' ')


def _write_header_array(f, ctype, name, values, per_line=8, suffix=''):
    values = np.asarray(values).ravel()
    f.write(f'static const {ctype} {name}[{len(values)}] = {{\n')
    for i in range(0, len(values), per_line):
        f.write('    ' + ', '.join(f'{v:.8f}{suffix}' for v in values[i:i + per_line]) + ',\n')
    f.write('};\n\n')


def export_hls(model: EMGRoFormer, cfg: Config, out_dir: str, sample_x: np.ndarray, sample_prev: Optional[np.ndarray],
               hil_windows: Sequence[Tuple[np.ndarray, np.ndarray, int, int]] = (),
               norm_stats: Optional[dict] = None) -> dict:
    """Writes everything the HLS project consumes, all derived from ONE checkpoint:
      verification_data/*.txt   golden intermediates on a REAL window + float weights (full_*.txt)
      weights/*.h               quantised weight headers (names identical to the existing HLS)
      rope_tables.h, config.h, results.h, input_output.h, types_suggested.h
      hil_verification_samples/ real windows + float-model outputs + manifest
      normalization_constants.{json,h}  per-channel EMG scale, glove min/max, mu, causal SOS filter
    sample_x: [C, seq_len] channel-major real input. Returns the fixed-point simulation error report."""
    assert cfg.num_layers == 1, 'HLS currently implements NUM_LAYERS=1'
    model = model.cpu().eval()
    # self-check against a float twin (a QAT model fake-quantises in forward, the numpy reference is float)
    float_model = EMGRoFormer(cfg); float_model.load_state_dict(model.state_dict()); float_model.eval()
    w = extract_weights(float_model)
    vd, wd, hd = (os.path.join(out_dir, d) for d in ('verification_data', 'weights', 'hil_verification_samples'))
    for d in (vd, wd, hd):
        os.makedirs(d, exist_ok=True)

    # --- golden (float) intermediates and self-check against torch ---
    ref = numpy_forward(w, cfg, sample_x, sample_prev)
    with torch.no_grad():
        xt = torch.from_numpy(sample_x[None].astype(np.float32))
        pt = torch.from_numpy(sample_prev[None].astype(np.float32)) if sample_prev is not None else None
        tout = float_model(xt, pt if cfg.cond_prev else None)
    if 'abs' in tout:
        mism = float(np.abs(tout['abs'][0].numpy() - ref['full_output']).max())
        assert mism < 1e-3, f'numpy vs torch mismatch {mism}'
    _write_txt(os.path.join(vd, 'full_input.txt'), sample_x)
    if sample_prev is not None:
        _write_txt(os.path.join(vd, 'full_prev_angles.txt'), sample_prev)
    for k, v in ref.items():
        _write_txt(os.path.join(vd, f'{k}.txt'), v)
    for k, v in w.items():
        _write_txt(os.path.join(vd, f'full_{k}.txt'), v)
    delta_scale = w.pop('delta_scale', None)

    # --- quantised weight headers (same array names as hls_code/weights/*.h) ---
    wt = cfg.export_weight_type
    groups = {
        'conv1d_weights.h': [('conv_w_flat', 'conv_w'), ('conv_b_flat', 'conv_b')],
        'l0_mha_weights.h': [(f'l0_{n}_w_flat', f'l0_mha_{n}_w') for n in 'qkvo'] + [(f'l0_{n}_b_flat', f'l0_mha_{n}_b') for n in 'qkvo'],
        'l0_ln_weights.h': [('l0_ln1_g_flat', 'l0_ln1_g'), ('l0_ln1_b_flat', 'l0_ln1_b'), ('l0_ln2_g_flat', 'l0_ln2_g'), ('l0_ln2_b_flat', 'l0_ln2_b')],
        'l0_ff_weights.h': [('l0_ff_w1_flat', 'l0_ff_w1'), ('l0_ff_b1_flat', 'l0_ff_b1'), ('l0_ff_w2_flat', 'l0_ff_w2'), ('l0_ff_b2_flat', 'l0_ff_b2')],
        'fc_weights.h': [('fc_w_flat', 'fc_w'), ('fc_b_flat', 'fc_b'), ('fc_delta_w_flat', 'fc_delta_w'),
                         ('fc_delta_b_flat', 'fc_delta_b'), ('prev_w_flat', 'prev_w'), ('prev_b_flat', 'prev_b')],
    }
    for fname, items in groups.items():
        guard = fname.upper().replace('.', '_')
        with open(os.path.join(wd, fname), 'w') as f:
            f.write(f'#ifndef {guard}\n#define {guard}\n#include "../types.h"\n\n')
            f.write(f'// generated by emg_common.export_hls  weight format ap_fixed<{cfg.weight_fmt[0]},{cfg.weight_fmt[1]}>\n\n')
            for cname, key in items:
                if key in w:
                    _write_header_array(f, wt, cname, quantise_array(w[key], cfg.weight_fmt))
            f.write(f'#endif // {guard}\n')

    # --- RoPE tables (half-dim, 2-D, same shape as hls_code/rope_tables.h) ---
    T, D = cfg.num_tokens, cfg.d_model
    cos_h, sin_h = rope_tables(T, D)
    with open(os.path.join(out_dir, 'rope_tables.h'), 'w') as f:
        f.write(f'// generated: SEQ_LEN(tokens)={T}, DIM={D}, HALF_DIM={D // 2}, type ap_fixed<{cfg.rope_fmt[0]},{cfg.rope_fmt[1]}>\n')
        f.write('#ifndef ROPE_TABLES_H\n#define ROPE_TABLES_H\n#include "types.h"\n#ifdef __SYNTHESIS__\n#include <ap_fixed.h>\n#else\n#include "mock_ap_fixed.h"\n#endif\n')
        f.write(f'typedef ap_fixed<{cfg.rope_fmt[0]}, {cfg.rope_fmt[1]}> rope_t;\n#define ROPE_HALF_DIM {D // 2}\n\n')
        for name, tab in (('ROPE_COS_TABLE', cos_h), ('ROPE_SIN_TABLE', sin_h)):
            f.write(f'static const rope_t {name}[{T}][{D // 2}] = {{\n')
            for t in range(T):
                f.write('  {' + ', '.join(f'{v:.10f}f' for v in quantise_array(tab[t], cfg.rope_fmt)) + ('},\n' if t < T - 1 else '}\n'))
            f.write('};\n\n')
        f.write('#endif // ROPE_TABLES_H\n')

    # --- config.h ---
    with open(os.path.join(out_dir, 'config.h'), 'w') as f:
        f.write(f'#define OUT_DIM {cfg.num_angles}\n#define SEQ_LEN {cfg.num_tokens}\n#define RAW_SEQ_LEN {cfg.seq_len}\n')
        f.write(f'#define D_MODEL {D}\n#define IN_CHANNELS {cfg.num_channels}\n#define OUT_CHANNELS {D}\n')
        f.write(f'#define KERNEL_SIZE {cfg.conv_kernel}\n#define CONV_STRIDE {cfg.conv_stride}\n#define CONV_PADDING {cfg.conv_padding}\n')
        f.write(f'#define NUM_HEADS {cfg.num_heads}\n#define HEAD_DIM {cfg.head_dim}\n#define D_FF {cfg.d_ff}\n')
        f.write(f'#define ROPE_HALF_DIM {D // 2}\n#define NUM_LAYERS {cfg.num_layers}\n')
        f.write(f'#define USE_ATTENTION {int(cfg.use_attention)}\n#define CAUSAL_ATTENTION {int(cfg.causal)}\n')
        f.write(f'#define NORM_IS_RMS {int(cfg.norm == "rmsnorm")}\n#define POOL_LAST_TOKEN {int(cfg.pooling == "last")}\n')
        f.write(f'#define HAS_ABS_HEAD {int("fc_w" in w)}\n#define HAS_DELTA_HEAD {int("fc_delta_w" in w)}\n#define COND_PREV {int(cfg.cond_prev)}\n')
    with open(os.path.join(out_dir, 'types_suggested.h'), 'w') as f:
        f.write(f'typedef ap_fixed<{cfg.act_fmt[0]}, {cfg.act_fmt[1]}, AP_RND, AP_SAT> data_t;\n')
        f.write(f'typedef ap_fixed<{cfg.weight_fmt[0]}, {cfg.weight_fmt[1]}, AP_RND, AP_SAT> weight_t;\n')
        f.write(f'typedef ap_fixed<{cfg.rope_fmt[0]}, {cfg.rope_fmt[1]}, AP_RND, AP_SAT> rope_t;\n')
        f.write(f'typedef ap_fixed<{cfg.resid_fmt[0]}, {cfg.resid_fmt[1]}, AP_RND, AP_SAT> resid_t;  // branch outputs + pre-norm residual sums (ff_out, ln_in)\n')

    # --- results.h and input_output.h for the C testbench ---
    with open(os.path.join(wd, 'results.h'), 'w') as f:
        f.write('#ifndef RESULTS_H\n#define RESULTS_H\n#include "../types.h"\n\n')
        for k, v in ref.items():
            _write_header_array(f, 'float', f'ref_{k}', v)
        f.write('#endif // RESULTS_H\n')
    with open(os.path.join(wd, 'input_output.h'), 'w') as f:
        f.write('#ifndef INPUT_OUTPUT_H\n#define INPUT_OUTPUT_H\n#include "../types.h"\n\n')
        _write_header_array(f, 'data_t', 'input', quantise_array(sample_x, cfg.act_fmt))
        if sample_prev is not None:
            _write_header_array(f, 'data_t', 'prev_angles', quantise_array(sample_prev, cfg.act_fmt))
        if 'full_output' in ref:
            _write_header_array(f, 'float', 'output', ref['full_output'])
        if 'delta_output' in ref:
            _write_header_array(f, 'float', 'delta_output', ref['delta_output'])
        f.write('#endif // INPUT_OUTPUT_H\n')

    # --- HIL samples (real windows, float-model outputs) ---
    if delta_scale is not None:
        w['delta_scale'] = delta_scale
    with open(os.path.join(hd, 'manifest.txt'), 'w') as mf:
        mf.write('idx subject restim rep input_file abs_output... | delta_output...\n')
        for i, (xw, pw, restim, rep) in enumerate(hil_windows):
            r = numpy_forward(w, cfg, xw, pw)
            _write_txt(os.path.join(hd, f'sample_{i:03d}_input.txt'), xw)
            if pw is not None:
                _write_txt(os.path.join(hd, f'sample_{i:03d}_prev.txt'), pw)
            outs = []
            if 'full_output' in r:
                _write_txt(os.path.join(hd, f'sample_{i:03d}_output.txt'), r['full_output']); outs += list(r['full_output'])
            if 'delta_output' in r:
                _write_txt(os.path.join(hd, f'sample_{i:03d}_delta.txt'), r['delta_output']); outs += ['|'] + list(r['delta_output'])
            mf.write(f'{i} - {restim} {rep} sample_{i:03d}_input.txt ' + ' '.join(f'{v:.5f}' if not isinstance(v, str) else v for v in outs) + '\n')

    # --- normalisation constants for the firmware ---
    if norm_stats is not None:
        sos = bandpass_sos(cfg)
        ns = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in norm_stats.items()}
        ns.update({'causal_filter': cfg.causal_filter, 'sos': sos.tolist(), 'fs': cfg.fs, 'seq_len': cfg.seq_len, 'stride': cfg.stride})
        if delta_scale is not None:
            ns['delta_scale'] = delta_scale.tolist()
        with open(os.path.join(out_dir, 'normalization_constants.json'), 'w') as f:
            json.dump(ns, f, indent=2)
        with open(os.path.join(out_dir, 'normalization_constants.h'), 'w') as f:
            f.write('#ifndef NORMALIZATION_CONSTANTS_H\n#define NORMALIZATION_CONSTANTS_H\n')
            f.write(f'#define EMG_MU {cfg.mu}\n#define EMG_FS {cfg.fs}\n#define EMG_SOS_SECTIONS {len(sos)}\n\n')
            _write_header_array(f, 'float', 'EMG_SCALE', norm_stats['emg_scale'])
            _write_header_array(f, 'float', 'GLOVE_MIN', norm_stats['glove_min'])
            _write_header_array(f, 'float', 'GLOVE_MAX', norm_stats['glove_max'])
            _write_header_array(f, 'float', 'EMG_SOS', sos, per_line=6)
            if delta_scale is not None:
                _write_header_array(f, 'float', 'DELTA_SCALE', delta_scale)
            f.write('#endif\n')

    w.pop('delta_scale', None)
    # --- expected fixed-point error (what a correct HLS build should reproduce) ---
    Qa = lambda a: quantise_array(a, cfg.act_fmt)
    wq = {k: quantise_array(v, cfg.weight_fmt) for k, v in w.items()}
    if delta_scale is not None:
        w['delta_scale'] = delta_scale; wq['delta_scale'] = delta_scale
    fx = numpy_forward(wq, cfg, sample_x, sample_prev, Q=Qa, rope_fmt=cfg.rope_fmt, Qr=lambda a: quantise_array(a, cfg.resid_fmt))
    report = {}
    for k in ref:
        a, b = ref[k].ravel(), fx[k].ravel()
        report[k] = {'mae': float(np.abs(a - b).mean()), 'cos': float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))}
    with open(os.path.join(out_dir, 'fixed_point_expectation.json'), 'w') as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, 'EXPORT_INFO.json'), 'w') as f:
        json.dump({'cfg': cfg.to_dict(), 'macs': estimate_macs(cfg), 'params': count_params(model),
                   'array_names': sorted(w.keys())}, f, indent=2)
    return report
