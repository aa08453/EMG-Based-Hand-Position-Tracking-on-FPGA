"""Inference pipeline: raw EMG -> preprocessing -> ONNX model -> Kalman fusion.

Depends only on numpy, scipy and onnxruntime. All constants are in model/runtime_config.json.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.signal import sosfilt

ROOT = Path(__file__).resolve().parent


def load_config(path=ROOT / 'model' / 'runtime_config.json') -> dict:
    with open(path) as f:
        cfg = json.load(f)
    for key in ('sos', 'emg_scale', 'glove_min', 'glove_max', 'delta_scale'):
        cfg[key] = np.asarray(cfg[key], dtype=np.float64)
    return cfg


def rectify(emg: np.ndarray, sos: np.ndarray, zi=None):
    """Causal band-pass then rectify, as in training. emg: [samples, channels]."""
    if zi is None:
        return np.abs(sosfilt(sos, emg, axis=0))
    y, zi = sosfilt(sos, emg, axis=0, zi=zi)
    return np.abs(y), zi


def normalise_rectified(rect: np.ndarray, scale: np.ndarray, mu: float) -> np.ndarray:
    """Per-channel scale, clip to 0..1, then mu-law compression."""
    x = np.clip(rect / scale, 0.0, 1.0)
    return (np.log1p(mu * x) / np.log1p(mu)).astype(np.float32)


class StreamingPreprocessor:
    """Takes raw EMG in chunks of any size and returns model-ready windows every `stride` samples.

    The filter state carries across chunks, so chunk size does not change the result.
    """

    def __init__(self, cfg: dict, emg_scale=None):
        self.sos = cfg['sos']
        self.mu = cfg['mu']
        self.scale = cfg['emg_scale'] if emg_scale is None else emg_scale
        self.seq_len, self.stride, self.channels = cfg['seq_len'], cfg['stride'], cfg['num_channels']
        self.reset()

    def reset(self):
        self.zi = np.zeros((self.sos.shape[0], 2, self.channels))
        self.buffer = np.zeros((self.seq_len, self.channels), np.float32)
        self.seen = 0
        self.until_next = self.stride

    def push(self, samples: np.ndarray) -> list:
        """samples: [n, channels] raw EMG. Returns a list of [1, channels, seq_len] float32 windows."""
        rect, self.zi = rectify(np.asarray(samples, np.float64), self.sos, self.zi)
        y = normalise_rectified(rect, self.scale, self.mu)
        windows, i = [], 0
        while i < len(y):
            take = min(self.until_next, len(y) - i)
            self.buffer = np.concatenate([self.buffer[take:], y[i:i + take]])
            self.seen += take
            self.until_next -= take
            i += take
            if self.until_next == 0:
                self.until_next = self.stride
                if self.seen >= self.seq_len:
                    windows.append(np.ascontiguousarray(self.buffer.T[None]))
        return windows


class HandModel:
    """ONNX model. Input [batch, 12, 400] float32; outputs absolute angle and per-step delta."""

    def __init__(self, cfg: dict):
        self.session = ort.InferenceSession(str(ROOT / cfg['model_file']), providers=['CPUExecutionProvider'])

    def predict(self, window: np.ndarray):
        abs_pred, delta = self.session.run(None, {'emg': window})
        return abs_pred, delta


class KalmanFusion:
    """Per-joint scalar Kalman filter, seeded from the first absolute prediction.

    Predict with the delta head, correct with the absolute head. The gain settles at `fusion_gain`,
    the constant the FPGA firmware uses.
    """

    def __init__(self, cfg: dict):
        self.q, self.r = cfg['kalman_q'], cfg['kalman_r']
        self.lo, self.hi = cfg['kalman_clamp']
        self.reset()

    def reset(self):
        self.x = None
        self.p = None
        self.gain = None

    def update(self, delta: np.ndarray, abs_pred: np.ndarray) -> np.ndarray:
        if self.x is None:
            self.x = np.asarray(abs_pred, np.float64).copy()
            self.p = np.full_like(self.x, self.r)
        self.x = self.x + delta
        self.p = self.p + self.q
        self.gain = self.p / (self.p + self.r)
        self.x = self.x + self.gain * (abs_pred - self.x)
        self.p = (1 - self.gain) * self.p
        self.x = np.clip(self.x, self.lo, self.hi)
        return self.x.copy()


def to_glove_units(norm: np.ndarray, cfg: dict) -> np.ndarray:
    """Normalised 0..1 angle -> glove units (close to degrees)."""
    return norm * (cfg['glove_max'] - cfg['glove_min']) + cfg['glove_min']
