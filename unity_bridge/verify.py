#!/usr/bin/env python3
"""Checks that this folder works on its own. Prints PASS or FAIL for each check."""
import sys

import numpy as np

from emg_runtime import ROOT, HandModel, KalmanFusion, StreamingPreprocessor, load_config, normalise_rectified, rectify

failed = False


def check(name, ok, detail):
    global failed
    failed |= not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


cfg = load_config()
model = HandModel(cfg)

# 1. The model reproduces the fixed-point reference exported with the FPGA build (real EMG windows).
with np.load(ROOT / 'sample_data' / 'hil_windows.npz') as d:
    emg, ref = d['emg'], d['fixed_reference']
abs_err = delta_err = 0.0
for window, expected in zip(emg, ref):
    a, dl = model.predict(window[None])
    abs_err = max(abs_err, float(np.abs(a[0] - expected[:10]).max()))
    delta_err = max(delta_err, float(np.abs(dl[0] / cfg['delta_scale'] - expected[10:]).max()))
check('absolute head vs fixed-point reference', abs_err < 0.02, f'max error {abs_err:.4f} over {len(emg)} windows')
check('delta head vs fixed-point reference', delta_err < 0.15, f'max error {delta_err:.4f} (standardised units)')

# 2. Streaming gives the same windows as filtering the whole signal at once, for any chunk size.
rng = np.random.default_rng(0)
raw = rng.normal(0, 2e-4, (4000, cfg['num_channels']))
whole = normalise_rectified(rectify(raw, cfg['sos']), cfg['emg_scale'], cfg['mu'])
L, S = cfg['seq_len'], cfg['stride']
expected = [whole[i:i + L].T for i in range(0, len(raw) - L + 1, S)]
for chunk in (1, 37, 100, 1000):
    pre, windows = StreamingPreprocessor(cfg), []
    for i in range(0, len(raw), chunk):
        windows += pre.push(raw[i:i + chunk])
    err = max(float(np.abs(w[0] - e).max()) for w, e in zip(windows, expected))
    check(f'streaming, chunks of {chunk}', len(windows) == len(expected) and err < 1e-5, f'{len(windows)} windows, max error {err:.1e}')

# 3. The Kalman gain settles at the constant used by the firmware.
fusion = KalmanFusion(cfg)
zeros = np.zeros(cfg['num_angles'])
for _ in range(2000):
    fusion.update(zeros, zeros)
gain = float(fusion.gain[0])
check('steady-state Kalman gain', abs(gain - cfg['fusion_gain']) < 1e-4, f"{gain:.5f} vs firmware {cfg['fusion_gain']:.5f}")

sys.exit(1 if failed else 0)
