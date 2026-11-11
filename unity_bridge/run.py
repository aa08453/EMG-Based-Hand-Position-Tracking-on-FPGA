#!/usr/bin/env python3
"""Run the EMG hand model and stream joint angles to Unity over UDP. See README.md."""
import argparse
import json
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from emg_runtime import ROOT, HandModel, KalmanFusion, StreamingPreprocessor, load_config, rectify, to_glove_units


@dataclass
class Frame:
    window: np.ndarray                    # [1, 12, 400], preprocessed
    truth: Optional[np.ndarray] = None    # normalised glove reading, when the source has one
    reset: bool = False                   # the stream is not continuous with the previous frame


def sample_frames(args, cfg):
    """Bundled test windows. They are spread across a recording, so motion between them jumps."""
    with np.load(ROOT / 'sample_data' / 'hil_windows.npz') as d:
        emg = d['emg']
    while True:
        for i, window in enumerate(emg):
            yield Frame(window[None], reset=(i == 0))
        if not args.loop:
            return


def load_raw(path: Path, cfg):
    """Returns (emg [n, 12], glove [n, 10] or None) from a NinaPro .mat, a .npy or a .csv file."""
    glove = None
    if path.suffix == '.mat':
        from scipy.io import loadmat
        d = loadmat(path)
        emg = d['emg']
        if 'glove' in d:
            glove = d['glove'][:, cfg['glove_cols']].astype(np.float64)
    elif path.suffix == '.npy':
        emg = np.load(path)
    else:
        emg = np.loadtxt(path, delimiter=',')
    emg = emg.astype(np.float64)
    if emg.ndim != 2 or emg.shape[1] != cfg['num_channels']:
        sys.exit(f"{path}: expected EMG shaped [samples, {cfg['num_channels']}], got {emg.shape}")
    return emg, glove


def raw_frames(args, cfg):
    """Replays a raw recording through the full streaming pipeline."""
    emg, glove = load_raw(Path(args.file), cfg)
    scale = None
    if args.self_scale:
        scale = np.percentile(rectify(emg, cfg['sos']), 99.9, axis=0) + 1e-8
        if glove is not None:
            cfg['glove_min'], cfg['glove_max'] = glove.min(axis=0), glove.max(axis=0)
    pre = StreamingPreprocessor(cfg, scale)
    stride = cfg['stride']
    while True:
        pre.reset()
        first = True
        for start in range(0, len(emg) - stride + 1, stride):
            for window in pre.push(emg[start:start + stride]):
                truth = None
                if glove is not None:
                    span = cfg['glove_max'] - cfg['glove_min'] + 1e-8
                    truth = (glove[start + stride - 1] - cfg['glove_min']) / span
                yield Frame(window, truth, reset=first)
                first = False
        if not args.loop:
            return


def udp_frames(args, cfg):
    """Live EMG: each datagram is little-endian float32, [n_samples x 12] row-major, in sensor units."""
    pre = StreamingPreprocessor(cfg)
    row_bytes = 4 * cfg['num_channels']
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.emg_host, args.emg_port))
    print(f'waiting for EMG on udp://{args.emg_host}:{args.emg_port}')
    while True:
        data = sock.recv(65507)
        if not data or len(data) % row_bytes:
            print(f'dropped a malformed datagram of {len(data)} bytes', file=sys.stderr)
            continue
        samples = np.frombuffer(data, '<f4').reshape(-1, cfg['num_channels'])
        for window in pre.push(samples):
            yield Frame(window)


def as_list(values):
    return [] if values is None else [round(float(v), 5) for v in values]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['samples', 'raw', 'udp'], default='samples')
    p.add_argument('--file', help='recording for --source raw (.mat, .npy or .csv, shaped [samples, 12])')
    p.add_argument('--self-scale', action='store_true',
                   help='--source raw: compute EMG and glove normalisation from the file, for subjects other than the one in runtime_config.json')
    p.add_argument('--loop', action='store_true', help='restart the recording when it ends')
    p.add_argument('--speed', type=float, default=1.0, help='replay speed; 1 is real time (20 updates per second)')
    p.add_argument('--host', default='127.0.0.1', help='machine running Unity')
    p.add_argument('--port', type=int, default=5005, help='UDP port Unity listens on')
    p.add_argument('--emg-host', default='0.0.0.0', help='--source udp: address to listen on')
    p.add_argument('--emg-port', type=int, default=5006, help='--source udp: port to listen on')
    p.add_argument('--max-frames', type=int, default=0, help='stop after this many frames (0 = no limit)')
    args = p.parse_args()
    if args.source == 'raw' and not args.file:
        p.error('--source raw needs --file')
    if args.speed <= 0:
        p.error('--speed must be positive')

    cfg = load_config()
    model, fusion = HandModel(cfg), KalmanFusion(cfg)
    frames = {'samples': sample_frames, 'raw': raw_frames, 'udp': udp_frames}[args.source](args, cfg)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    step = cfg['stride'] / cfg['fs'] / args.speed
    paced = args.source != 'udp'              # live EMG arrives at its own rate
    print(f"{cfg['model_name']}: source={args.source}, sending to udp://{args.host}:{args.port}")

    seq, start, next_time = 0, time.perf_counter(), time.perf_counter()
    try:
        for frame in frames:
            if frame.reset:
                fusion.reset()
            t0 = time.perf_counter()
            abs_pred, delta = model.predict(frame.window)
            abs_pred, delta = abs_pred[0].astype(np.float64), delta[0].astype(np.float64)
            fused = fusion.update(delta, abs_pred)
            latency_ms = (time.perf_counter() - t0) * 1e3

            truth_angles = None if frame.truth is None else to_glove_units(frame.truth, cfg)
            packet = {
                'seq': seq, 'time': round(time.perf_counter() - start, 4), 'latency_ms': round(latency_ms, 3),
                'fused': as_list(fused), 'abs': as_list(abs_pred), 'delta': as_list(delta), 'truth': as_list(frame.truth),
                'fused_angles': as_list(to_glove_units(fused, cfg)),
                'abs_angles': as_list(to_glove_units(abs_pred, cfg)),
                'truth_angles': as_list(truth_angles),
            }
            sock.sendto(json.dumps(packet).encode(), (args.host, args.port))
            if seq % 20 == 0:
                print(f'frame {seq:6d}  inference {latency_ms:6.2f} ms  fused[0:4] {np.round(fused[:4], 3)}')
            seq += 1
            if args.max_frames and seq >= args.max_frames:
                break
            if paced:
                next_time += step
                time.sleep(max(0.0, next_time - time.perf_counter()))
    except KeyboardInterrupt:
        pass
    print(f'sent {seq} frames')


if __name__ == '__main__':
    main()
