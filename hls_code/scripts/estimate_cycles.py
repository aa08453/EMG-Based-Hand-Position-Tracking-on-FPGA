"""
estimate_cycles.py
------------------
Analytic latency estimate for RoFormer_Top from its loop structure, before synthesis.
Dimensions come from model_config.h (the installed model) unless overridden on the command line.
Assumptions: every pipelined loop reaches its target II; each pipelined loop invocation adds a
fill/drain overhead (DEPTH cycles); float exp/div/rsqrt latencies are typical Vitis values.
Replace with the csynth report numbers once synthesis has been run.

Usage: python scripts/estimate_cycles.py [--unroll 2 4 8] [--layers 2] [--clock_mhz 100] [--budget_ms 50]
"""
import argparse
import os
import re

DEPTH = 8          # pipeline fill/drain per loop invocation (cycles)
EXP_LAT, DIV_LAT, RSQRT_LAT = 20, 16, 20


def read_model_config(path):
    cfg = {}
    if os.path.exists(path):
        for k, v in re.findall(r'#define (\w+) (-?\d+)', open(path).read()):
            cfg[k] = int(v)
    return cfg


def dense(n_in, n_out, u):
    return n_out * n_in // u + DEPTH


def estimate(u, L, T, D, F, H, C, K, RAW, OUT, heads):
    c = {}
    c['read input (AXI)'] = C * RAW + DEPTH
    c['conv embedding'] = T * ((C * K + DEPTH) + (D * C + DEPTH))
    c['rope'] = L * T * (2 * (D // 2) + DEPTH)                              # II=2
    c['Q,K,V projections'] = L * 3 * T * dense(D, D, u)
    per_query = (T + DEPTH) + (T + EXP_LAT) + DIV_LAT + (T + DEPTH) + (T + DEPTH)   # dot, exp, 1/sum, norm, wsum
    c['attention (scores, softmax, weighting)'] = L * H * T * per_query
    c['O projection'] = L * T * dense(D, D, u)
    ln = 3 * (D + DEPTH) + RSQRT_LAT
    c['residual + LayerNorm 1'] = L * T * ((D + DEPTH) + ln)
    c['FFN'] = L * T * (dense(D, F, u) + dense(F, D, u))
    c['residual + LayerNorm 2'] = L * T * ((D + DEPTH) + ln)
    c['mean pooling'] = D + T * D + D + 3 * DEPTH
    c['output heads'] = heads * dense(D, OUT, u)
    c['write output'] = heads * OUT + DEPTH
    return c


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    m = read_model_config(os.path.join(here, '..', 'model_config.h'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--unroll', type=int, nargs='*', default=[2, 4, 8])
    ap.add_argument('--layers', type=int, default=m.get('NUM_LAYERS', 1))
    ap.add_argument('--tokens', type=int, default=m.get('SEQ_LEN', 25))
    ap.add_argument('--raw', type=int, default=m.get('RAW_SEQ_LEN', 200), help='EMG samples per window')
    ap.add_argument('--d_ff', type=int, default=m.get('D_FF', 256))
    ap.add_argument('--clock_mhz', type=float, default=100.0)
    ap.add_argument('--budget_ms', type=float, default=50.0, help='window step in ms (50 for the 200 ms / 50 ms model)')
    a = ap.parse_args()
    K = a.raw // a.tokens
    D, H, C, OUT = m.get('D_MODEL', 128), m.get('NUM_HEADS', 8), m.get('IN_CHANNELS', 12), m.get('OUT_DIM', 10)
    heads = 1 + m.get('HAS_DELTA_HEAD', 1)
    print(f'{a.layers} encoder layer(s), {a.tokens} tokens from {a.raw} samples (kernel {K}), D_FF {a.d_ff}')
    for u in a.unroll:
        c = estimate(u, a.layers, a.tokens, D, a.d_ff, H, C, K, a.raw, OUT, heads); tot = sum(c.values())
        ms = tot / a.clock_mhz / 1e3
        print(f'\nMAC_UNROLL={u}: {tot:,} cycles = {ms:.2f} ms at {a.clock_mhz:.0f} MHz (budget {a.budget_ms:.0f} ms per step)')
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f'  {k:40s} {v:>9,}  {100 * v / tot:5.1f} %')
    print('\nOriginal design (200 tokens, 1 MAC/cycle): >= 37,377,280 cycles = 374 ms, before softmax/LayerNorm float latencies.')
