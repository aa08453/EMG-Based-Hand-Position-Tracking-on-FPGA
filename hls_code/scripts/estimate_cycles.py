"""
estimate_cycles.py
------------------
Analytic latency estimate for RoFormer_Top (v2) from its loop structure, before synthesis.
Assumptions: every pipelined loop reaches its target II; each pipelined loop invocation adds a
fill/drain overhead (DEPTH cycles); float exp/div/rsqrt latencies are typical Vitis values.
Replace with the csynth report numbers once synthesis has been run.

Usage: python scripts/estimate_cycles.py [--unroll 4] [--clock_mhz 100]
"""
import argparse

T, D, F, H, HD, C, K, RAW, OUT = 25, 128, 256, 8, 16, 12, 8, 200, 10
DEPTH = 8          # pipeline fill/drain per loop invocation (cycles)
EXP_LAT, DIV_LAT, RSQRT_LAT = 20, 16, 20


def dense(n_in, n_out, u):
    return n_out * n_in // u + DEPTH


def estimate(u):
    c = {}
    c['read input (AXI)'] = C * RAW + DEPTH
    c['conv embedding'] = T * ((C * K + DEPTH) + (D * C + DEPTH))
    c['rope'] = T * (2 * (D // 2) + DEPTH)                                  # II=2
    c['Q,K,V projections'] = 3 * T * dense(D, D, u)
    per_query = (T + DEPTH) + (T + EXP_LAT) + DIV_LAT + (T + DEPTH) + (T + DEPTH)   # dot, exp, 1/sum, norm, wsum
    c['attention (scores, softmax, weighting)'] = H * T * per_query
    c['O projection'] = T * dense(D, D, u)
    ln = 3 * (D + DEPTH) + RSQRT_LAT
    c['residual + LayerNorm 1'] = T * ((D + DEPTH) + ln)
    c['FFN'] = T * (dense(D, F, u) + dense(F, D, u))
    c['residual + LayerNorm 2'] = T * ((D + DEPTH) + ln)
    c['mean pooling'] = D + T * D + D + 3 * DEPTH
    c['two output heads'] = 2 * dense(D, OUT, u)
    c['write output'] = 2 * OUT + DEPTH
    return c


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--unroll', type=int, nargs='*', default=[2, 4, 8])
    ap.add_argument('--clock_mhz', type=float, default=100.0)
    a = ap.parse_args()
    for u in a.unroll:
        c = estimate(u); tot = sum(c.values())
        print(f'\nMAC_UNROLL={u}: {tot:,} cycles = {tot / a.clock_mhz / 1e3:.2f} ms at {a.clock_mhz:.0f} MHz (budget 25 ms)')
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
            print(f'  {k:40s} {v:>9,}  {100 * v / tot:5.1f} %')
    print('\nOriginal design (200 tokens, 1 MAC/cycle): >= 37,377,280 cycles = 374 ms, before softmax/LayerNorm float latencies.')
