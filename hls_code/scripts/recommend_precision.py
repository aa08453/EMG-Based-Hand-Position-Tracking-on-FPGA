"""
recommend_precision.py
----------------------
Recommends ap_fixed<W, I> bit-widths for each layer by:

  1. Simulating quantization at candidate precisions and measuring MAE
     between the quantized and full-precision weights.
  2. Computing accumulator overflow risk for each linear layer
     (n_macs × max_input × max_weight) and recommending safe int bits.
  3. Printing a full report with per-layer best precision and an
     overall recommendation table.

Usage:
    python recommend_precision.py [--vdir PATH_TO_verification_data]
"""

import os
import sys
import math
import argparse
import numpy as np

# ── default path ───────────────────────────────────────────────────────────────
DEFAULT_VDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "verification_data",
)

# ── candidate precisions to sweep ─────────────────────────────────────────────
# Each entry: (total_bits, int_bits)  →  fractional_bits = total - int
CANDIDATES = [
    (8,  2), (8,  3), (8,  4),
    (10, 2), (10, 3), (10, 4),
    (12, 2), (12, 3), (12, 4),
    (14, 2), (14, 3), (14, 4),
    (16, 2), (16, 3), (16, 4), (16, 6),
    (18, 2), (18, 4), (18, 6),
    (20, 4), (20, 6),
    (24, 6), (24, 8), (24, 10), (24, 12),
    (32, 8), (32, 10), (32, 12), (32, 16),
]

# ── layer definitions ──────────────────────────────────────────────────────────
# Each entry:
#   name           : display name
#   weight_files   : list of txt filenames (weights + biases)
#   n_macs         : number of MAC ops per output neuron (= input dimension)
#   max_input      : realistic worst-case input magnitude entering this layer
#                    (use post-LayerNorm ≈ 3–4 for most; post-ReLU can be higher)
#   current_w_type : current ap_fixed type string for weights
#   current_a_type : current ap_fixed type string for accumulator (sum_t)
LAYERS = [
    {
        "name":          "Input",
        "weight_files":  ["full_input.txt"],
        "n_macs":        None,          # not a linear layer
        "max_input":     None,
        "current_w_type":"ap_fixed<10, 4>",
        "current_a_type": "—",
    },
    {
        "name":          "Conv1d",
        "weight_files":  ["full_conv_w.txt", "full_conv_b.txt"],
        "n_macs":        12,            # kernel_size × in_channels = 3 × 4? adjust
        "max_input":     3.83,          # raw EMG input max from analysis
        "current_w_type":"ap_fixed<10, 4>",
        "current_a_type":"ap_fixed<32, 16>",
    },
    {
        "name":          "MHA Q",
        "weight_files":  ["full_l0_mha_q_w.txt", "full_l0_mha_q_b.txt"],
        "n_macs":        128,           # D_MODEL
        "max_input":     3.83,          # post-RoPE / post-Conv1d
        "current_w_type":"ap_fixed<16, 6>",
        "current_a_type":"ap_fixed<32, 16>",
    },
    {
        "name":          "MHA K",
        "weight_files":  ["full_l0_mha_k_w.txt", "full_l0_mha_k_b.txt"],
        "n_macs":        128,
        "max_input":     3.83,
        "current_w_type":"ap_fixed<16, 6>",
        "current_a_type":"ap_fixed<32, 16>",
    },
    {
        "name":          "MHA V",
        "weight_files":  ["full_l0_mha_v_w.txt", "full_l0_mha_v_b.txt"],
        "n_macs":        128,
        "max_input":     3.83,
        "current_w_type":"ap_fixed<16, 6>",
        "current_a_type":"ap_fixed<32, 16>",
    },
    {
        "name":          "MHA O",
        "weight_files":  ["full_l0_mha_o_w.txt", "full_l0_mha_o_b.txt"],
        "n_macs":        128,
        "max_input":     2.0,           # context vector after weighted sum ≈ ±2
        "current_w_type":"ap_fixed<16, 6>",
        "current_a_type":"ap_fixed<32, 16>",
    },
    {
        "name":          "LN1",
        "weight_files":  ["full_l0_ln1_g.txt", "full_l0_ln1_b.txt"],
        "n_macs":        1,             # element-wise multiply, not a dot-product
        "max_input":     3.0,           # post-LayerNorm input is ~N(0,1), ±3
        "current_w_type":"ap_fixed<10, 4>",
        "current_a_type":"ap_fixed<24, 12>",
    },
    {
        "name":          "LN2",
        "weight_files":  ["full_l0_ln2_g.txt", "full_l0_ln2_b.txt"],
        "n_macs":        1,
        "max_input":     3.0,
        "current_w_type":"ap_fixed<10, 4>",
        "current_a_type":"ap_fixed<24, 12>",
    },
    {
        "name":          "FF1",
        "weight_files":  ["full_l0_ff_w1.txt", "full_l0_ff_b1.txt"],
        "n_macs":        128,           # D_MODEL inputs
        "max_input":     3.0,           # post-LN1
        "current_w_type":"ap_fixed<10, 4>",
        "current_a_type":"ap_fixed<32, 16>",
    },
    {
        "name":          "FF2",
        "weight_files":  ["full_l0_ff_w2.txt", "full_l0_ff_b2.txt"],
        "n_macs":        256,           # FF hidden dim = 2×D_MODEL = 256
        "max_input":     4.0,           # post-ReLU can be larger
        "current_w_type":"ap_fixed<10, 4>",
        "current_a_type":"ap_fixed<32, 16>",
    },
    {
        "name":          "Final FC",
        "weight_files":  ["full_fc_w.txt", "full_fc_b.txt"],
        "n_macs":        128,
        "max_input":     3.0,           # post-LN2
        "current_w_type":"ap_fixed<10, 4>",
        "current_a_type":"ap_fixed<32, 16>",
    },
    {
        "name":          "Output",
        "weight_files":  ["full_output.txt"],
        "n_macs":        None,
        "max_input":     None,
        "current_w_type":"ap_fixed<10, 4>",
        "current_a_type": "—",
    },
]

# ── MAE targets: what is acceptable per layer ─────────────────────────────────
# Expressed as fraction of the layer's own std dev.
# A good rule of thumb: quantization noise < 1% of signal std dev.
MAE_TARGET_FRACTION = 0.01


# ══════════════════════════════════════════════════════════════════════════════
# Core quantization simulation
# ══════════════════════════════════════════════════════════════════════════════

def load_values(filepath):
    with open(filepath, "r") as f:
        return np.array([float(v) for v in f.read().split()])


def quantize(values, total_bits, int_bits):
    """
    Simulate ap_fixed<W, I> quantization.
    frac_bits = total_bits - int_bits
    LSB = 2^(-frac_bits)
    Range: [-2^(int_bits-1),  2^(int_bits-1) - LSB]
    """
    frac_bits = total_bits - int_bits
    lsb = 2.0 ** (-frac_bits)
    min_val = -(2 ** (int_bits - 1))
    max_val =  (2 ** (int_bits - 1)) - lsb

    quantized = np.round(values / lsb) * lsb
    quantized = np.clip(quantized, min_val, max_val)
    return quantized


def quant_error(values, total_bits, int_bits):
    """Return MAE between original and quantized values."""
    q = quantize(values, total_bits, int_bits)
    return float(np.mean(np.abs(values - q)))


def overflow_count(values, total_bits, int_bits):
    """Number of values that would saturate at this precision."""
    frac_bits = total_bits - int_bits
    lsb = 2.0 ** (-frac_bits)
    min_val = -(2 ** (int_bits - 1))
    max_val =  (2 ** (int_bits - 1)) - lsb
    return int(np.sum((values < min_val) | (values > max_val)))


# ══════════════════════════════════════════════════════════════════════════════
# Accumulator analysis
# ══════════════════════════════════════════════════════════════════════════════

def accumulator_analysis(n_macs, max_input, weight_abs_max, weight_std):
    """
    For a linear layer with n_macs multiply-accumulates:

    Worst-case accumulator value = n_macs × max_input × max_weight
    Typical   accumulator value  ≈ sqrt(n_macs) × max_input × weight_std
      (by CLT, random weights partially cancel)

    Returns dict with needed int bits and recommended type.
    """
    worst = n_macs * max_input * weight_abs_max
    typical = math.sqrt(n_macs) * max_input * weight_std

    # int bits must cover worst case (with sign)
    needed_int = max(4, math.ceil(math.log2(worst + 1e-9)) + 2)

    # fractional bits: want quantization noise < 1% of typical output std
    # typical output std ≈ typical accumulator / sqrt(n_macs) ≈ max_input * weight_std
    output_std = max_input * weight_std * math.sqrt(n_macs)
    desired_lsb = output_std * 0.01
    needed_frac = max(6, math.ceil(-math.log2(desired_lsb + 1e-9)))

    total = needed_int + needed_frac
    # round up total to a power-of-2-friendly value
    for t in [16, 18, 20, 24, 28, 32]:
        if t >= total:
            total = t
            break
    else:
        total = 32

    frac = total - needed_int

    return {
        "worst_case":  worst,
        "typical":     typical,
        "needed_int":  needed_int,
        "needed_frac": frac,
        "total":       total,
        "recommended": f"ap_fixed<{total}, {needed_int}>",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-layer sweep
# ══════════════════════════════════════════════════════════════════════════════

def sweep_layer(layer, vdir):
    """
    For one layer, load all weight files, sweep candidate precisions,
    and find the minimum precision that keeps MAE < target.
    """
    all_values = []
    for fname in layer["weight_files"]:
        fpath = os.path.join(vdir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found — skipping", file=sys.stderr)
            continue
        all_values.append(load_values(fpath))

    if not all_values:
        return None

    values = np.concatenate(all_values)
    abs_max = float(np.abs(values).max())
    std_dev = float(values.std())
    mae_target = std_dev * MAE_TARGET_FRACTION

    results = []
    for (w, i) in CANDIDATES:
        if i < 1:
            continue
        mae = quant_error(values, w, i)
        clips = overflow_count(values, w, i)
        frac = w - i
        lsb = 2.0 ** (-frac)
        representable_max = 2 ** (i - 1) - lsb
        results.append({
            "total":    w,
            "int":      i,
            "frac":     frac,
            "lsb":      lsb,
            "mae":      mae,
            "clips":    clips,
            "covers":   representable_max >= abs_max,
        })

    # Best = smallest total bits with zero clips AND mae < target
    valid = [r for r in results if r["clips"] == 0 and r["mae"] <= mae_target]
    if valid:
        best = min(valid, key=lambda r: (r["total"], r["int"]))
    else:
        # Relax: zero clips, best mae
        no_clip = [r for r in results if r["clips"] == 0]
        best = min(no_clip, key=lambda r: r["mae"]) if no_clip else min(results, key=lambda r: r["mae"])

    return {
        "name":       layer["name"],
        "count":      len(values),
        "abs_max":    abs_max,
        "std_dev":    std_dev,
        "mae_target": mae_target,
        "best":       best,
        "results":    results,
        "layer":      layer,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Report helpers
# ══════════════════════════════════════════════════════════════════════════════

W = 128  # line width

def sep(char="─"):
    print(char * W)

def header(text):
    sep("═")
    print(f"  {text}")
    sep("═")


def compare_types(current_str, recommended_str):
    """Return ✅ OK / ⚠ OVERFLOW RISK / 💡 CAN REDUCE / ➡ UPGRADE."""
    def parse(s):
        try:
            parts = s.replace("ap_fixed<", "").replace(">", "").split(",")
            return int(parts[0].strip()), int(parts[1].strip())
        except Exception:
            return None, None

    cw, ci = parse(current_str)
    rw, ri = parse(recommended_str)
    if cw is None or rw is None:
        return "—"

    curr_max = 2 ** (ci - 1)
    rec_max  = 2 ** (ri - 1)

    if curr_max < rec_max:
        return "⚠  OVERFLOW RISK"
    if cw >= rw + 4 and ci >= ri:
        return "💡 CAN REDUCE"
    if cw < rw:
        return "➡  UPGRADE"
    return "✅ OK"


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global MAE_TARGET_FRACTION

    parser = argparse.ArgumentParser()
    parser.add_argument("--vdir", default=DEFAULT_VDIR,
                        help="Path to verification_data directory")
    parser.add_argument("--mae-target", type=float, default=MAE_TARGET_FRACTION,
                        help="MAE target as fraction of std dev (default 0.01 = 1%%)")
    args = parser.parse_args()

    vdir = os.path.abspath(args.vdir)

    MAE_TARGET_FRACTION = args.mae_target

    if not os.path.isdir(vdir):
        print(f"ERROR: verification_data dir not found: {vdir}", file=sys.stderr)
        sys.exit(1)

    print()
    header("RoFormer HLS — Precision Recommendation Report")
    print(f"  MAE target: {MAE_TARGET_FRACTION*100:.2f}% of per-layer std dev")
    print(f"  Verification data: {vdir}")
    print(f"  Candidates swept: {len(CANDIDATES)} bit-width configurations\n")

    layer_results = []
    for layer in LAYERS:
        r = sweep_layer(layer, vdir)
        if r:
            layer_results.append(r)

    # ── Per-layer detail ──────────────────────────────────────────────────────
    sep("═")
    print("  SECTION 1 — WEIGHT STORAGE PRECISION (per layer)")
    sep("═")
    print()

    col = f"  {'Layer':<12} {'Count':>8} {'AbsMax':>10} {'StdDev':>10} "
    col += f"{'MAE Target':>12} {'Best Precision':<22} {'MAE @Best':>12} "
    col += f"{'LSB @Best':>12} {'Clips':>7} {'Current':<18} {'Status'}"
    print(col)
    sep()

    for r in layer_results:
        b = r["best"]
        rec_str = f"ap_fixed<{b['total']}, {b['int']}>"
        curr    = r["layer"]["current_w_type"]
        status  = compare_types(curr, rec_str)

        print(
            f"  {r['name']:<12} {r['count']:>8,} {r['abs_max']:>10.6f} "
            f"{r['std_dev']:>10.6f} {r['mae_target']:>12.8f} "
            f"{rec_str:<22} {b['mae']:>12.8f} "
            f"{b['lsb']:>12.8f} {b['clips']:>7} "
            f"{curr:<18} {status}"
        )

    print()

    # ── MAE vs precision table per layer ─────────────────────────────────────
    sep("═")
    print("  SECTION 2 — MAE vs BIT-WIDTH SWEEP (top candidates per layer)")
    sep("═")

    for r in layer_results:
        print(f"\n  [{r['name']}]  AbsMax={r['abs_max']:.6f}  "
              f"StdDev={r['std_dev']:.6f}  MAE Target={r['mae_target']:.8f}")
        print(f"  {'ap_fixed':<22} {'Int Bits':>9} {'Frac Bits':>10} "
              f"{'LSB':>12} {'MAE':>14} {'Clips':>7} {'Pass?':>7}")
        sep("·")

        # show only configurations that don't clip
        no_clip = [x for x in r["results"] if x["clips"] == 0]
        # sort by total bits then int bits
        no_clip.sort(key=lambda x: (x["total"], x["int"]))
        # show up to 12 rows
        shown = no_clip[:12]
        for x in shown:
            label = f"ap_fixed<{x['total']}, {x['int']}>"
            passed = "✅" if x["mae"] <= r["mae_target"] else "  "
            best_marker = " ◀ BEST" if (x["total"] == r["best"]["total"] and
                                         x["int"]   == r["best"]["int"]) else ""
            print(f"  {label:<22} {x['int']:>9} {x['frac']:>10} "
                  f"{x['lsb']:>12.8f} {x['mae']:>14.10f} "
                  f"{x['clips']:>7} {passed:>7}{best_marker}")

    print()

    # ── Accumulator analysis ─────────────────────────────────────────────────
    sep("═")
    print("  SECTION 3 — ACCUMULATOR (sum_t) ANALYSIS per linear layer")
    sep("═")
    print()
    print(f"  {'Layer':<12} {'n_MACs':>7} {'MaxInput':>10} {'WtAbsMax':>10} "
          f"{'WorstCase':>12} {'Typical':>10} {'Need Int':>9} "
          f"{'Need Frac':>10} {'Rec Acc Type':<22} {'Current Acc':<22} {'Status'}")
    sep()

    for r in layer_results:
        layer = r["layer"]
        if layer["n_macs"] is None or layer["n_macs"] <= 1:
            # element-wise or not a linear layer
            if layer["n_macs"] == 1:
                # LayerNorm — element-wise multiply
                acc_max = layer["max_input"] * r["abs_max"]
                need_int = max(2, math.ceil(math.log2(acc_max + 1e-9)) + 2)
                rec = f"ap_fixed<{need_int + 8}, {need_int}>"
                curr_acc = layer["current_a_type"]
                status = compare_types(curr_acc, rec) if curr_acc != "—" else "—"
                print(
                    f"  {r['name']:<12} {'elem-wise':>7} "
                    f"{layer['max_input']:>10.3f} {r['abs_max']:>10.6f} "
                    f"{acc_max:>12.4f} {'n/a':>10} {need_int:>9} "
                    f"{'8':>10} {rec:<22} {curr_acc:<22} {status}"
                )
            else:
                print(f"  {r['name']:<12} {'—':>7}  (not a linear layer — skip)")
            continue

        n   = layer["n_macs"]
        mi  = layer["max_input"]
        wm  = r["abs_max"]
        ws  = r["std_dev"]
        acc = accumulator_analysis(n, mi, wm, ws)

        curr_acc = layer["current_a_type"]
        status = compare_types(curr_acc, acc["recommended"]) if curr_acc != "—" else "—"

        print(
            f"  {r['name']:<12} {n:>7} {mi:>10.3f} {wm:>10.6f} "
            f"{acc['worst_case']:>12.2f} {acc['typical']:>10.2f} "
            f"{acc['needed_int']:>9} {acc['needed_frac']:>10} "
            f"{acc['recommended']:<22} {curr_acc:<22} {status}"
        )

    print()

    # ── Summary recommendation table ─────────────────────────────────────────
    sep("═")
    print("  SECTION 4 — FINAL RECOMMENDATION SUMMARY")
    sep("═")
    print()
    print(f"  {'Layer':<12} {'Weight Type':>22}  {'Acc Type':>22}  "
          f"{'Weight Status':<18} {'Acc Status'}")
    sep()

    for r in layer_results:
        layer = r["layer"]
        b = r["best"]
        rec_w   = f"ap_fixed<{b['total']}, {b['int']}>"
        curr_w  = layer["current_w_type"]
        w_stat  = compare_types(curr_w, rec_w)

        if layer["n_macs"] and layer["n_macs"] > 1:
            acc = accumulator_analysis(
                layer["n_macs"], layer["max_input"], r["abs_max"], r["std_dev"]
            )
            rec_a  = acc["recommended"]
            curr_a = layer["current_a_type"]
            a_stat = compare_types(curr_a, rec_a) if curr_a != "—" else "—"
        elif layer["n_macs"] == 1:
            acc_max  = layer["max_input"] * r["abs_max"]
            need_int = max(2, math.ceil(math.log2(acc_max + 1e-9)) + 2)
            rec_a    = f"ap_fixed<{need_int + 8}, {need_int}>"
            curr_a   = layer["current_a_type"]
            a_stat   = compare_types(curr_a, rec_a) if curr_a != "—" else "—"
        else:
            rec_a  = "—"
            a_stat = "—"

        print(f"  {r['name']:<12} {rec_w:>22}  {rec_a:>22}  "
              f"{w_stat:<18} {a_stat}")

    print()
    sep()
    print("  LEGEND:")
    print("  ✅ OK          Current type is appropriate")
    print("  💡 CAN REDUCE  Current type is ≥4 bits wider than needed (saves area)")
    print("  ➡  UPGRADE     Current total bits too narrow (rounding error too high)")
    print("  ⚠  OVERFLOW    Current integer bits too narrow (values will saturate)")
    sep()
    print()
    print("  NOTES:")
    print(f"  • MAE target = {MAE_TARGET_FRACTION*100:.2f}% of per-layer weight std dev")
    print("  • Weight recommendation: minimum bits with zero clips AND MAE < target")
    print("  • Accumulator worst-case = n_MACs × max_input × max_weight_abs")
    print("  • Accumulator typical    = sqrt(n_MACs) × max_input × weight_std  (CLT)")
    print("  • Int bits for accumulator include +2 headroom above log2(worst_case)")
    print("  • Frac bits for accumulator keep quant noise < 1% of output std dev")
    print("  • For element-wise layers (LN), n_MACs=1 and acc range = input × weight")
    sep()
    print()


if __name__ == "__main__":
    main()