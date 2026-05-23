"""
gen_weight_headers_quantized.py
-------------------------------
Converts verification_data/full_*.txt weight files into C++ header files
in the weights/ directory for the HLS testbench, but quantizes values into
a signed 10-bit fixed-point format with 3 integer bits and 6 fractional bits.

This ensures the generated header arrays match the original names but contain
quantized weights suitable for fixed-point HLS evaluation.

Usage:
    python scripts/gen_weight_headers_quantized.py
"""

import os
import argparse

VERIFICATION_DIR = os.path.join(os.path.dirname(__file__), "..", "verification_data")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

SCALE = 1 << 7  # frac bits
MIN_VALUE = - (1 << 2)  # int bits - 1
MAX_VALUE = (1 << 2) - 1.0 / SCALE  # 7.984375
MIN_INT = - (1 << 9)  # -512
MAX_INT = (1 << 9) - 1  # 511


def read_values(filepath):
    with open(filepath, "r") as f:
        text = f.read().strip()
    if not text:
        return []
    return [float(v) for v in text.split()]


def quantize_value(value):
    scaled = round(value * SCALE)
    clamped = max(MIN_INT, min(MAX_INT, scaled))
    return clamped / SCALE


def quantize_values(values):
    quantized = []
    for value in values:
        q = quantize_value(value)
        # Clamp again to the representable minimum and maximum values explicitly
        if q < MIN_VALUE:
            q = MIN_VALUE
        elif q > MAX_VALUE:
            q = MAX_VALUE
        quantized.append(q)
    return quantized


def format_array(name, dtype, values, values_per_line=8):
    n = len(values)
    lines = []
    lines.append(f"static const {dtype} {name}[{n}] = {{")
    for i in range(0, n, values_per_line):
        chunk = values[i:i + values_per_line]
        row = "    " + ", ".join(f"{v:.8f}" for v in chunk)
        if i + values_per_line < n:
            row += ","
        lines.append(row)
    lines.append("};")
    return "\n".join(lines)


def write_header(filepath, guard, includes, arrays):
    with open(filepath, "w", newline="\n") as f:
        f.write(f"#ifndef {guard}\n")
        f.write(f"#define {guard}\n")
        for inc in includes:
            f.write(f"{inc}\n")
        f.write("\n")
        for arr in arrays:
            f.write(arr + "\n\n")
        f.write(f"#endif // {guard}\n")
    print(f"  Generated: {os.path.basename(filepath)}")


def main():
    parser = argparse.ArgumentParser(description="Generate quantized weight headers in Q3.6 fixed-point format.")
    parser.add_argument("--verification-dir", default=VERIFICATION_DIR,
                        help="Path to verification_data directory")
    parser.add_argument("--output-dir", default=WEIGHTS_DIR,
                        help="Path to weights output directory")
    args = parser.parse_args()

    vdir = os.path.abspath(os.path.expanduser(args.verification_dir))
    wdir = os.path.abspath(os.path.expanduser(args.output_dir))

    def load(name):
        path = os.path.join(vdir, f"{name}.txt")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found")
            return []
        values = read_values(path)
        print(f"  Loaded {name}: {len(values)} values")
        return quantize_values(values)

    print("Loading verification data weights...")

    input_ = load("full_input")

    conv_w = load("full_conv_w")
    conv_b = load("full_conv_b")

    mha_q_w = load("full_l0_mha_q_w")
    mha_q_b = load("full_l0_mha_q_b")
    mha_k_w = load("full_l0_mha_k_w")
    mha_k_b = load("full_l0_mha_k_b")
    mha_v_w = load("full_l0_mha_v_w")
    mha_v_b = load("full_l0_mha_v_b")
    mha_o_w = load("full_l0_mha_o_w")
    mha_o_b = load("full_l0_mha_o_b")

    ln1_g = load("full_l0_ln1_g")
    ln1_b = load("full_l0_ln1_b")
    ln2_g = load("full_l0_ln2_g")
    ln2_b = load("full_l0_ln2_b")

    ff_w1 = load("full_l0_ff_w1")
    ff_b1 = load("full_l0_ff_b1")
    ff_w2 = load("full_l0_ff_w2")
    ff_b2 = load("full_l0_ff_b2")

    fc_w = load("full_fc_w")
    fc_b = load("full_fc_b")

    print("\nGenerating quantized weight headers...")

    write_header(
        os.path.join(wdir, "conv1d_weights.h"),
        "CONV1D_PARAMS_H",
        ["#include \"../types.h\""],
        [
            format_array("conv_w_flat", "data_t", conv_w),
            format_array("conv_b_flat", "data_t", conv_b),
        ]
    )

    write_header(
        os.path.join(wdir, "input_output.h"),
        "INPUT_OUTPUT_H",
        ["#include \"../types.h\""],
        [
            format_array("input", "data_t", input_)
        ]
    )

    write_header(
        os.path.join(wdir, "l0_mha_weights.h"),
        "L0_MHA_WEIGHTS_H",
        ["#include \"../types.h\""],
        [
            format_array("l0_q_w_flat", "data_t", mha_q_w),
            format_array("l0_q_b_flat", "data_t", mha_q_b),
            format_array("l0_k_w_flat", "data_t", mha_k_w),
            format_array("l0_k_b_flat", "data_t", mha_k_b),
            format_array("l0_v_w_flat", "data_t", mha_v_w),
            format_array("l0_v_b_flat", "data_t", mha_v_b),
            format_array("l0_o_w_flat", "data_t", mha_o_w),
            format_array("l0_o_b_flat", "data_t", mha_o_b),
        ]
    )

    write_header(
        os.path.join(wdir, "l0_ln_weights.h"),
        "L0_LN_WEIGHTS_H",
        ["#include \"../types.h\""],
        [
            format_array("l0_ln1_g_flat", "data_t", ln1_g),
            format_array("l0_ln1_b_flat", "data_t", ln1_b),
            format_array("l0_ln2_g_flat", "data_t", ln2_g),
            format_array("l0_ln2_b_flat", "data_t", ln2_b),
        ]
    )

    write_header(
        os.path.join(wdir, "l0_ff_weights.h"),
        "L0_FF_WEIGHTS_H",
        ["#include \"../types.h\""],
        [
            format_array("l0_ff_w1_flat", "data_t", ff_w1),
            format_array("l0_ff_b1_flat", "data_t", ff_b1),
            format_array("l0_ff_w2_flat", "data_t", ff_w2),
            format_array("l0_ff_b2_flat", "data_t", ff_b2),
        ]
    )

    write_header(
        os.path.join(wdir, "fc_weights.h"),
        "FC_WEIGHTS_H",
        ["#include \"../types.h\""],
        [
            format_array("fc_w_flat", "data_t", fc_w),
            format_array("fc_b_flat", "data_t", fc_b),
        ]
    )

    print("\nDone! Quantized weight headers written to:")
    print(f"  {wdir}")


if __name__ == "__main__":
    main()
