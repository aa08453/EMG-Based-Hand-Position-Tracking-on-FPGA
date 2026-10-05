"""
gen_results_header.py
---------------------
Reads every .txt file in verification_data/ and converts them into
C++ array constants inside weights/results.h for the HLS testbench.

Usage:
    python scripts/gen_results_header.py

The generated arrays follow the naming convention:
    ref_<filename_without_extension>[<num_elements>]

For example, verification_data/conv1d_out.txt  →  ref_conv1d_out[25600]
"""

import os, sys, glob

VERIFICATION_DIR = os.path.join(os.path.dirname(__file__), "..", "verification_data")
OUTPUT_FILE      = os.path.join(os.path.dirname(__file__), "..", "weights", "results.h")

# Files to convert (order matters for readability; we include all .txt)
# We skip weight files (full_*_w.txt, full_*_b.txt, full_*_g.txt) 
# as those are already in separate weight headers.
SKIP_PREFIXES = [
    "full_conv_w", "full_conv_b",
    "full_fc_w", "full_fc_b",
    "full_l0_ff_w1", "full_l0_ff_b1", "full_l0_ff_w2", "full_l0_ff_b2",
    "full_l0_ln1_g", "full_l0_ln1_b", "full_l0_ln2_g", "full_l0_ln2_b",
    "full_l0_mha_q_w", "full_l0_mha_q_b",
    "full_l0_mha_k_w", "full_l0_mha_k_b",
    "full_l0_mha_v_w", "full_l0_mha_v_b",
    "full_l0_mha_o_w", "full_l0_mha_o_b",
    "rope_cos_table", "rope_sin_table",
]

def should_skip(basename):
    name_no_ext = os.path.splitext(basename)[0]
    return name_no_ext in SKIP_PREFIXES

def read_values(filepath):
    """Read space-separated floats from a single-line (or multi-line) text file."""
    with open(filepath, "r") as f:
        text = f.read().strip()
    if not text:
        return []
    return text.split()

def main():
    abs_vdir = os.path.abspath(VERIFICATION_DIR)
    abs_out  = os.path.abspath(OUTPUT_FILE)
    
    txt_files = sorted(glob.glob(os.path.join(abs_vdir, "*.txt")))
    
    if not txt_files:
        print(f"ERROR: No .txt files found in {abs_vdir}")
        sys.exit(1)

    lines = []
    lines.append("#ifndef RESULTS_H")
    lines.append("#define RESULTS_H")
    lines.append('#include "../types.h"')
    lines.append("")
    
    stats = []
    
    for fpath in txt_files:
        basename = os.path.basename(fpath)
        if should_skip(basename):
            continue
        
        name_no_ext = os.path.splitext(basename)[0]
        array_name  = f"ref_{name_no_ext}"
        
        values = read_values(fpath)
        n = len(values)
        
        if n == 0:
            print(f"  SKIP (empty): {basename}")
            continue
        
        stats.append((array_name, n, basename))
        
        # Format: 12 values per line for readability
        lines.append(f"// Source: verification_data/{basename}  ({n} values)")
        lines.append(f"const float {array_name}[{n}] = {{")
        
        VALUES_PER_LINE = 12
        for i in range(0, n, VALUES_PER_LINE):
            chunk = values[i:i+VALUES_PER_LINE]
            row = "    " + ", ".join(chunk)
            if i + VALUES_PER_LINE < n:
                row += ","
            lines.append(row)
        
        lines.append("};")
        lines.append("")
    
    lines.append("#endif // RESULTS_H")
    lines.append("")
    
    with open(abs_out, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\nGenerated {abs_out}")
    print(f"Arrays created: {len(stats)}")
    print(f"{'Array Name':<40} {'Elements':>10}  Source")
    print("-" * 75)
    for name, count, src in stats:
        print(f"  {name:<38} {count:>10}  {src}")

if __name__ == "__main__":
    main()
