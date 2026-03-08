"""
Generate RoPE cos/sin lookup tables as C++ header constants.
Exploits symmetry: cos[t][i] == cos[t][i + DIM/2], so we only store [SEQ_LEN][DIM/2].
Tables use ap_fixed<8,2> (range [-2, 2), 6 fractional bits).

Output: rope_tables.h
"""
import numpy as np
import os

SEQ_LEN = 400
DIM = 128
HALF_DIM = DIM // 2
BASE = 10000

def generate_rope_tables():
    # inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
    inv_freq = 1.0 / (BASE ** (np.arange(0, DIM, 2, dtype=np.float64) / DIM))
    # inv_freq shape: [half_dim]
    
    t = np.arange(SEQ_LEN, dtype=np.float64)
    
    # freqs = outer(t, inv_freq)  shape: [seq_len, half_dim]
    freqs = np.outer(t, inv_freq)
    
    # Only store the first half — second half is identical
    cos_table = np.cos(freqs)  # [seq_len, half_dim]
    sin_table = np.sin(freqs)  # [seq_len, half_dim]
    
    return cos_table, sin_table

def format_fixed_value(val):
    """Format a float as a C++ initializer for ap_fixed<8,2>"""
    val = max(-2.0, min(val, 1.999))
    return f"{val:.10f}f"

def write_header(cos_table, sin_table, filepath):
    with open(filepath, 'w') as f:
        f.write("// Auto-generated RoPE cos/sin tables (HALF-DIM — exploits symmetry)\n")
        f.write("// DO NOT EDIT — regenerate with gen_rope_tables.py\n")
        f.write(f"// SEQ_LEN={SEQ_LEN}, DIM={DIM}, HALF_DIM={HALF_DIM}, BASE={BASE}\n")
        f.write(f"// Type: ap_fixed<8,2> (range [-2, 2), 6 fractional bits)\n")
        f.write(f"// Symmetry: COS[t][i] == COS[t][i+HALF_DIM], access with i % HALF_DIM\n\n")
        f.write("#ifndef ROPE_TABLES_H\n")
        f.write("#define ROPE_TABLES_H\n\n")
        f.write("#include \"types.h\"\n\n")
        
        f.write("// Narrower fixed-point type for cos/sin values (range [-1, 1])\n")
        f.write("#ifdef __SYNTHESIS__\n")
        f.write("#include <ap_fixed.h>\n")
        f.write("#else\n")
        f.write("#include \"mock_ap_fixed.h\"\n")
        f.write("#endif\n")
        f.write("typedef ap_fixed<8, 2> rope_t;\n\n")
        
        f.write(f"#define ROPE_HALF_DIM {HALF_DIM}\n\n")
        
        # Write cos table — only [SEQ_LEN][HALF_DIM]
        f.write(f"static const rope_t ROPE_COS_TABLE[{SEQ_LEN}][{HALF_DIM}] = {{\n")
        for t in range(SEQ_LEN):
            f.write("  {")
            vals = [format_fixed_value(cos_table[t][d]) for d in range(HALF_DIM)]
            f.write(", ".join(vals))
            if t < SEQ_LEN - 1:
                f.write("},\n")
            else:
                f.write("}\n")
        f.write("};\n\n")
        
        # Write sin table — only [SEQ_LEN][HALF_DIM]
        f.write(f"static const rope_t ROPE_SIN_TABLE[{SEQ_LEN}][{HALF_DIM}] = {{\n")
        for t in range(SEQ_LEN):
            f.write("  {")
            vals = [format_fixed_value(sin_table[t][d]) for d in range(HALF_DIM)]
            f.write(", ".join(vals))
            if t < SEQ_LEN - 1:
                f.write("},\n")
            else:
                f.write("}\n")
        f.write("};\n\n")
        
        f.write("#endif // ROPE_TABLES_H\n")

if __name__ == "__main__":
    cos_t, sin_t = generate_rope_tables()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "rope_tables.h")
    
    write_header(cos_t, sin_t, out_path)
    print(f"Generated {out_path}")
    print(f"  cos range: [{cos_t.min():.6f}, {cos_t.max():.6f}]")
    print(f"  sin range: [{sin_t.min():.6f}, {sin_t.max():.6f}]")
    print(f"  Table size: {SEQ_LEN} x {HALF_DIM} x 2 = {SEQ_LEN*HALF_DIM*2} entries (HALVED)")
