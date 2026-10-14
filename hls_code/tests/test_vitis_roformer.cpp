// Vitis HLS C-simulation / co-simulation testbench for RoFormer_Top (v2).
// Runs one real test window (weights/input_output.h) and compares both heads with the float model.
#include "../types.h"
#include "../config.h"
#include "../weights/input_output.h"
#include "check.h"

void RoFormer_Top(const data_t input_axi[IN_WORDS], data_t output_axi[NUM_OUTPUTS]);

int main() {
    static data_t in[IN_WORDS], out[NUM_OUTPUTS];
    for (int i = 0; i < IN_WORDS; ++i) in[i] = input[i];
    RoFormer_Top(in, out);
    std::printf("RoFormer_Top v2: %d inputs -> %d outputs\n", IN_WORDS, NUM_OUTPUTS);
    std::printf("  %-4s %10s %10s | %10s %10s\n", "jnt", "abs hw", "abs ref", "dlt hw", "dlt ref");
    for (int j = 0; j < OUT_DIM; ++j)
        std::printf("  %-4d %10.4f %10.4f | %10.4f %10.4f\n", j, (double)out[j], output[j],
                    HAS_DELTA_HEAD ? (double)out[OUT_DIM + j] : 0.0, HAS_DELTA_HEAD ? delta_output[j] : 0.0);
    bool ok = check("abs head", out, output, OUT_DIM, 0.99, 0.05);
#if HAS_DELTA_HEAD
    ok &= check("delta head", out + OUT_DIM, delta_output, OUT_DIM, 0.99, 0.15);
#endif
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
