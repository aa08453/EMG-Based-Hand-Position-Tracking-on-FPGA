// Vitis HLS C-simulation / co-simulation testbench for RoFormer_Top.
// Runs one real test window (weights/input_output.h). Pass/fail: both heads against the numpy fixed-point
// simulation (fixed_*); the float-model outputs are printed alongside.
// #include "../types.h"
// #include "../config.h"
// #include "../weights/input_output.h"
// #include "check.h"

// void RoFormer_Top(const data_t input_axi[IN_WORDS], data_t output_axi[NUM_OUTPUTS]);

// int main() {
//     static data_t in[IN_WORDS], out[NUM_OUTPUTS];
//     for (int i = 0; i < IN_WORDS; ++i) in[i] = input[i];
//     RoFormer_Top(in, out);
//     std::printf("RoFormer_Top (%d encoder layers): %d inputs -> %d outputs\n", NUM_LAYERS, IN_WORDS, NUM_OUTPUTS);
//     std::printf("  %-4s %9s %9s %9s | %9s %9s %9s\n", "jnt", "abs hw", "abs fixed", "abs float", "dlt hw", "dlt fixed", "dlt float");
//     for (int j = 0; j < OUT_DIM; ++j)
//         std::printf("  %-4d %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f\n", j, (double)out[j], fixed_full_output[j], output[j],
//                     HAS_DELTA_HEAD ? (double)out[OUT_DIM + j] : 0.0, HAS_DELTA_HEAD ? fixed_delta_output[j] : 0.0,
//                     HAS_DELTA_HEAD ? delta_output[j] : 0.0);
//     bool ok = check_fx("abs head", out, fixed_full_output, output, OUT_DIM);
// #if HAS_DELTA_HEAD
//     ok &= check_fx("delta head", out + OUT_DIM, fixed_delta_output, delta_output, OUT_DIM, 0.02);
// #endif
//     std::printf("%s\n", ok ? "PASS" : "FAIL");
//     return ok ? 0 : 1;
// }


// Vitis HLS C-simulation / co-simulation testbench for RoFormer_Top (AXI Stream version).
// Runs one real test window (weights/input_output.h). Pass/fail: both heads against the numpy fixed-point
// simulation (fixed_*); the float-model outputs are printed alongside.
#include <hls_stream.h>
#include "../types.h"
#include "../config.h"
#include "../weights/input_output.h"
#include "check.h"

// Updated signature using hls::stream
void RoFormer_Top(hls::stream<data_t> &input_stream, hls::stream<data_t> &output_stream);

int main() {
    // Declare hls streams
    hls::stream<data_t> input_stream("input_stream");
    hls::stream<data_t> output_stream("output_stream");

    // Populate the input stream from the test vector array
    for (int i = 0; i < IN_WORDS; ++i) {
        input_stream.write(input[i]);
    }

    // Run the top module with streams
    RoFormer_Top(input_stream, output_stream);

    // Read the results from the output stream back into a local array for verification
    static data_t out[NUM_OUTPUTS];
    for (int i = 0; i < NUM_OUTPUTS; ++i) {
        out[i] = output_stream.read();
    }

    std::printf("RoFormer_Top (%d encoder layers): %d inputs -> %d outputs\n", NUM_LAYERS, IN_WORDS, NUM_OUTPUTS);
    std::printf("  %-4s %9s %9s %9s | %9s %9s %9s\n", "jnt", "abs hw", "abs fixed", "abs float", "dlt hw", "dlt fixed", "dlt float");
    for (int j = 0; j < OUT_DIM; ++j)
        std::printf("  %-4d %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f\n", j, (double)out[j], fixed_full_output[j], output[j],
                    HAS_DELTA_HEAD ? (double)out[OUT_DIM + j] : 0.0, HAS_DELTA_HEAD ? fixed_delta_output[j] : 0.0,
                    HAS_DELTA_HEAD ? delta_output[j] : 0.0);
                    
    bool ok = check_fx("abs head", out, fixed_full_output, output, OUT_DIM);
#if HAS_DELTA_HEAD
    ok &= check_fx("delta head", out + OUT_DIM, fixed_delta_output, delta_output, OUT_DIM, 0.02);
#endif
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}