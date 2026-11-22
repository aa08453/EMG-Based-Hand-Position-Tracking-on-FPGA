// Vitis HLS C-simulation / co-simulation testbench for RoFormer_Top (either top-level interface, tests/top_call.h).
//  1. The golden window (weights/input_output.h): per-joint table, both heads against the numpy fixed-point outputs.
//  2. The hardware-in-the-loop windows (weights/hil_windows.h, 40 real test windows): every output of every window
//     against the numpy fixed-point simulation; the float-model gap is reported for information.
// Optional first argument: number of HIL windows to run (default all). Co-simulation is slow (about 2.4 M cycles
// per window), so scripts/run_hls.tcl passes a smaller number there (COSIM_WINDOWS, default 3).
// Pass/fail uses the same limits as tests/test_layers.cpp.
#include "../types.h"
#include "../config.h"
#include "../weights/input_output.h"
#include "../weights/hil_windows.h"
#include "check.h"
#include "top_call.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    int n_win = HIL_N;
    if (argc > 1 && std::atoi(argv[1]) > 0) n_win = std::atoi(argv[1]) < HIL_N ? std::atoi(argv[1]) : HIL_N;
    static data_t in[IN_WORDS], out[NUM_OUTPUTS];

    // ---- 1. golden window ----
    for (int i = 0; i < IN_WORDS; ++i) in[i] = input[i];
    run_top(in, out);
    std::printf("RoFormer_Top (%d encoder layers, %s ports): %d inputs -> %d outputs\n", NUM_LAYERS,
                TOP_AXIS_STREAM ? "AXI-Stream" : "m_axi", IN_WORDS, NUM_OUTPUTS);
    std::printf("Golden window (weights/input_output.h)\n");
    std::printf("  %-4s %9s %9s %9s | %9s %9s %9s\n", "jnt", "abs hw", "abs fixed", "abs float", "dlt hw", "dlt fixed", "dlt float");
    for (int j = 0; j < OUT_DIM; ++j)
        std::printf("  %-4d %9.4f %9.4f %9.4f | %9.4f %9.4f %9.4f\n", j, (double)out[j], fixed_full_output[j], output[j],
                    HAS_DELTA_HEAD ? (double)out[OUT_DIM + j] : 0.0, HAS_DELTA_HEAD ? fixed_delta_output[j] : 0.0,
                    HAS_DELTA_HEAD ? delta_output[j] : 0.0);
    bool ok = check_fx("abs head", out, fixed_full_output, output, OUT_DIM);
#if HAS_DELTA_HEAD
    ok &= check_fx("delta head", out + OUT_DIM, fixed_delta_output, delta_output, OUT_DIM, 0.02);
#endif

    // ---- 2. HIL windows ----
    std::printf("HIL windows: %d of %d (weights/hil_windows.h)\n", n_win, HIL_N);
    std::printf("  %-4s %10s %10s %10s | %10s %10s\n", "win", "abs fx", "dlt fx", "max fx", "abs float", "dlt float");
    double fa_sum = 0, fd_sum = 0, ra_sum = 0, rd_sum = 0, worst = 0;
    int worst_win = -1;
    for (int w = 0; w < n_win; ++w) {
        for (int i = 0; i < IN_WORDS; ++i) in[i] = (data_t)hil_input[w][i];
        run_top(in, out);
        double fa = 0, fd = 0, ra = 0, rd = 0, mx = 0;
        for (int k = 0; k < NUM_OUTPUTS; ++k) {
            const double h = out[k].to_double(), ef = std::fabs(h - hil_fixed[w][k]), er = std::fabs(h - hil_float[w][k]);
            if (k < OUT_DIM) { fa += ef; ra += er; } else { fd += ef; rd += er; }
            mx = ef > mx ? ef : mx;
        }
        const int nd = NUM_OUTPUTS > OUT_DIM ? NUM_OUTPUTS - OUT_DIM : 1;
        fa /= OUT_DIM; fd /= nd; ra /= OUT_DIM; rd /= nd;
        std::printf("  %-4d %10.5f %10.5f %10.4f | %10.4f %10.4f\n", w, fa, fd, mx, ra, rd);
        fa_sum += fa; fd_sum += fd; ra_sum += ra; rd_sum += rd;
        if (mx > worst) { worst = mx; worst_win = w; }
    }
    const double fa_m = fa_sum / n_win, fd_m = fd_sum / n_win;
    const bool hil_ok = fa_m <= 0.004 && fd_m <= 0.02;
    std::printf("HIL vs numpy fixed point: mean abs MAE=%.5f (limit 0.004)  delta MAE=%.5f (limit 0.02)  "
                "largest single error %.4f (window %d)  %s\n", fa_m, fd_m, worst, worst_win, hil_ok ? "PASS" : "FAIL");
    std::printf("HIL vs float model (quantisation cost, information): abs MAE=%.4f  delta MAE=%.4f\n", ra_sum / n_win, rd_sum / n_win);
    ok &= hil_ok;

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
