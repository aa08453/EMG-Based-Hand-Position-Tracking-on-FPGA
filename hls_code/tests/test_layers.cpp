// Layer-by-layer, bit-accurate C simulation of the RoFormer against the float golden model.
// Build: make (uses third_party/ap_types, the open-source Xilinx ap_fixed headers).
// Usage: ./test_layers [path/to/hil_verification_samples [dump_outputs.txt]]
// Pass criterion at every layer: agreement with the numpy fixed-point simulation (fx_* in results.h, see
// check_fx in check.h). The float-model columns show the quantisation cost (fixed_point_expectation.json).
#include "../RoFormer_Top.cpp"      // single translation unit: the layer functions live in headers
#include "../weights/input_output.h"
#include "../weights/results.h"
#include "check.h"
#include <fstream>
#include <string>
#include <vector>

#ifndef HIL_ABS_MAE
#define HIL_ABS_MAE 0.04
#endif
#ifndef HIL_ABS_MAX
#define HIL_ABS_MAX 0.06
#endif
#ifndef HIL_DLT_MAE
#define HIL_DLT_MAE 0.2
#endif

static data_t A[SEQ_LEN * D_MODEL], B[SEQ_LEN * D_MODEL], R[SEQ_LEN * D_MODEL], P[D_MODEL], O[NUM_OUTPUTS];

static std::vector<float> load(const std::string& p) {
    std::ifstream f(p); std::vector<float> v; float x; while (f >> x) v.push_back(x); return v;
}

int main(int argc, char** argv) {
    bool ok = true;
    std::printf("Layer-by-layer (input: verification_data/full_input.txt, %d encoder layers, MAC_UNROLL=%d)\n", NUM_LAYERS, MAC_UNROLL);
    const int N = SEQ_LEN * D_MODEL;
    conv1d_forward(input, A);                 ok &= check_fx("conv1d_out", A, fx_conv1d_out, ref_conv1d_out, N);
    for (int l = 0; l < NUM_LAYERS; ++l) {    // each layer starts from the hardware output of the previous one
        char nm[3][32];
        std::snprintf(nm[0], 32, "l%d_rope_out", l); std::snprintf(nm[1], 32, "l%d_add_norm1_out", l);
        std::snprintf(nm[2], 32, "l%d_add_norm2_out", l);
        rope_forward(A, R);                   ok &= check_fx(nm[0], R, fx_enc_rope_out + l * N, ref_enc_rope_out + l * N, N);
        mha_norm_forward(l, A, B);            ok &= check_fx(nm[1], B, fx_enc_norm1_out + l * N, ref_enc_norm1_out + l * N, N);
        ffn_norm_forward(l, B, A);            ok &= check_fx(nm[2], A, fx_enc_norm2_out + l * N, ref_enc_norm2_out + l * N, N);
    }
    pooling(A, P);                            ok &= check_fx("pool_out", P, fx_pool_out, ref_pool_out, D_MODEL);
    dense<D_MODEL, OUT_DIM, false>(P, 0, O, 0, fc_w_flat, fc_b_flat);
    ok &= check_fx("full_output (abs)", O, fx_full_output, ref_full_output, OUT_DIM);
#if HAS_DELTA_HEAD
    dense<D_MODEL, OUT_DIM, false>(P, 0, O, OUT_DIM, fc_delta_w_flat, fc_delta_b_flat);
    ok &= check_fx("delta_output", O + OUT_DIM, fx_delta_output, ref_delta_output, OUT_DIM, 0.02);
#endif

    std::printf("Top level RoFormer_Top on the same window\n");
    data_t top[NUM_OUTPUTS];
    RoFormer_Top(input, top);
    ok &= check_fx("top abs", top, fx_full_output, ref_full_output, OUT_DIM);
#if HAS_DELTA_HEAD
    ok &= check_fx("top delta", top + OUT_DIM, fx_delta_output, ref_delta_output, OUT_DIM, 0.02);
#endif

    if (argc > 1) {   // hardware-in-the-loop samples: real test windows with float-model outputs
        const std::string dir = argv[1];
        // Two criteria, mean absolute error per window:
        //  1. against sample_*_fixed.txt, the numpy fixed-point simulation of the same formats (written by the
        //     export): checks the HLS arithmetic itself. v2 matched to 0.0004 (abs) / 0.002 (delta).
        //  2. against the float model: the cost of quantisation. v2 (QAT, 10-bit) was 0.024 (abs) / 0.12 (delta).
        //     Limits can be changed with -DHIL_ABS_MAE=... -DHIL_ABS_MAX=... -DHIL_DLT_MAE=...
        // Cosine is not used because delta vectors at rest are near zero.
        double abs_mae = 0, abs_max = 0, dlt_mae = 0, fx_abs = 0, fx_dlt = 0; int n = 0, n_fx = 0;
        for (int i = 0; i < 1000; ++i) {
            char nm[64]; std::snprintf(nm, sizeof nm, "/sample_%03d_", i);
            auto x = load(dir + nm + "input.txt"); if (x.size() != IN_WORDS) break;
            auto ya = load(dir + nm + "output.txt"); auto yd = load(dir + nm + "delta.txt");
            auto yf = load(dir + nm + "fixed.txt");
            static data_t xin[IN_WORDS]; for (int k = 0; k < IN_WORDS; ++k) xin[k] = x[k];
            RoFormer_Top(xin, top);
            if (argc > 2) {   // optional dump: one line per window, NUM_OUTPUTS values
                static std::ofstream dump(argv[2]);
                for (int k = 0; k < NUM_OUTPUTS; ++k) dump << top[k].to_double() << (k + 1 < NUM_OUTPUTS ? ' ' : '\n');
            }
            double ea = 0, ed = 0;
            for (int k = 0; k < OUT_DIM; ++k) ea += std::fabs(top[k].to_double() - ya[k]);
#if HAS_DELTA_HEAD
            for (int k = 0; k < OUT_DIM; ++k) ed += std::fabs(top[OUT_DIM + k].to_double() - yd[k]);
#endif
            abs_mae += ea / OUT_DIM; dlt_mae += ed / OUT_DIM; abs_max = std::max(abs_max, ea / OUT_DIM); ++n;
            if (yf.size() == NUM_OUTPUTS) {
                double fa = 0, fd = 0;
                for (int k = 0; k < OUT_DIM; ++k) fa += std::fabs(top[k].to_double() - yf[k]);
                for (int k = OUT_DIM; k < NUM_OUTPUTS; ++k) fd += std::fabs(top[k].to_double() - yf[k]);
                fx_abs += fa / OUT_DIM; fx_dlt += (NUM_OUTPUTS > OUT_DIM ? fd / OUT_DIM : 0); ++n_fx;
            }
        }
        abs_mae /= (n ? n : 1); dlt_mae /= (n ? n : 1);
        const bool hil_ok = n > 0 && abs_mae <= HIL_ABS_MAE && abs_max <= HIL_ABS_MAX && dlt_mae <= HIL_DLT_MAE;
        std::printf("HIL vs float model: %d windows  abs MAE mean=%.4f max=%.4f (limit %.3f/%.3f)  delta MAE=%.4f (limit %.2f)  %s\n",
                    n, abs_mae, abs_max, HIL_ABS_MAE, HIL_ABS_MAX, dlt_mae, HIL_DLT_MAE, hil_ok ? "PASS" : "FAIL");
        ok &= hil_ok;
        if (n_fx) {
            fx_abs /= n_fx; fx_dlt /= n_fx;
            const bool fx_ok = fx_abs <= 0.004 && fx_dlt <= 0.02;
            std::printf("HIL vs numpy fixed point: %d windows  abs MAE=%.5f (limit 0.004)  delta MAE=%.5f (limit 0.02)  %s\n",
                        n_fx, fx_abs, fx_dlt, fx_ok ? "PASS" : "FAIL");
            ok &= fx_ok;
        } else {
            std::printf("HIL vs numpy fixed point: no sample_*_fixed.txt files (older export), skipped\n");
        }
    }
    std::printf("%s\n", ok ? "ALL PASS" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}
