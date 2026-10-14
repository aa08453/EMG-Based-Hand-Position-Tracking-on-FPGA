// Layer-by-layer, bit-accurate C simulation of the v2 RoFormer against the float golden model.
// Build: make (uses third_party/ap_types, the open-source Xilinx ap_fixed headers).
// Usage: ./test_layers [path/to/hil_verification_samples [dump_outputs.txt]]
// Pass criterion: cosine >= 0.99 at every layer. fixed_point_expectation.json lists ~0.998-0.9999.
#include "../RoFormer_Top.cpp"      // single translation unit: the layer functions live in headers
#include "../weights/input_output.h"
#include "../weights/results.h"
#include "check.h"
#include <fstream>
#include <string>
#include <vector>

static data_t A[SEQ_LEN * D_MODEL], B[SEQ_LEN * D_MODEL], R[SEQ_LEN * D_MODEL], P[D_MODEL], O[NUM_OUTPUTS];

static std::vector<float> load(const std::string& p) {
    std::ifstream f(p); std::vector<float> v; float x; while (f >> x) v.push_back(x); return v;
}

int main(int argc, char** argv) {
    bool ok = true;
    std::printf("Layer-by-layer (input: verification_data/full_input.txt, MAC_UNROLL=%d)\n", MAC_UNROLL);
    conv1d_forward(input, A);                 ok &= check("conv1d_out", A, ref_conv1d_out, SEQ_LEN * D_MODEL, 0.99, 0.05);
    rope_forward(A, R);                       ok &= check("l0_rope_out", R, ref_l0_rope_out, SEQ_LEN * D_MODEL, 0.99, 0.05);
    mha_norm_forward(A, B);                   ok &= check("l0_add_norm1_out", B, ref_l0_add_norm1_out, SEQ_LEN * D_MODEL, 0.99, 0.08);
    ffn_norm_forward(B, A);                   ok &= check("l0_add_norm2_out", A, ref_l0_add_norm2_out, SEQ_LEN * D_MODEL, 0.99, 0.08);
    pooling(A, P);                            ok &= check("pool_out", P, ref_pool_out, D_MODEL, 0.99, 0.08);
    dense<D_MODEL, OUT_DIM, false>(P, 0, O, 0, fc_w_flat, fc_b_flat);
    ok &= check("full_output (abs)", O, ref_full_output, OUT_DIM, 0.99, 0.05);
#if HAS_DELTA_HEAD
    dense<D_MODEL, OUT_DIM, false>(P, 0, O, OUT_DIM, fc_delta_w_flat, fc_delta_b_flat);
    ok &= check("delta_output", O + OUT_DIM, ref_delta_output, OUT_DIM, 0.99, 0.15);
#endif

    std::printf("Top level RoFormer_Top on the same window\n");
    data_t top[NUM_OUTPUTS];
    RoFormer_Top(input, top);
    ok &= check("top abs", top, ref_full_output, OUT_DIM, 0.99, 0.05);
#if HAS_DELTA_HEAD
    ok &= check("top delta", top + OUT_DIM, ref_delta_output, OUT_DIM, 0.99, 0.15);
#endif

    if (argc > 1) {   // hardware-in-the-loop samples: real test windows with float-model outputs
        const std::string dir = argv[1];
        // Criterion: mean absolute error against the float model. The 10-bit format alone costs about
        // 0.024 (abs, about 3 steps of 1/128) and 0.12 (standardised delta) on these windows; the C model
        // matches the numpy fixed-point simulation to 0.0004 / 0.002. Cosine is not used here because
        // delta vectors at rest are near zero.
        double abs_mae = 0, abs_max = 0, dlt_mae = 0; int n = 0;
        for (int i = 0; i < 1000; ++i) {
            char nm[64]; std::snprintf(nm, sizeof nm, "/sample_%03d_", i);
            auto x = load(dir + nm + "input.txt"); if (x.size() != IN_WORDS) break;
            auto ya = load(dir + nm + "output.txt"); auto yd = load(dir + nm + "delta.txt");
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
        }
        abs_mae /= (n ? n : 1); dlt_mae /= (n ? n : 1);
        const bool hil_ok = n > 0 && abs_mae <= 0.04 && abs_max <= 0.06 && dlt_mae <= 0.2;
        std::printf("HIL samples: %d windows  abs MAE mean=%.4f max=%.4f (limit 0.04/0.06)  delta MAE=%.4f (limit 0.2)  %s\n",
                    n, abs_mae, abs_max, dlt_mae, hil_ok ? "PASS" : "FAIL");
        ok &= hil_ok;
    }
    std::printf("%s\n", ok ? "ALL PASS" : "SOME CHECKS FAILED");
    return ok ? 0 : 1;
}
