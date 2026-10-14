#ifndef TYPES_H
#define TYPES_H
// Fixed-point types for the v2 RoFormer (patch embedding, two heads).
// Formats and quantisation points match QAT in kaggle_scripts/emg_common.py (act_fmt, resid_fmt, weight_fmt).
// Ranges were checked on real test windows: see PROJECT_REPORT.md, section 10.
#include <ap_fixed.h>

// Activations: linear inputs, norm outputs, pooled feature, head outputs. QAT act_fmt (10,3): range +/-4, step 1/128.
typedef ap_fixed<10, 3, AP_RND, AP_SAT> data_t;
// Weights and biases. QAT weight_fmt (10,3). Header values are already on this grid.
typedef ap_fixed<10, 3, AP_RND, AP_SAT> weight_t;
// Residual stream: attention/FFN branch outputs and pre-norm sums. QAT resid_fmt (16,8): range +/-128.
// Trained FFN outputs reach |66| and pre-norm sums |70|; data_t would saturate at 4.
typedef ap_fixed<16, 8, AP_RND, AP_SAT> resid_t;
// RoPE cos/sin tables.
typedef ap_fixed<8, 2, AP_RND, AP_SAT> rope_t;
// Q/K/V projections (|q| <= 5.2 measured). 18 bits so each multiply fits one DSP48.
typedef ap_fixed<18, 5, AP_RND, AP_SAT> qkv_t;
// Softmax probabilities in [0, 1].
typedef ap_ufixed<18, 1, AP_RND, AP_SAT> prob_t;
// Accumulators: exact 10x10-bit products (14 fractional bits) summed over up to 256 terms.
typedef ap_fixed<32, 12> acc_t;
// LayerNorm: sums and squared deviations of up to 128 resid_t values (variance <= 117 measured).
// Only adders use the wide ln_acc_t; multipliers use the narrower types below to save DSPs.
typedef ap_fixed<48, 24> ln_acc_t;
typedef ap_fixed<24, 10> ln_diff_t;   // x - mean: |.| < 512, 14 fractional bits
typedef ap_fixed<24, 6>  ln_z_t;      // normalised value: |.| < 32 (|z| <= sqrt(127) by construction)
// LayerNorm 1/sqrt(var + eps): between 0.09 and 1.2 measured, up to 316 for eps-only input.
typedef ap_fixed<28, 10> ln_inv_t;
// 1/SEQ_LEN for mean pooling.
typedef ap_ufixed<18, 1> pool_scale_t;

#endif // TYPES_H
