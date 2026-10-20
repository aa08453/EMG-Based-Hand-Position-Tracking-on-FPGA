#ifndef TYPES_H
#define TYPES_H
// Fixed-point types for the RoFormer. The four QAT formats (act_fmt, weight_fmt, resid_fmt, rope_fmt in
// kaggle_scripts/emg_common.py) come from the generated model_config.h; the defaults are the v2 formats.
// Ranges were checked on real test windows: see PROJECT_REPORT.md, section 10.
#include <ap_fixed.h>
#include "model_config.h"
#ifndef ACT_W
#define ACT_W 10
#define ACT_I 3
#endif
#ifndef WEIGHT_W
#define WEIGHT_W 10
#define WEIGHT_I 3
#endif
#ifndef RESID_W
#define RESID_W 16
#define RESID_I 8
#endif
#ifndef ROPE_W
#define ROPE_W 8
#define ROPE_I 2
#endif

// Activations: linear inputs, norm outputs, pooled feature, head outputs. v2 (10,3): range +/-4, step 1/128.
typedef ap_fixed<ACT_W, ACT_I, AP_RND, AP_SAT> data_t;
// Weights and biases (all layers). v2 (10,3). Header values are already on this grid.
typedef ap_fixed<WEIGHT_W, WEIGHT_I, AP_RND, AP_SAT> weight_t;
// Residual stream: attention/FFN branch outputs and pre-norm sums. v2 (16,8): range +/-128.
// Trained FFN outputs reach |66| and pre-norm sums |70|; data_t would saturate at 4.
typedef ap_fixed<RESID_W, RESID_I, AP_RND, AP_SAT> resid_t;
// Storage type of the constant weight arrays in weights/*.h. Synthesis: weight_t, so each ROM word is
// WEIGHT_W bits. C simulation: float holding the same grid values (exact for these widths); every read
// converts to weight_t, so the arithmetic is identical. Reason: a few hundred thousand ap_fixed static
// initialisers make g++ run out of memory (2+ encoder layers) and slow down Vitis C simulation.
#ifdef __SYNTHESIS__
typedef weight_t wrom_t;
#else
typedef float wrom_t;
#endif
// RoPE cos/sin tables.
typedef ap_fixed<ROPE_W, ROPE_I, AP_RND, AP_SAT> rope_t;
// Q/K/V projections (|q| <= 5.2 measured). 18 bits so each multiply fits one DSP48.
typedef ap_fixed<18, 5, AP_RND, AP_SAT> qkv_t;
// Softmax probabilities in [0, 1].
typedef ap_ufixed<18, 1, AP_RND, AP_SAT> prob_t;
// Accumulators: exact activation x weight products (up to 16 fractional bits) summed over up to 256 terms.
typedef ap_fixed<34, 14> acc_t;
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
