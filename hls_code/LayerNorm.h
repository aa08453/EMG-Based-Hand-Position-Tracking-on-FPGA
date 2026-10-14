#ifndef LAYERNORM_H
#define LAYERNORM_H
#include "types.h"
#include "config.h"
#include <hls_math.h>

// y = (x - mean) / sqrt(var + eps) * gamma + beta, over D_MODEL values. Input is the residual
// stream (resid_t), output data_t. Wide sums because pre-norm values reach |70| (variance 117).
template <typename TI>
void ln_forward(const TI input[], int off, data_t output[], int out_off,
                const data_t gamma[], const data_t beta[])
{
#pragma HLS INLINE
    const ln_acc_t eps = 1e-5;                       // PyTorch LayerNorm default
    ln_acc_t sum = 0;
    LN_MEAN:
    for (int i = 0; i < D_MODEL; ++i) {
#pragma HLS PIPELINE II=1
        sum += (ln_acc_t)input[off + i];
    }
    const ln_acc_t mean = sum >> 7;                   // / 128 (D_MODEL)
    ln_acc_t sq = 0;
    LN_VAR:
    for (int i = 0; i < D_MODEL; ++i) {
#pragma HLS PIPELINE II=1
        const ln_diff_t d = (ln_diff_t)((ln_acc_t)input[off + i] - mean);
        sq += (ln_acc_t)(d * d);
    }
    const ln_acc_t var = sq >> 7;
    const ln_inv_t inv = (ln_inv_t)hls::rsqrt((float)(var + eps));
    LN_NORM:
    for (int i = 0; i < D_MODEL; ++i) {
#pragma HLS PIPELINE II=1
        const ln_diff_t d = (ln_diff_t)((ln_acc_t)input[off + i] - mean);
        const ln_z_t z = (ln_z_t)(d * inv);
        output[out_off + i] = (data_t)(z * gamma[i] + beta[i]);
    }
}
#endif // LAYERNORM_H
