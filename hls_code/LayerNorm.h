#ifndef LAYERNORM_H
#define LAYERNORM_H
#include "types.h"
#include "config.h"

#ifndef __SYNTHESIS__
#include "mock_hls_math.h"
#else
#include <hls_math.h>
#endif

void ln_forward(
    const data_t input[],
    data_t output[],
    const data_t gamma[],
    const data_t beta[]
) {
#pragma HLS INLINE off

    const ln_calc_t epsilon   = (ln_calc_t)0.0009765625f;
    const ln_calc_t inv_dmodel = (ln_calc_t)(1.0f / D_MODEL); // multiply instead of divide

    ln_calc_t sum = 0;

LN_CALC_MEAN:
    for (int i = 0; i < D_MODEL; ++i) {
#pragma HLS PIPELINE II=1   // no UNROLL here — reduction needs serial accumulation
        sum += (ln_calc_t)input[i];
    }

    // Multiply by reciprocal — avoids the integer-divisor precision collapse
    const ln_calc_t mean = sum * inv_dmodel;

    ln_calc_t sum_sq = 0;

LN_CALC_VAR:
    for (int i = 0; i < D_MODEL; ++i) {
#pragma HLS PIPELINE II=1
        ln_calc_t diff = (ln_calc_t)input[i] - mean;
        sum_sq += diff * diff;
    }

    const ln_calc_t variance = sum_sq * inv_dmodel;

    // rsqrt is fine — cast to float only for the hls::rsqrt call
    const ln_calc_t std_dev_inv = (ln_calc_t)hls::rsqrt((float)(variance + epsilon));

    LN_NORMALIZE:
    for (int i = 0; i < D_MODEL; ++i) {
#pragma HLS PIPELINE II=2
// #pragma HLS UNROLL factor=8

        // Step 1: Normalize
        ln_calc_t norm = ((ln_calc_t)input[i] - mean) * std_dev_inv;
        
        // Step 2: Scale
        ln_calc_t scaled = norm * (ln_calc_t)gamma[i];
        
        // Step 3: Shift
        ln_calc_t shifted = scaled + (ln_calc_t)beta[i];
        
        // Step 4: Cast and Write
        output[i] = (data_t)shifted;
    }
}

#endif // LAYERNORM_H