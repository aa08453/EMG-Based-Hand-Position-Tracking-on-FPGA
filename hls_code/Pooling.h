#ifndef POOLING_H
#define POOLING_H
#include "types.h"
#include "config.h"

// Mean over the SEQ_LEN tokens: pooled[d] = (1/SEQ_LEN) * sum_t x[t][d].
void pooling(const data_t input[SEQ_LEN * D_MODEL], data_t pooled[D_MODEL])
{
#pragma HLS INLINE off
    acc_t acc[D_MODEL];
    POOL_RESET:
    for (int d = 0; d < D_MODEL; ++d) {
#pragma HLS PIPELINE II=1
        acc[d] = 0;
    }
    POOL_SUM:
    for (int t = 0; t < SEQ_LEN; ++t) {
        for (int d = 0; d < D_MODEL; ++d) {
#pragma HLS PIPELINE II=1
            acc[d] += (acc_t)input[t * D_MODEL + d];
        }
    }
    const pool_scale_t inv = 1.0 / SEQ_LEN;
    POOL_SCALE:
    for (int d = 0; d < D_MODEL; ++d) {
#pragma HLS PIPELINE II=1
        pooled[d] = (data_t)(acc[d] * inv);
    }
}
#endif // POOLING_H
