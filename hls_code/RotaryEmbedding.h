#ifndef ROTARYEMBEDDING_H
#define ROTARYEMBEDDING_H
#include "types.h"
#include "config.h"
#include "rope_tables.h"

// RoPE on the full D_MODEL vector of every token (rotation before the Q/K projections).
// out[i] = x[i]*cos + rot[i]*sin, rot = (-x[i+64] for i<64, x[i-64] otherwise). Tables: [SEQ_LEN][64].
void rope_forward(const data_t input[SEQ_LEN * D_MODEL], data_t output[SEQ_LEN * D_MODEL])
{
#pragma HLS INLINE
    ROPE_TOKEN:
    for (int t = 0; t < SEQ_LEN; ++t) {
        ROPE_HALF:
        for (int i = 0; i < ROPE_HALF_DIM; ++i) {
#pragma HLS PIPELINE II=2
            const int lo = t * D_MODEL + i, hi = lo + ROPE_HALF_DIM;
            const data_t x1 = input[lo], x2 = input[hi];
            const rope_t c = ROPE_COS_TABLE[t][i], s = ROPE_SIN_TABLE[t][i];
            output[lo] = (data_t)(x1 * c - x2 * s);
            output[hi] = (data_t)(x2 * c + x1 * s);
        }
    }
}
#endif // ROTARYEMBEDDING_H
