#ifndef FFNNORMBLOCK_H
#define FFNNORMBLOCK_H
#include "FCLayer.h"
#include "FeedForward.h"
#include "LayerNorm.h"
#include "config.h"
#include "./weights/l0_ff_weights.h"  // <-- weights live here now
#include "./weights/l0_ln_weights.h"

void ffn_norm_forward(const data_t input[], data_t output[])
{
#pragma HLS INLINE off
    FFN_SEQ_LOOP:
    for (int i = 0; i < SEQ_LEN; ++i) {
        data_t ff_in[D_MODEL], ff_out[D_MODEL];
        data_t ln_in[D_MODEL], ln_out[D_MODEL];

#pragma HLS ARRAY_RESHAPE variable=ff_in  cyclic factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=ff_out cyclic factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=ln_in  cyclic factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=ln_out cyclic factor=8 dim=1

#pragma HLS ARRAY_PARTITION variable=l0_ln2_g_flat complete dim=1
#pragma HLS ARRAY_PARTITION variable=l0_ln2_b_flat complete dim=1
#pragma HLS ARRAY_PARTITION variable=l0_ff_b1_flat complete dim=1
#pragma HLS ARRAY_PARTITION variable=l0_ff_b2_flat complete dim=1


        BLOCK_READ_IN:
        for (int d = 0; d < D_MODEL; ++d) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=16
            ff_in[d] = input[(i * D_MODEL) + d];
        }

        ff_forward(ff_in, l0_ff_w1_flat, l0_ff_b1_flat, l0_ff_w2_flat, l0_ff_b2_flat, ff_out);

        BLOCK_RESIDUAL2:
        for (int d = 0; d < D_MODEL; ++d) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=16
            ln_in[d] = ff_in[d] + ff_out[d];
        }

        ln_forward(ln_in, ln_out, l0_ln2_g_flat, l0_ln2_b_flat);

        BLOCK_FINAL_COPY:
        for (int d = 0; d < D_MODEL; ++d) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=16
            output[(i * D_MODEL) + d] = ln_out[d];
        }
    }
}
#endif // FFNNORMBLOCK_H