#ifndef FFNNORMBLOCK_H
#define FFNNORMBLOCK_H
#include "types.h"
#include "config.h"
#include "FeedForward.h"
#include "LayerNorm.h"
#include "./weights/l0_ff_weights.h"
#include "./weights/l0_ln_weights.h"

// out = LN2( x + FFN(x) ), token by token. FFN output and residual sum are resid_t.
void ffn_norm_forward(const data_t input[SEQ_LEN * D_MODEL], data_t output[SEQ_LEN * D_MODEL])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=l0_ff_w1_flat cyclic factor=MAC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=l0_ff_w2_flat cyclic factor=MAC_UNROLL dim=1
    resid_t ff_out[D_MODEL], ln_in[D_MODEL];
    FFN_TOKEN:
    for (int t = 0; t < SEQ_LEN; ++t) {
        ff_forward(input, t * D_MODEL, ff_out, l0_ff_w1_flat, l0_ff_b1_flat, l0_ff_w2_flat, l0_ff_b2_flat);
        FFN_RESID:
        for (int d = 0; d < D_MODEL; ++d) {
#pragma HLS PIPELINE II=1
            ln_in[d] = (resid_t)(input[t * D_MODEL + d] + ff_out[d]);
        }
        ln_forward<resid_t>(ln_in, 0, output, t * D_MODEL, l0_ln2_g_flat, l0_ln2_b_flat);
    }
}
#endif // FFNNORMBLOCK_H
