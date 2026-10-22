#ifndef MHANORM_H
#define MHANORM_H
#include "types.h"
#include "config.h"
#include "FCLayer.h"
#include "Softmax.h"
#include "LayerNorm.h"
#include "RotaryEmbedding.h"
#include "./weights/encoder_weights.h"

// out = LN1( x + O( MHA( Q(rope(x)), K(rope(x)), V(x) ) ) ) for encoder layer `layer`.
// Q, K, V are projected once for all tokens and heads (no per-head recomputation).
// Quantisation points match QAT: projections qkv_t, context data_t, O output and residual sum resid_t.
void mha_norm_forward(int layer, const data_t input[SEQ_LEN * D_MODEL], data_t output[SEQ_LEN * D_MODEL])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=enc_q_w cyclic factor=MAC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=enc_k_w cyclic factor=MAC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=enc_v_w cyclic factor=MAC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=enc_o_w cyclic factor=MAC_UNROLL dim=1

    data_t  rope_out[SEQ_LEN * D_MODEL];
    qkv_t   q_all[SEQ_LEN * D_MODEL], k_all[SEQ_LEN * D_MODEL], v_all[SEQ_LEN * D_MODEL];
    data_t  ctx[SEQ_LEN * D_MODEL];
    resid_t o_tok[D_MODEL], ln_in[D_MODEL];
#pragma HLS ARRAY_PARTITION variable=rope_out cyclic factor=MAC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=q_all cyclic factor=HEAD_DIM dim=1
#pragma HLS ARRAY_PARTITION variable=k_all cyclic factor=HEAD_DIM dim=1
#pragma HLS ARRAY_PARTITION variable=v_all cyclic factor=HEAD_DIM dim=1
#pragma HLS ARRAY_PARTITION variable=ctx cyclic factor=HEAD_DIM dim=1
#if LUTRAM_ATTN_BUFFERS
#pragma HLS BIND_STORAGE variable=q_all type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=k_all type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=v_all type=ram_1p impl=lutram
#pragma HLS BIND_STORAGE variable=ctx type=ram_1p impl=lutram
#endif

    rope_forward(input, rope_out);

    MHA_PROJ:
    for (int t = 0; t < SEQ_LEN; ++t) {
        dense<D_MODEL, D_MODEL, false>(rope_out, t * D_MODEL, q_all, t * D_MODEL, enc_q_w, enc_q_b, layer);
        dense<D_MODEL, D_MODEL, false>(rope_out, t * D_MODEL, k_all, t * D_MODEL, enc_k_w, enc_k_b, layer);
        dense<D_MODEL, D_MODEL, false>(input,    t * D_MODEL, v_all, t * D_MODEL, enc_v_w, enc_v_b, layer);
    }

    MHA_HEAD:
    for (int h = 0; h < NUM_HEADS; ++h) {
        MHA_QUERY:
        for (int i = 0; i < SEQ_LEN; ++i) {
            acc_t scores[SEQ_LEN];
            acc_t row_max = -2048;
            MHA_DOT:
            for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE II=1
                acc_t d = 0;
                for (int dd = 0; dd < HEAD_DIM; ++dd) {
#pragma HLS UNROLL
                    d += (acc_t)(q_all[i * D_MODEL + h * HEAD_DIM + dd] * k_all[j * D_MODEL + h * HEAD_DIM + dd]);
                }
                const acc_t s = d >> 2;                  // / sqrt(HEAD_DIM) = / 4
                scores[j] = s;
                row_max = (s > row_max) ? s : row_max;
            }
            prob_t p[SEQ_LEN];
            softmax_row(scores, row_max, p);

            acc_t acc[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=acc complete
            for (int dd = 0; dd < HEAD_DIM; ++dd) {
#pragma HLS UNROLL
                acc[dd] = 0;
            }
            MHA_WSUM:
            for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE II=1
                for (int dd = 0; dd < HEAD_DIM; ++dd) {
#pragma HLS UNROLL
                    acc[dd] += (acc_t)(p[j] * v_all[j * D_MODEL + h * HEAD_DIM + dd]);
                }
            }
            for (int dd = 0; dd < HEAD_DIM; ++dd) {
#pragma HLS UNROLL
                ctx[i * D_MODEL + h * HEAD_DIM + dd] = (data_t)acc[dd];
            }
        }
    }

    MHA_OUT:
    for (int t = 0; t < SEQ_LEN; ++t) {
        dense<D_MODEL, D_MODEL, false>(ctx, t * D_MODEL, o_tok, 0, enc_o_w, enc_o_b, layer);
        MHA_RESID:
        for (int d = 0; d < D_MODEL; ++d) {
#pragma HLS PIPELINE II=1
            ln_in[d] = (resid_t)(input[t * D_MODEL + d] + o_tok[d]);
        }
        ln_forward<resid_t>(ln_in, 0, output, t * D_MODEL, enc_ln1_g, enc_ln1_b, layer * D_MODEL);
    }
}
#endif // MHANORM_H
