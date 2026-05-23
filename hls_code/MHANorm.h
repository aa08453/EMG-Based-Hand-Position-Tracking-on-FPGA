#ifndef MHANORM_H
#define MHANORM_H
#include "FCLayer.h"
#include "Softmax.h"
#include "LayerNorm.h"
#include "RotaryEmbedding.h"
#include "config.h"
#include "./weights/l0_mha_weights.h"  // <-- weights live here now
#include "./weights/l0_ln_weights.h"
#include "config_debug.h"


void linear_head_proj(const data_t input[], mha_sum_t output[],
                      const data_t weights[], const data_t bias[],
                      int head)
{
#pragma HLS INLINE off
    int base = head * HEAD_DIM;

    FC_HEAD_OUTPUT:
    for (int o = 0; o < HEAD_DIM; ++o) {
        mha_sum_t sum = (mha_sum_t)bias[base + o];
        FC_HEAD_MAC:
        for (int i = 0; i < D_MODEL; ++i) {
#pragma HLS PIPELINE II=1
            sum += (mha_sum_t)input[i] * (mha_sum_t)weights[(base + o) * D_MODEL + i];
        }
        output[o] = sum;
    }
}


void mha_norm_forward(const data_t input[], data_t output[])
{
#pragma HLS INLINE off
    const mha_sum_t scale = 0.25;

    // --- SMALL per-head buffers instead of full D_MODEL-wide ---
    mha_sum_t k_head[SEQ_LEN * HEAD_DIM];   // 3200 vs 25600
    mha_sum_t v_head[SEQ_LEN * HEAD_DIM];   // 3200 vs 25600
#pragma HLS ARRAY_RESHAPE variable=k_head cyclic factor=8 dim=1
#pragma HLS ARRAY_RESHAPE variable=v_head cyclic factor=8 dim=1

    // Concatenated attention output — one full array instead of two
    mha_sum_t attn_concat[SEQ_LEN * D_MODEL];
#pragma HLS ARRAY_RESHAPE variable=attn_concat cyclic factor=8 dim=1

#pragma HLS ARRAY_PARTITION variable=l0_ln1_g_flat complete dim=1
#pragma HLS ARRAY_PARTITION variable=l0_ln1_b_flat complete dim=1
#pragma HLS ARRAY_PARTITION variable=l0_q_b_flat complete dim=1
#pragma HLS ARRAY_PARTITION variable=l0_k_b_flat complete dim=1
#pragma HLS ARRAY_PARTITION variable=l0_v_b_flat complete dim=1
#pragma HLS ARRAY_PARTITION variable=l0_o_b_flat complete dim=1

    data_t rope_out[SEQ_LEN * D_MODEL];
    rope_forward(input, rope_out);

    // ===== HEAD-OUTER LOOP =====
    MHA_HEAD_LOOP:
    for (int h = 0; h < NUM_HEADS; ++h) {

        // Project K and V for ALL positions, but only HEAD_DIM outputs for this head
        MHA_PROJECT_KV_HEAD:
        for (int t = 0; t < SEQ_LEN; ++t) {
            linear_head_proj(&rope_out[t * D_MODEL], &k_head[t * HEAD_DIM],
                             l0_k_w_flat, l0_k_b_flat, h);
            linear_head_proj(&input[t * D_MODEL],    &v_head[t * HEAD_DIM],
                             l0_v_w_flat, l0_v_b_flat, h);
        }

        // Query loop for this head
        MHA_QUERY_LOOP:
        for (int i = 0; i < SEQ_LEN; ++i) {
            mha_sum_t q_head[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=q_head complete dim=1

            linear_head_proj(&rope_out[i * D_MODEL], q_head,
                             l0_q_w_flat, l0_q_b_flat, h);

            // Dot product: q_head[HEAD_DIM] x k_head[j][HEAD_DIM]
            mha_sum_t row_scores[SEQ_LEN];
            MHA_DOT_PRODUCT:
            for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE II=1
                mha_sum_t dot = 0;
                MHA_DOT_DIM:
                for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL
                    dot += (mha_sum_t)q_head[d] * (mha_sum_t)k_head[j * HEAD_DIM + d];
                }
                row_scores[j] = dot * scale;
            }

            mha_sum_t row_probs[SEQ_LEN];
            softmax_wide(row_scores, row_probs);

            // Weighted sum
            mha_sum_t head_acc[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=head_acc complete dim=1
            MHA_ACC_RESET:
            for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL
                head_acc[d] = 0;
            }

            MHA_WEIGHTED_SUM_SEQ:
            for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE II=1
                MHA_WEIGHTED_SUM_DIM:
                for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL
                    head_acc[d] += (mha_sum_t)row_probs[j] * (mha_sum_t)v_head[j * HEAD_DIM + d];
                }
            }

            // Store into concatenated buffer
            MHA_ACC_WRITEBACK:
            for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL
                attn_concat[i * D_MODEL + h * HEAD_DIM + d] = head_acc[d];
            }
        } // end query loop
    } // end head loop

    // ===== OUTPUT PROJECTION + RESIDUAL + LAYERNORM =====
    MHA_OUTPUT_LOOP:
    for (int i = 0; i < SEQ_LEN; ++i) {
        mha_sum_t mha_out[D_MODEL];
        linear_wide3(&attn_concat[i * D_MODEL], mha_out, l0_o_w_flat, l0_o_b_flat, D_MODEL, D_MODEL);

        data_t ln_in[D_MODEL], ln_out[D_MODEL];
        BLOCK_RESIDUAL1:
        for (int d = 0; d < D_MODEL; ++d) {
#pragma HLS PIPELINE II=1
            ln_in[d] = input[i * D_MODEL + d] + (data_t)mha_out[d];
        }
        ln_forward(ln_in, ln_out, l0_ln1_g_flat, l0_ln1_b_flat);
        BLOCK_LN1_COPY_OUT:
        for (int d = 0; d < D_MODEL; ++d) {
#pragma HLS PIPELINE II=1
            output[i * D_MODEL + d] = ln_out[d];
        }
    }
}


#endif // MHANORM_H
