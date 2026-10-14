// RoFormer_Top (v2): sEMG window -> 10 absolute joint angles + 10 standardised 25 ms deltas.
//   input_axi : IN_WORDS = 12 x 200 data_t, channel-major (channel 0 samples 0..199, then channel 1, ...)
//   output_axi: NUM_OUTPUTS = [ abs angle (10) | delta / DELTA_SCALE (10) ]
// Firmware multiplies the delta outputs by DELTA_SCALE (normalization_constants.h) and fuses
// est += delta; est += K * (abs - est), K ~ 0.056 (PROJECT_REPORT.md, section 7).
#include "types.h"
#include "config.h"
#include "Conv1dLayer.h"
#include "MHANorm.h"
#include "FFNorm.h"
#include "Pooling.h"
#include "FCLayer.h"
#include "./weights/fc_weights.h"

void RoFormer_Top(const data_t input_axi[IN_WORDS], data_t output_axi[NUM_OUTPUTS])
{
#pragma HLS INTERFACE mode=m_axi port=input_axi  bundle=gmem0 depth=IN_WORDS    offset=slave
#pragma HLS INTERFACE mode=m_axi port=output_axi bundle=gmem1 depth=NUM_OUTPUTS offset=slave
#pragma HLS INTERFACE mode=s_axilite port=input_axi
#pragma HLS INTERFACE mode=s_axilite port=output_axi
#pragma HLS INTERFACE mode=s_axilite port=return
#pragma HLS ARRAY_PARTITION variable=fc_w_flat cyclic factor=MAC_UNROLL dim=1
#if HAS_DELTA_HEAD
#pragma HLS ARRAY_PARTITION variable=fc_delta_w_flat cyclic factor=MAC_UNROLL dim=1
#endif

    data_t in_buf[IN_WORDS];
    data_t act_A[SEQ_LEN * D_MODEL];
    data_t act_B[SEQ_LEN * D_MODEL];
    data_t pooled[D_MODEL];
    data_t result[NUM_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=act_A cyclic factor=MAC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=act_B cyclic factor=MAC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=pooled cyclic factor=MAC_UNROLL dim=1

    READ_INPUT:
    for (int i = 0; i < IN_WORDS; ++i) {
#pragma HLS PIPELINE II=1
        in_buf[i] = input_axi[i];
    }

    conv1d_forward(in_buf, act_A);      // [12 x 200] -> [25 x 128]
    mha_norm_forward(act_A, act_B);     // attention + residual + LN1
    ffn_norm_forward(act_B, act_A);     // FFN + residual + LN2
    pooling(act_A, pooled);             // mean over tokens

    dense<D_MODEL, OUT_DIM, false>(pooled, 0, result, 0, fc_w_flat, fc_b_flat);
#if HAS_DELTA_HEAD
    dense<D_MODEL, OUT_DIM, false>(pooled, 0, result, OUT_DIM, fc_delta_w_flat, fc_delta_b_flat);
#endif

    WRITE_OUTPUT:
    for (int i = 0; i < NUM_OUTPUTS; ++i) {
#pragma HLS PIPELINE II=1
        output_axi[i] = result[i];
    }
}
