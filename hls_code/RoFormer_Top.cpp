// RoFormer_Top: sEMG window -> 10 absolute joint angles + 10 standardised per-step deltas.
//   input_axi : IN_WORDS = 12 x RAW_SEQ_LEN data_t, channel-major (channel 0 samples 0..RAW_SEQ_LEN-1, then channel 1, ...)
//   output_axi: NUM_OUTPUTS = [ abs angle (10) | delta / DELTA_SCALE (10) ]
// NUM_LAYERS encoder layers run one after another on the same hardware; only the weight offset changes.
// Firmware multiplies the delta outputs by DELTA_SCALE (normalization_constants.h) and fuses
// est += delta; est += K * (abs - est), with K from the filter tuned for the model (PROJECT_REPORT.md, section 7).
#include "types.h"
#include "config.h"
#include "Conv1dLayer.h"
#include "MHANorm.h"
#include "FFNorm.h"
#include "Pooling.h"
#include "FCLayer.h"
#include "./weights/fc_weights.h"

// void RoFormer_Top(const data_t input_axi[IN_WORDS], data_t output_axi[NUM_OUTPUTS])
// {
// #pragma HLS INTERFACE mode=m_axi port=input_axi  bundle=gmem0 depth=IN_WORDS    offset=slave
// #pragma HLS INTERFACE mode=m_axi port=output_axi bundle=gmem1 depth=NUM_OUTPUTS offset=slave
// #pragma HLS INTERFACE mode=s_axilite port=input_axi
// #pragma HLS INTERFACE mode=s_axilite port=output_axi
// #pragma HLS INTERFACE mode=s_axilite port=return
// #pragma HLS ARRAY_PARTITION variable=fc_w_flat cyclic factor=MAC_UNROLL dim=1
// #if HAS_DELTA_HEAD
// #pragma HLS ARRAY_PARTITION variable=fc_delta_w_flat cyclic factor=MAC_UNROLL dim=1
// #endif

//     data_t in_buf[IN_WORDS];
//     data_t act_A[SEQ_LEN * D_MODEL];
//     data_t act_B[SEQ_LEN * D_MODEL];
//     data_t pooled[D_MODEL];
//     data_t result[NUM_OUTPUTS];
// #pragma HLS ARRAY_PARTITION variable=act_A cyclic factor=MAC_UNROLL dim=1
// #pragma HLS ARRAY_PARTITION variable=act_B cyclic factor=MAC_UNROLL dim=1
// #pragma HLS ARRAY_PARTITION variable=pooled cyclic factor=MAC_UNROLL dim=1

//     READ_INPUT:
//     for (int i = 0; i < IN_WORDS; ++i) {
// #pragma HLS PIPELINE II=1
//         in_buf[i] = input_axi[i];
//     }

//     conv1d_forward(in_buf, act_A);              // [12 x RAW_SEQ_LEN] -> [SEQ_LEN x 128]
//     ENCODER:
//     for (int l = 0; l < NUM_LAYERS; ++l) {
//         mha_norm_forward(l, act_A, act_B);      // attention + residual + LN1
//         ffn_norm_forward(l, act_B, act_A);      // FFN + residual + LN2
//     }
//     pooling(act_A, pooled);                     // mean over tokens

//     dense<D_MODEL, OUT_DIM, false>(pooled, 0, result, 0, fc_w_flat, fc_b_flat);
// #if HAS_DELTA_HEAD
//     dense<D_MODEL, OUT_DIM, false>(pooled, 0, result, OUT_DIM, fc_delta_w_flat, fc_delta_b_flat);
// #endif

//     WRITE_OUTPUT:
//     for (int i = 0; i < NUM_OUTPUTS; ++i) {
// #pragma HLS PIPELINE II=1
//         output_axi[i] = result[i];
//     }
// }


#include <hls_stream.h>

// Stage 1: Read incoming AXI Stream into an input buffer
void stream_to_conv(hls::stream<data_t> &in_strm, data_t out_buf[IN_WORDS]) {
#pragma HLS INLINE off
    STREAM_IN:
    for (int i = 0; i < IN_WORDS; ++i) {
#pragma HLS PIPELINE II=1
        out_buf[i] = in_strm.read();
    }
}

// Stage 2: Execute 1D Convolution
void conv_stage(const data_t in_buf[IN_WORDS], data_t out_act[SEQ_LEN * D_MODEL]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=out_act cyclic factor=MAC_UNROLL dim=1
    conv1d_forward(in_buf, out_act);
}

// Stage 3: Single encoder layer block (MHA + FFN + Residuals + LayerNorms)
void encoder_layer_proc(int l, const data_t in_act[SEQ_LEN * D_MODEL], data_t out_act[SEQ_LEN * D_MODEL]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in_act  cyclic factor=MAC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=out_act cyclic factor=MAC_UNROLL dim=1
    data_t temp[SEQ_LEN * D_MODEL];
#pragma HLS ARRAY_PARTITION variable=temp cyclic factor=MAC_UNROLL dim=1
    
    mha_norm_forward(l, in_act, temp);
    ffn_norm_forward(l, temp, out_act);
}

// Stage 4: Pipeline all encoder layers using internal dataflow buffers
void encoder_pipeline(
    const data_t act_in[SEQ_LEN * D_MODEL], 
    data_t act_out[SEQ_LEN * D_MODEL]
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=act_in  cyclic factor=MAC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=act_out cyclic factor=MAC_UNROLL dim=1

    data_t layer_acts[NUM_LAYERS + 1][SEQ_LEN * D_MODEL];
#pragma HLS ARRAY_PARTITION variable=layer_acts cyclic factor=MAC_UNROLL dim=2
#if NUM_LAYERS > 1
#pragma HLS ARRAY_PARTITION variable=layer_acts complete dim=1
#endif

    INIT_BUF:
    for (int i = 0; i < SEQ_LEN * D_MODEL; ++i) {
#pragma HLS PIPELINE II=1
        layer_acts[0][i] = act_in[i];
    }

    ENCODER_UNROLL:
    for (int l = 0; l < NUM_LAYERS; ++l) {
#pragma HLS UNROLL
        encoder_layer_proc(l, layer_acts[l], layer_acts[l + 1]);
    }

    FINI_BUF:
    for (int i = 0; i < SEQ_LEN * D_MODEL; ++i) {
#pragma HLS PIPELINE II=1
        act_out[i] = layer_acts[NUM_LAYERS][i];
    }
}

// Stage 5: Pooling, Dense layers, and AXI Stream Output writing
void run_pooling_and_fc(
    const data_t act_in[SEQ_LEN * D_MODEL],
    hls::stream<data_t> &output_stream
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=act_in cyclic factor=MAC_UNROLL dim=1

    data_t pooled[D_MODEL];
    data_t result[NUM_OUTPUTS];
#pragma HLS ARRAY_PARTITION variable=pooled cyclic factor=MAC_UNROLL dim=1

    pooling(act_in, pooled);

    dense<D_MODEL, OUT_DIM, false>(pooled, 0, result, 0, fc_w_flat, fc_b_flat);
#if HAS_DELTA_HEAD
    dense<D_MODEL, OUT_DIM, false>(pooled, 0, result, OUT_DIM, fc_delta_w_flat, fc_delta_b_flat);
#endif

    WRITE_OUTPUT:
    for (int i = 0; i < NUM_OUTPUTS; ++i) {
#pragma HLS PIPELINE II=1
        output_stream.write(result[i]);
    }
}

// Top-Level Module
void RoFormer_Top(
    hls::stream<data_t> &input_stream, 
    hls::stream<data_t> &output_stream
) {
#pragma HLS INTERFACE mode=axis port=input_stream
#pragma HLS INTERFACE mode=axis port=output_stream
#pragma HLS INTERFACE mode=s_axilite port=return

#pragma HLS ARRAY_PARTITION variable=fc_w_flat cyclic factor=MAC_UNROLL dim=1
#if HAS_DELTA_HEAD
#pragma HLS ARRAY_PARTITION variable=fc_delta_w_flat cyclic factor=MAC_UNROLL dim=1
#endif

    data_t in_buf[IN_WORDS];
#pragma HLS ARRAY_PARTITION variable=in_buf cyclic factor=192 dim=1

    data_t act_A[SEQ_LEN * D_MODEL];
    data_t act_B[SEQ_LEN * D_MODEL];
#pragma HLS ARRAY_PARTITION variable=act_A  cyclic factor=MAC_UNROLL dim=1
#pragma HLS ARRAY_PARTITION variable=act_B  cyclic factor=MAC_UNROLL dim=1

    // Enables task-level pipelining across all sub-modules via channels/buffers
#pragma HLS DATAFLOW

    stream_to_conv(input_stream, in_buf);
    conv_stage(in_buf, act_A);
    encoder_pipeline(act_A, act_B);
    run_pooling_and_fc(act_B, output_stream);
}