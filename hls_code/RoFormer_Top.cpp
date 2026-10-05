#include "Conv1dLayer.h"
#include "FCLayer.h"
#include "FFNorm.h"
#include "Pooling.h"
#include "MHANorm.h"
#include "types.h"
#include "config.h"
// #include "config_debug.h"
#include "./weights/fc_weights.h"

#define MAX_DEBUG_OUT (SEQ_LEN * D_MODEL)  // 25600

void RoFormer_Top(
    const data_t input_axi[],
    data_t       output_axi[]
)
{
#pragma HLS INTERFACE mode=m_axi port=input_axi  bundle=gmem0 depth=2400          offset=slave
#pragma HLS INTERFACE mode=m_axi port=output_axi bundle=gmem1 depth=MAX_DEBUG_OUT offset=slave
#pragma HLS INTERFACE mode=s_axilite port=input_axi
#pragma HLS INTERFACE mode=s_axilite port=output_axi
#pragma HLS INTERFACE mode=s_axilite port=return

    // MAGIC FIX 1: Ping-Pong Buffers
    data_t act_A[D_MODEL * SEQ_LEN];
    data_t act_B[D_MODEL * SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=act_A cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=act_B cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=fc_w_flat complete dim=1
#pragma HLS ARRAY_PARTITION variable=fc_b_flat complete dim=1


    // MAGIC FIX 2: Safe AXI reads (No out-of-bounds pointer evaluation)
    READ_INPUT_VALID:
    for (int i = 0; i < IN_CHANNELS * SEQ_LEN; ++i) {
#pragma HLS PIPELINE II=1
        act_A[i] = input_axi[i];
    }

    // Pad the rest of the first buffer with zeros
    READ_INPUT_PAD:
    for (int i = IN_CHANNELS * SEQ_LEN; i < D_MODEL * SEQ_LEN; ++i) {
#pragma HLS PIPELINE II=1
        act_A[i] = 0;
    }

    // ==========================================
    // LAYER PIPELINE (A -> B -> A -> B)
    // ==========================================

    // Layer 1: Reads A, Writes B (Input is safe!)
    conv1d_forward(act_A, act_B);

// #if defined(DEBUG_CONV1D)
//     DEBUG_WRITE:
//     for (int i = 0; i < SEQ_LEN * OUT_CHANNELS; ++i) {
// #pragma HLS PIPELINE II=1
//         output_axi[i] = act_B[i];
//     }
//     return;
// #endif


    // Layer 3: Reads A, Writes B
    mha_norm_forward(act_B, act_A);

// #if defined(DEBUG_MHA)
//     DEBUG_WRITE:
//     for (int i = 0; i < SEQ_LEN * D_MODEL; ++i) {
// #pragma HLS PIPELINE II=1
//         output_axi[i] = act_A[i];
//     }
//     return;
// #endif

    // Layer 4: Reads B, Writes A
    ffn_norm_forward(act_A, act_B);

// #if defined(DEBUG_FFN)
//     DEBUG_WRITE:
//     for (int i = 0; i < SEQ_LEN * D_MODEL; ++i) {
// #pragma HLS PIPELINE II=1
//         output_axi[i] = act_B[i];
//     }
//     return;
// #endif

    // Layer 5: Reads A, Writes to local pooled array
    data_t pooled[D_MODEL];
    pooling(act_B, pooled);

// #if defined(DEBUG_POOL)
//     DEBUG_WRITE:
//     for (int i = 0; i < D_MODEL; ++i) {
// #pragma HLS PIPELINE II=1
//         output_axi[i] = pooled[i];
//     }
//     return;
// #endif

    // Layer 6: FC Layer
    data_t local_output[OUT_DIM];
    linear2(pooled, local_output, fc_w_flat, fc_b_flat, D_MODEL, OUT_DIM);

    TOP_WRITE_OUTPUT:
    for (int i = 0; i < OUT_DIM; ++i) {
#pragma HLS PIPELINE II=1
        output_axi[i] = local_output[i];
    }
}