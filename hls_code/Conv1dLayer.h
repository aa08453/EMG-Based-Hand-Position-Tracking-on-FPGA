#ifndef CONV1DLAYER_H
#define CONV1DLAYER_H
#include "types.h"
#include "config.h"
#include "./weights/conv1d_weights.h"

// Strided embedding: input [IN_CHANNELS][RAW_SEQ_LEN] channel-major (the AXI layout),
// output [SEQ_LEN][D_MODEL]. out[t][o] = b[o] + sum_c sum_k w[o][c][k] * x[c][t*STRIDE + k - PAD].
// void conv1d_forward(const data_t input[IN_WORDS], data_t output[SEQ_LEN * D_MODEL])
// {
// #pragma HLS INLINE off
// #pragma HLS ARRAY_PARTITION variable=conv_w_flat cyclic factor=KERNEL_SIZE dim=1

//     CONV_TOKEN:
//     for (int t = 0; t < SEQ_LEN; ++t) {
//         data_t window[IN_CHANNELS][KERNEL_SIZE];
// #pragma HLS ARRAY_PARTITION variable=window complete dim=2

//         CONV_FILL:
//         for (int c = 0; c < IN_CHANNELS; ++c) {
//             for (int k = 0; k < KERNEL_SIZE; ++k) {
// #pragma HLS PIPELINE II=1
//                 const int s = t * CONV_STRIDE + k - CONV_PADDING;
//                 const bool ok = (s >= 0) && (s < RAW_SEQ_LEN);
//                 window[c][k] = ok ? input[c * RAW_SEQ_LEN + s] : (data_t)0;
//             }
//         }

//         acc_t sum = 0;
//         CONV_MAC:
//         for (int n = 0; n < D_MODEL * IN_CHANNELS; ++n) {
// #pragma HLS PIPELINE II=1
//             const int o = n / IN_CHANNELS;
//             const int c = n % IN_CHANNELS;
//             acc_t partial = 0;
//             for (int k = 0; k < KERNEL_SIZE; ++k) {
// #pragma HLS UNROLL
//                 partial += (acc_t)(window[c][k] * (weight_t)conv_w_flat[(o * IN_CHANNELS + c) * KERNEL_SIZE + k]);
//             }
//             const acc_t s = (c == 0 ? (acc_t)(weight_t)conv_b_flat[o] : sum) + partial;
//             sum = s;
//             if (c == IN_CHANNELS - 1) output[t * D_MODEL + o] = (data_t)s;
//         }
//     }
// }





// void conv1d_forward(const data_t input[IN_CHANNELS * RAW_SEQ_LEN],  data_t output[SEQ_LEN * D_MODEL]) {
// #pragma HLS INLINE off

//     // Outer loop: Iterate over each output feature channel (D_MODEL)
//     // For each output channel, the 192 weights remain stationary while all 25 tokens stream past.
//     OUT_CHANNEL_LOOP:
//     for (int o = 0; o < D_MODEL; ++o) {

//         // Cache or sequentially access the 192 weights for this output channel
//         weight_t channel_weights[IN_CHANNELS * KERNEL_SIZE];
//         #pragma HLS ARRAY_PARTITION variable=channel_weights complete

//         LOAD_WEIGHTS:
//         for (int i = 0; i < IN_CHANNELS * KERNEL_SIZE; ++i) {
//             #pragma HLS PIPELINE II=1
//             int c = i / KERNEL_SIZE;
//             int k = i % KERNEL_SIZE;
//             channel_weights[i] = conv_w_flat[(o * IN_CHANNELS + c) * KERNEL_SIZE + k];
//         }

//         data_t bias = conv_b_flat[o];

//         // Inner loop: Stream all 25 tokens through the stationary weights
//         TOKEN_LOOP:
//         for (int t = 0; t < SEQ_LEN; ++t) {
//             #pragma HLS PIPELINE II=1

//             // Local buffers fully partitioned into registers
//             data_t input_window[IN_CHANNELS * KERNEL_SIZE];
//             #pragma HLS ARRAY_PARTITION variable=input_window complete

//             GATHER_INPUTS:
//             for (int i = 0; i < IN_CHANNELS * KERNEL_SIZE; ++i) {
//                 #pragma HLS UNROLL
//                 int c = i / KERNEL_SIZE;
//                 int k = i % KERNEL_SIZE;
//                 int s = t * CONV_STRIDE + k;
//                 const bool ok = (s >= 0) && (s < RAW_SEQ_LEN);
//                 input_window[i] = ok ? input[c * RAW_SEQ_LEN + s] : (data_t)0;
//             }

//             // Step 2: Parallel Multiply (192 multipliers)
//             acc_t products[IN_CHANNELS * KERNEL_SIZE];
//             #pragma HLS ARRAY_PARTITION variable=products complete

//             MULTIPLY_ALL:
//             for (int i = 0; i < IN_CHANNELS * KERNEL_SIZE; ++i) {
//                 #pragma HLS UNROLL
//                 products[i] = (acc_t)(input_window[i] * channel_weights[i]);
//             }

//             // Step 3: Pipelined Adder Tree Reduction (192 -> 1)
//             acc_t tree_sum = 0;
//             REDUCE_TREE:
//             for (int i = 0; i < IN_CHANNELS * KERNEL_SIZE; ++i) {
//                 #pragma HLS UNROLL
//                 tree_sum += products[i];
//             }

//             // Step 4: Add bias and write out in correct output layout order [t * D_MODEL + o]
//             output[t * D_MODEL + o] = (data_t)(tree_sum + (acc_t)bias);
//         }
//     }
// }

#include <hls_stream.h>

// Sub-process 1: Feeds weights into a FIFO stream
void load_conv_weights(hls::stream<weight_t> &weight_fifo, hls::stream<data_t> &bias_fifo) {
#pragma HLS INLINE off
    LOAD_OUT_CH:
    for (int o = 0; o < D_MODEL; ++o) {
#pragma HLS PIPELINE II=1
        bias_fifo.write(conv_b_flat[o]);
        
        LOAD_W_INNER:
        for (int i = 0; i < IN_CHANNELS * KERNEL_SIZE; ++i) {
#pragma HLS PIPELINE II=1
            int c = i / KERNEL_SIZE;
            int k = i % KERNEL_SIZE;
            weight_fifo.write(conv_w_flat[(o * IN_CHANNELS + c) * KERNEL_SIZE + k]);
        }
    }
}

// Sub-process 2: Consumes weights from FIFO and processes all 25 tokens
void compute_conv_tokens( const data_t input[IN_CHANNELS * RAW_SEQ_LEN], 
    hls::stream<weight_t> &weight_fifo, 
    hls::stream<data_t> &bias_fifo,
    data_t output[SEQ_LEN * D_MODEL]
) {
#pragma HLS INLINE off
    COMPUTE_OUT_CH:
    for (int o = 0; o < D_MODEL; ++o) {
        
        weight_t channel_weights[IN_CHANNELS * KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable=channel_weights complete

        READ_W:
        for (int i = 0; i < IN_CHANNELS * KERNEL_SIZE; ++i) {
#pragma HLS PIPELINE II=1
            channel_weights[i] = weight_fifo.read();
        }

        data_t bias = bias_fifo.read();

        TOKEN_LOOP:
        for (int t = 0; t < SEQ_LEN; ++t) {
#pragma HLS PIPELINE II=1

            data_t input_window[IN_CHANNELS * KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable=input_window complete

            acc_t products[IN_CHANNELS * KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable=products complete

            GATHER_INPUTS:
            for (int i = 0; i < IN_CHANNELS * KERNEL_SIZE; ++i) {
#pragma HLS UNROLL
                int c = i / KERNEL_SIZE;
                int k = i % KERNEL_SIZE;
                int s = t * CONV_STRIDE + k;
                const bool ok = (s >= 0) && (s < RAW_SEQ_LEN);
                input_window[i] = ok ? input[c * RAW_SEQ_LEN + s] : (data_t)0;
            }

            MULTIPLY_ALL:
            for (int i = 0; i < IN_CHANNELS * KERNEL_SIZE; ++i) {
#pragma HLS UNROLL
                products[i] = (acc_t)(input_window[i] * channel_weights[i]);
            }

            acc_t tree_sum = 0;
            REDUCE_TREE:
            for (int i = 0; i < IN_CHANNELS * KERNEL_SIZE; ++i) {
#pragma HLS UNROLL
                tree_sum += products[i];
            }

            output[t * D_MODEL + o] = (data_t)(tree_sum + (acc_t)bias);
        }
    }
}

// Main Conv1D Wrapper with internal DATAFLOW
void conv1d_forward(const data_t input[IN_CHANNELS * RAW_SEQ_LEN], data_t output[SEQ_LEN * D_MODEL]
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<weight_t> weight_fifo("weight_fifo");
#pragma HLS STREAM variable=weight_fifo depth=192

    hls::stream<data_t> bias_fifo("bias_fifo");
#pragma HLS STREAM variable=bias_fifo depth=2

    load_conv_weights(weight_fifo, bias_fifo);
    compute_conv_tokens(input, weight_fifo, bias_fifo, output);
}


#endif // CONV1DLAYER_H
