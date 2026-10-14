#ifndef CONV1DLAYER_H
#define CONV1DLAYER_H
#include "types.h"
#include "config.h"
#include "./weights/conv1d_weights.h"

// Strided embedding: input [IN_CHANNELS][RAW_SEQ_LEN] channel-major (the AXI layout),
// output [SEQ_LEN][D_MODEL]. out[t][o] = b[o] + sum_c sum_k w[o][c][k] * x[c][t*STRIDE + k - PAD].
void conv1d_forward(const data_t input[IN_WORDS], data_t output[SEQ_LEN * D_MODEL])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=conv_w_flat cyclic factor=KERNEL_SIZE dim=1

    CONV_TOKEN:
    for (int t = 0; t < SEQ_LEN; ++t) {
        data_t window[IN_CHANNELS][KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable=window complete dim=2

        CONV_FILL:
        for (int c = 0; c < IN_CHANNELS; ++c) {
            for (int k = 0; k < KERNEL_SIZE; ++k) {
#pragma HLS PIPELINE II=1
                const int s = t * CONV_STRIDE + k - CONV_PADDING;
                const bool ok = (s >= 0) && (s < RAW_SEQ_LEN);
                window[c][k] = ok ? input[c * RAW_SEQ_LEN + (ok ? s : 0)] : (data_t)0;
            }
        }

        acc_t sum = 0;
        CONV_MAC:
        for (int n = 0; n < D_MODEL * IN_CHANNELS; ++n) {
#pragma HLS PIPELINE II=1
            const int o = n / IN_CHANNELS;
            const int c = n % IN_CHANNELS;
            acc_t partial = 0;
            for (int k = 0; k < KERNEL_SIZE; ++k) {
#pragma HLS UNROLL
                partial += (acc_t)(window[c][k] * conv_w_flat[(o * IN_CHANNELS + c) * KERNEL_SIZE + k]);
            }
            const acc_t s = (c == 0 ? (acc_t)conv_b_flat[o] : sum) + partial;
            sum = s;
            if (c == IN_CHANNELS - 1) output[t * D_MODEL + o] = (data_t)s;
        }
    }
}
#endif // CONV1DLAYER_H
