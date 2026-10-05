#ifndef CONV1DLAYER_H
#define CONV1DLAYER_H
#include "types.h"
#include "config.h"
#include "./weights/conv1d_weights.h"

void conv1d_forward(const data_t input[], data_t output[])
{
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=conv_w_flat cyclic factor=12 dim=1
#pragma HLS ARRAY_PARTITION variable=conv_b_flat complete dim=1


    CONV_TIME:
    for (int t = 0; t < SEQ_LEN; ++t) {
        
        data_t window[IN_CHANNELS][KERNEL_SIZE];
#pragma HLS ARRAY_PARTITION variable=window complete dim=0

        CONV_FILL_WINDOW:
        for (int i = 0; i < IN_CHANNELS; ++i) {
            for (int k = 0; k < KERNEL_SIZE; ++k) {
            #pragma HLS PIPELINE II=1
                int t_offset = t + k - (KERNEL_SIZE / 2);
                
                // HARDWARE FIX: Prevent BRAM Index -1 Segfault
                int safe_t = (t_offset >= 0 && t_offset < SEQ_LEN) ? t_offset : 0;
                data_t raw_val = input[(i * SEQ_LEN) + safe_t];
                
                window[i][k] = (t_offset >= 0 && t_offset < SEQ_LEN) ? raw_val : (data_t)0;
            }
        }

        CONV_OUT_CH:
        for (int o = 0; o < OUT_CHANNELS; ++o) {
        #pragma HLS PIPELINE II=1
            
            conv_sum_t sum = (conv_sum_t)conv_b_flat[o];
            
            CONV_IN_CH:
            for (int i = 0; i < IN_CHANNELS; ++i) {
                int w_base_idx = (o * IN_CHANNELS * KERNEL_SIZE) + (i * KERNEL_SIZE);
                
                CONV_KERNEL:
                for (int k = 0; k < KERNEL_SIZE; ++k) {
                    // HARDWARE FIX: No explicit casting on operands. Let the multiplier run wide.
                    sum += window[i][k] * conv_w_flat[w_base_idx + k];
                }
            }
            output[(t * OUT_CHANNELS) + o] = (data_t)sum;
        }
    }
}
#endif // CONV1DLAYER_H