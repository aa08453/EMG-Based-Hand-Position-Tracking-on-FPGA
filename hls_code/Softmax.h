#ifndef SOFTMAX_H
#define SOFTMAX_H

#include "config.h"
#include "types.h"

#include <hls_math.h>

void softmax_wide(const mha_sum_t input[], mha_sum_t output[]) {
    // find max in mha_sum_t
    mha_sum_t max_val = input[0];
// #pragma HLS pipeline     
    for (int i = 1; i < SEQ_LEN; ++i)
        if (input[i] > max_val) max_val = input[i];

    mha_sum_t sum = 0;
    for (int i = 0; i < SEQ_LEN; ++i) {
        mha_sum_t diff = input[i] - max_val;
        mha_sum_t exp_x = (mha_sum_t)hls::exp((float)diff);
        output[i] = exp_x;
        sum += exp_x;
    }
    mha_sum_t inv_sum = (mha_sum_t)(1.0f / (float)sum);
    for (int i = 0; i < SEQ_LEN; ++i)
        output[i] = output[i] * inv_sum;
}

#endif