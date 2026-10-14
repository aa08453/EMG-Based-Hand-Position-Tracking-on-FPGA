#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "types.h"
#include "config.h"
#include <hls_math.h>

// Row softmax over SEQ_LEN scores. exp is still a float operator (hls::exp); a fixed-point
// exp table is a later optimisation (PROJECT_REPORT.md, section 10.4).
void softmax_row(const acc_t scores[SEQ_LEN], acc_t row_max, prob_t probs[SEQ_LEN])
{
#pragma HLS INLINE
    prob_t e[SEQ_LEN];
    acc_t sum = 0;
    SM_EXP:
    for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE II=1
        e[j] = (prob_t)hls::exp((float)(scores[j] - row_max));
        sum += (acc_t)e[j];
    }
    const ap_ufixed<24, 1> inv = (ap_ufixed<24, 1>)(1.0f / (float)sum);   // sum >= 1
    SM_NORM:
    for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE II=1
        probs[j] = (prob_t)(e[j] * inv);
    }
}
#endif // SOFTMAX_H
