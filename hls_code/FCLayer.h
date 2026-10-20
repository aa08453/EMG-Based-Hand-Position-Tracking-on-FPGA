#ifndef FCLAYER_H
#define FCLAYER_H
#include "types.h"
#include "config.h"

// out[out_off + i] = act( b[i] + sum_j in[in_off + j] * w[i*IN + j] ),  i < OUT, j < IN
// `layer` selects one block of a stacked weight array (w[layer*OUT*IN ...], b[layer*OUT ...]): encoder
// layers share this hardware and differ only in the weight offset.
// One flattened pipeline: MAC_UNROLL multiply-accumulates per cycle, OUT*IN/MAC_UNROLL cycles total,
// no per-neuron pipeline refill. Inlined at every call site so the ARRAY_PARTITION pragmas on the
// global weight arrays (placed by the caller) apply. Callers partition `in` and `w` cyclic MAC_UNROLL;
// layer*OUT*IN is a multiple of MAC_UNROLL, so every lane still reads its own bank.
template <int IN, int OUT, bool RELU, typename TI, typename TO>
void dense(const TI in[], int in_off, TO out[], int out_off, const wrom_t w[], const wrom_t b[], int layer = 0)
{
#pragma HLS INLINE
    const int STEPS = IN / MAC_UNROLL;
    const int w_off = layer * (OUT * IN), b_off = layer * OUT;
    acc_t sum = 0;
    DENSE_MAC:
    for (int n = 0; n < OUT * STEPS; ++n) {
#pragma HLS PIPELINE II=1
        const int i = n / STEPS;
        const int j = (n % STEPS) * MAC_UNROLL;
        acc_t partial = 0;
        DENSE_LANE:
        for (int u = 0; u < MAC_UNROLL; ++u) {
#pragma HLS UNROLL
            partial += (acc_t)((data_t)in[in_off + j + u] * (weight_t)w[w_off + i * IN + j + u]);
        }
        const acc_t s = (j == 0 ? (acc_t)(weight_t)b[b_off + i] : sum) + partial;
        sum = s;
        if (j == IN - MAC_UNROLL) {
            acc_t y = s;
            if (RELU) {
                const TO yq = (TO)y;                  // quantise, then ReLU (QAT order)
                out[out_off + i] = (yq < (TO)0) ? (TO)0 : yq;
            } else {
                out[out_off + i] = (TO)y;
            }
        }
    }
}

#endif // FCLAYER_H
