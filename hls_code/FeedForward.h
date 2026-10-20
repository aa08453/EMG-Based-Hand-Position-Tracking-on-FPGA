#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H
#include "types.h"
#include "config.h"
#include "FCLayer.h"

// One token: hidden = ReLU(quantise(W1 x + b1)) in data_t; out = W2 hidden + b2 in resid_t.
// `layer` selects the block of the stacked weight arrays.
void ff_forward(const data_t input[], int off, resid_t output[D_MODEL], int layer,
                const wrom_t w1[], const wrom_t b1[], const wrom_t w2[], const wrom_t b2[])
{
#pragma HLS INLINE
    data_t hidden[D_FF];
#pragma HLS ARRAY_PARTITION variable=hidden cyclic factor=MAC_UNROLL dim=1
    dense<D_MODEL, D_FF, true>(input, off, hidden, 0, w1, b1, layer);
    dense<D_FF, D_MODEL, false>(hidden, 0, output, 0, w2, b2, layer);
}
#endif // FEEDFORWARD_H
