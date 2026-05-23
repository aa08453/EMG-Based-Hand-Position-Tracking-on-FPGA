#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "FCLayer.h"
#include "config.h"
// #include "hls_stream.h"
#include "types.h" 

void ff_forward(
    const data_t input[], // d_model
    const data_t weights1[],
    const data_t bias1[],
    const data_t weights2[],
    const data_t bias2[],
    data_t output[] // d_model
) 
{
#pragma HLS INLINE off

  // Buffer for hidden state
  data_t hidden[D_FF];

  #pragma HLS ARRAY_RESHAPE variable=hidden cyclic factor=8 dim=1



  // 1. FC1: D_MODEL -> D_FF
  linear(input, hidden, weights1, bias1, D_MODEL, D_FF);

  // 2. ReLU (inline): y = max(0, x)
  FF_RELU:
  for (int i = 0; i < D_FF; ++i) {
  #pragma HLS PIPELINE II=1
  #pragma HLS UNROLL factor=8

    // Use ternary operator for cleaner hardware multiplexing
    hidden[i] = (hidden[i] < (data_t)0) ? (data_t)0 : hidden[i];
  }

  // 3. FC2: D_FF -> D_MODEL
  linear(hidden, output, weights2, bias2, D_FF, D_MODEL);
}

#endif // FEEDFORWARD_H
