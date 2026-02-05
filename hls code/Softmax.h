#ifndef SOFTMAX_H
#define SOFTMAX_H

#ifdef __SYNTHESIS__
#include <hls_math.h>
#else
#include "hls_math.h"
#endif

template <typename T, int DIM> class Softmax {
public:
  void forward(const T input[DIM], T output[DIM]) {
#pragma HLS INLINE off

    // ---- PASS 1: Find max (numerical stability)
    T max_val = input[0];
    for (int i = 1; i < DIM; ++i) {
#pragma HLS PIPELINE
      if (input[i] > max_val)
        max_val = input[i];
    }

    // ---- PASS 2: Calculate exp and accumulate sum
    T sum = 0;
    T exp_buf[DIM];
#pragma HLS BIND_STORAGE variable = exp_buf type = ram_1p impl = bram

    for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE
      T x = input[i] - max_val;
      T exp_x = hls::exp(x);

      exp_buf[i] = exp_x;
      sum += exp_x;
    }

    // ---- PASS 3: Normalize
    T inv_sum = (T)1 / sum;
    for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE
      output[i] = exp_buf[i] * inv_sum;
    }
  }
};

#endif
