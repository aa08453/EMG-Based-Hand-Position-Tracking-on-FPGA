#ifndef LAYERNORM_H
#define LAYERNORM_H
#ifdef __SYNTHESIS__
#include <hls_math.h>
#else
#include "mock_hls_math.h"
#include <cmath>
#include <iostream>

namespace hls {
// Fallback for rsqrt in standard C++
template <typename T> T rsqrt(T x) { return 1.0 / std::sqrt(x); }
} // namespace hls
#endif

// Template parameters:
// T: Data type
// DIM: Dimension of the input/output array
template <typename T, int DIM> class LayerNorm {
private:
  T gamma[DIM];
  T beta[DIM];
  T epsilon;

public:
  LayerNorm(T eps = (T)1e-5) : epsilon(eps) {
    // Initialize gamma to 1 and beta to 0 (params usually loaded later)
    for (int i = 0; i < DIM; ++i) {
      gamma[i] = (T)1.0;
      beta[i] = (T)0.0;
    }
  }

  void setGamma(const T g[DIM]) {
    for (int i = 0; i < DIM; ++i)
      gamma[i] = g[i];
  }

  void setBeta(const T b[DIM]) {
    for (int i = 0; i < DIM; ++i)
      beta[i] = b[i];
  }

  // Forward pass: y = ((x - mean) / sqrt(var + eps)) * gamma + beta
  void forward(T input[DIM], T output[DIM]) {
#pragma HLS INLINE off

    T sum = 0;
    T sum_sq = 0;

#pragma HLS BIND_STORAGE variable = gamma type = ram_1p impl = bram
#pragma HLS BIND_STORAGE variable = beta type = ram_1p impl = bram

    const T inv_DIM = (T)1.0 / (T)DIM;

    // 1. Calculate Mean
    for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE
      sum += input[i];
    }
    T mean = sum * inv_DIM;

    // 2. Calculate Variance
    for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE
      T diff = input[i] - mean;
      sum_sq += diff * diff;
    }
    T variance = sum_sq * inv_DIM;
    T std_dev_inv = hls::rsqrt(variance + epsilon);
    // If rsqrt is unstable, fallback:
    // T inv_std = (T)1 / hls::sqrt(var + epsilon);

    // 3. Normalize and Scale/Shift
    for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE
      T norm = (input[i] - mean) * std_dev_inv;
      output[i] = norm * gamma[i] + beta[i];
    }
  }
};

#endif // LAYERNORM_H
