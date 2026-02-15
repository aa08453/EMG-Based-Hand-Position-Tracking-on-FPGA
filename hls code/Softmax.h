#ifndef SOFTMAX_H
#define SOFTMAX_H

#ifdef __SYNTHESIS__
#include <hls_math.h>
#else
#include "mock_hls_math.h"
#endif

#include "types.h"

// Piecewise linear exp approximation for range [-8, 0]
// 65 LUT entries (64 intervals), step = 0.125
// Values below -8 clamp to 0 (exp(-8) ≈ 0.000335, negligible for softmax)
template <typename T> T exp_approx(T x) {
#pragma HLS INLINE
#pragma HLS PIPELINE

  const int LUT_SIZE = 64;

  // Pre-computed exp(-8 + i*0.125) for i = 0..64
  static const data_t lut[LUT_SIZE + 1] = {
      0.00033546f, 0.00038013f, 0.00043074f, 0.00048810f, 0.00055308f,
      0.00062673f, 0.00071017f, 0.00080473f, 0.00091188f, 0.00103330f,
      0.00117088f, 0.00132678f, 0.00150344f, 0.00170362f, 0.00193045f,
      0.00218749f, 0.00247875f, 0.00280879f, 0.00318278f, 0.00360656f,
      0.00408677f, 0.00463092f, 0.00524752f, 0.00594622f, 0.00673795f,
      0.00763509f, 0.00865170f, 0.00980366f, 0.01110900f, 0.01258814f,
      0.01426423f, 0.01616349f, 0.01831564f, 0.02075434f, 0.02351775f,
      0.02664910f, 0.03019738f, 0.03421812f, 0.03877421f, 0.04393693f,
      0.04978707f, 0.05641614f, 0.06392786f, 0.07243976f, 0.08208500f,
      0.09301449f, 0.10539922f, 0.11943297f, 0.13533528f, 0.15335497f,
      0.17377394f, 0.19691168f, 0.22313016f, 0.25283960f, 0.28650480f,
      0.32465247f, 0.36787944f, 0.41686202f, 0.47236655f, 0.53526143f,
      0.60653066f, 0.68728928f, 0.77880078f, 0.88249690f, 1.00000000f};

  // Clamp to valid range
  if (x <= (T)(-8.0))
    return (T)0.0;
  if (x >= (T)(0.0))
    return (T)1.0;

  // Map x from [-8, 0] to index [0, 64]
  T x_shifted = x + (T)8.0;
  T idx_f = x_shifted * (T)8.0; // multiply is cheaper than divide

  int idx = (int)(float)idx_f;
  if (idx < 0)
    idx = 0;
  if (idx >= LUT_SIZE)
    idx = LUT_SIZE - 1;

  // Linear interpolation
  T frac = idx_f - (T)idx;
  T y0 = (T)lut[idx];
  T y1 = (T)lut[idx + 1];

  return y0 + frac * (y1 - y0);
}

template <typename T, int DIM> class Softmax {
public:
  void softmax_forward(const T input[DIM], T output[DIM]) {
#pragma HLS INLINE off

    // ---- PASS 1: Find max (numerical stability)
    T max_val = input[0];
  SOFTMAX_FIND_MAX:
    for (int i = 1; i < DIM; ++i) {
#pragma HLS PIPELINE
      if (input[i] > max_val)
        max_val = input[i];
    }

    // ---- PASS 2: Calculate exp (LUT) and accumulate sum
    T sum = 0;

  SOFTMAX_EXP_SUM:
    for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE
      T x = input[i] - max_val;
      T exp_x = exp_approx(x);

      output[i] = exp_x;
      sum += exp_x;
    }

    // ---- PASS 3: Normalize in-place
    T inv_sum = (T)1 / sum;
  SOFTMAX_NORMALIZE:
    for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE
      output[i] = output[i] * inv_sum;
    }
  }
};

#endif
