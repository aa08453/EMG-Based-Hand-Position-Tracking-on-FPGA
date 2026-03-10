#ifndef CONV1DLAYER_H
#define CONV1DLAYER_H

#ifdef __SYNTHESIS__
#include <hls_math.h>
#else
#include "mock_hls_math.h"
#endif

// Template parameters:
// T: Data type (float, fixed-point)
// IN_CHANNELS: Number of input channels
// OUT_CHANNELS: Number of output channels
// SEQ_LEN: Length of the sequence
// KERNEL_SIZE: Kernel size (fixed to 3 for this implementation)
template <typename T, int IN_CHANNELS, int OUT_CHANNELS, int SEQ_LEN,
          int KERNEL_SIZE = 3>
class Conv1dLayer {
private:
  T weights[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE];
  T bias[OUT_CHANNELS];

public:
  Conv1dLayer() {
  CONV_INIT_OUT:
    for (int o = 0; o < OUT_CHANNELS; ++o) {
      bias[o] = 0;
    CONV_INIT_IN:
      for (int i = 0; i < IN_CHANNELS; ++i) {
      CONV_INIT_K:
        for (int k = 0; k < KERNEL_SIZE; ++k) {
          weights[o][i][k] = 0;
        }
      }
    }
  }

  void setWeights(const T w[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE]) {
  CONV_SET_W_OUT:
    for (int o = 0; o < OUT_CHANNELS; ++o) {
    CONV_SET_W_IN:
      for (int i = 0; i < IN_CHANNELS; ++i) {
      CONV_SET_W_K:
        for (int k = 0; k < KERNEL_SIZE; ++k) {
          weights[o][i][k] = w[o][i][k];
        }
      }
    }
  }

  void setBias(const T b[OUT_CHANNELS]) {
  CONV_SET_BIAS:
    for (int o = 0; o < OUT_CHANNELS; ++o) {
      bias[o] = b[o];
    }
  }

  // Forward pass
  // Input format: [IN_CHANNELS][SEQ_LEN]
  // Output format: [SEQ_LEN][OUT_CHANNELS] (Transposed for Transformer)
  void conv1d_forward(const T input[IN_CHANNELS][SEQ_LEN],
                      T output[SEQ_LEN][OUT_CHANNELS]) {
#pragma HLS INLINE off

#pragma HLS BIND_STORAGE variable = weights type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = bias type = ram_1p impl = bram

  CONV_TIME:
    for (int t = 0; t < SEQ_LEN; ++t) {

    CONV_OUT_CH:
      for (int o = 0; o < OUT_CHANNELS; ++o) {
        T sum = bias[o];

      CONV_IN_CH:
        for (int i = 0; i < IN_CHANNELS; ++i) {
#pragma HLS PIPELINE
#pragma HLS LOOP_FLATTEN off

          // k=0 (Left neighbor)
          if (t > 0) {
            sum += input[i][t - 1] * weights[o][i][0];
          }

          // k=1 (Center)
          sum += input[i][t] * weights[o][i][1];

          // k=2 (Right neighbor)
          if (t < SEQ_LEN - 1) {
            sum += input[i][t + 1] * weights[o][i][2];
          }
        }
        output[t][o] = sum;
      }
    }
  }
};

#endif // CONV1DLAYER_H
