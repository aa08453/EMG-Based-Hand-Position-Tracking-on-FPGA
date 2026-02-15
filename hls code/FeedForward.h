#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "FCLayer.h"

// T: Data type
// D_MODEL: Input/Output dimension
// D_FF: Hidden dimension
template <typename T, int D_MODEL, int D_FF> class FeedForward {
private:
  FCLayer<T, D_MODEL, D_FF> fc1;
  FCLayer<T, D_FF, D_MODEL> fc2;

  // Buffer for hidden state
  T hidden[D_FF];

public:
  FeedForward() {}

  // Setters
  void set_fc1_weights(const T w[D_FF][D_MODEL], const T b[D_FF]) {
    fc1.setWeights(w);
    fc1.setBias(b);
  }

  void set_fc2_weights(const T w[D_MODEL][D_FF], const T b[D_MODEL]) {
    fc2.setWeights(w);
    fc2.setBias(b);
  }

  // Forward pass
  // input: [D_MODEL]
  // output: [D_MODEL]
  void ff_forward(const T input[D_MODEL], T output[D_MODEL]) {
#pragma HLS INLINE off

    // 1. FC1: D_MODEL -> D_FF
    fc1.forward(input, hidden);

  // 2. ReLU (inline): y = max(0, x)
  FF_RELU:
    for (int i = 0; i < D_FF; ++i) {
#pragma HLS PIPELINE
      if (hidden[i] < (T)0)
        hidden[i] = (T)0;
    }

    // 3. FC2: D_FF -> D_MODEL
    fc2.forward(hidden, output);
  }
};

#endif // FEEDFORWARD_H
