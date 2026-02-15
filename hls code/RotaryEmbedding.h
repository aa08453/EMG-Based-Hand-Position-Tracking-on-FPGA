#ifndef ROTARYEMBEDDING_H
#define ROTARYEMBEDDING_H

#include "rope_tables.h"

// T: Data type
// DIM: Dimension of the embedding (d_model)
// MAX_SEQ_LEN: Maximum sequence length
template <typename T, int DIM, int MAX_SEQ_LEN> class RotaryEmbedding {
public:
  RotaryEmbedding() {}

  // Forward Pass — uses precomputed ROM tables (no runtime trig)
  // input: [SEQ_LEN][DIM]
  // output: [SEQ_LEN][DIM]
  void rope_forward(const T input[MAX_SEQ_LEN][DIM],
                    T output[MAX_SEQ_LEN][DIM]) {
#pragma HLS INLINE off

    int half_dim = DIM / 2;

  ROPE_TIME:
    for (int t = 0; t < MAX_SEQ_LEN; ++t) {
#pragma HLS PIPELINE
    ROPE_DIM:
      for (int i = 0; i < DIM; ++i) {

        T x_val = input[t][i];
        T cos_val = (T)ROPE_COS_TABLE[t][i];
        T sin_val = (T)ROPE_SIN_TABLE[t][i];

        T rot_val;

        // rotate_half: if i < half_dim: -x[i+half], else: x[i-half]
        if (i < half_dim) {
          rot_val = -input[t][i + half_dim];
        } else {
          rot_val = input[t][i - half_dim];
        }

        output[t][i] = (x_val * cos_val) + (rot_val * sin_val);
      }
    }
  }
};

#endif // ROTARYEMBEDDING_H
