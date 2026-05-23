#ifndef ROTARYEMBEDDING_H
#define ROTARYEMBEDDING_H

#include "rope_tables.h"
  // Forward Pass — uses precomputed ROM tables (half-ROPE_HALF_DIM, symmetry exploited)
  // input: [SEQ_LEN][ROPE_HALF_DIM]
  // output: [SEQ_LEN][ROPE_HALF_DIM]
void rope_forward(const data_t input[], data_t output[]) 
{
  #pragma HLS INLINE

  // Reshape ONLY the inner dimension (dim=2) to match our 8-lane highway!
  #pragma HLS ARRAY_PARTITION variable=ROPE_COS_TABLE cyclic factor=8 dim=2
  #pragma HLS ARRAY_PARTITION variable=ROPE_SIN_TABLE cyclic factor=8 dim=2

  const int half_dim = ROPE_HALF_DIM; // 64

  ROPE_TIME:
  for (int t = 0; t < SEQ_LEN; ++t) 
  {
    ROPE_DIM:
    for (int i = 0; i < D_MODEL; ++i) 
    {
      #pragma HLS PIPELINE II=2
      // #pragma HLS UNROLL factor=8

      int global_idx = (t * D_MODEL) + i;
      data_t x_val = input[global_idx];

      int lut_idx = (i < half_dim) ? i : (i - half_dim);
      
      // Clean, bulletproof 2D access
      data_t cos_val = (data_t)ROPE_COS_TABLE[t][lut_idx];
      data_t sin_val = (data_t)ROPE_SIN_TABLE[t][lut_idx];

      data_t rot_val;
      if (i < half_dim) {
          rot_val = (data_t)(-input[global_idx + half_dim]);
      } else {
          rot_val = input[global_idx - half_dim];
      }

      output[global_idx] = (x_val * cos_val) + (rot_val * sin_val);
    }
  }
}

#endif // ROTARYEMBEDDING_H
