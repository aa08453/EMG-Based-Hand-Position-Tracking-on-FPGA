// Include I/O and dump path only for C-Simulation
// #ifndef __SYNTHESIS__
// #include <iostream>
// #include <fstream>
// #include "dump_path.h"
// #endif


#ifndef POOLING_H
#define POOLING_H
#include "types.h"
#include "config.h"

void pooling(const data_t buffer_seq1[], data_t pooled[])
{
  mha_sum_t pooled_acc[D_MODEL];
  #pragma HLS ARRAY_PARTITION variable=pooled_acc cyclic factor=8 dim=1

 POOL_RESET:
  for (int d = 0; d < D_MODEL; ++d) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=8
    pooled_acc[d] = 0;
  }

  // Accumulate safely without clamping at 3.99
  ROFORMER_POOL_SEQ:
  for (int t = 0; t < SEQ_LEN; ++t) {
    ROFORMER_POOL_DIM:
    for (int d = 0; d < D_MODEL; ++d) {
      #pragma HLS PIPELINE II=1
      #pragma HLS UNROLL factor=8
    pooled_acc[d] += buffer_seq1[t * D_MODEL + d];
    }
  }

  const pool_scale_t inv_seq_len = (pool_scale_t)(1.0f / SEQ_LEN);

  ROFORMER_POOL_SCALE:
  for (int d = 0; d < D_MODEL; ++d) {
    #pragma HLS PIPELINE II=1
    #pragma HLS UNROLL factor=8
    
    // The multiplication happens natively, then safely casts back to data_t
    pooled[d] = (data_t)(pooled_acc[d] * inv_seq_len);
  }

}
#endif