#ifndef ROFORMER_H
#define ROFORMER_H

#include "Conv1dLayer.h"
#include "FCLayer.h"
#include "RoFormerBlock.h"


// Template parameters
// T: Data type
// IN_CHANNELS: e.g. 12 (sEMG channels)
// SEQ_LEN: e.g. 100 (Time steps)
// D_MODEL: e.g. 128
// NUM_HEADS: e.g. 8
// D_FF: e.g. 512
// NUM_LAYERS: Number of transformer layers/blocks
// OUT_DIM: e.g. 10 (Joint angles)

template <typename T, int IN_CHANNELS, int SEQ_LEN, int D_MODEL, int NUM_HEADS,
          int D_FF, int NUM_LAYERS, int OUT_DIM>
class RoFormer {
private:
  // 1. Input Projection
  Conv1dLayer<T, IN_CHANNELS, D_MODEL, SEQ_LEN> conv1;

  // 2. Transformer Blocks
  RoFormerBlock<T, D_MODEL, NUM_HEADS, SEQ_LEN, D_FF> layers[NUM_LAYERS];

  // 3. Final Output Layer
  FCLayer<T, D_MODEL, OUT_DIM> fc_out;

  // Intermediate Buffers
  // Conv Output / Block Input/Output
  // We need to ping-pong or manage state between layers.
  // For simplicity in HLS, we might declare a buffer for each inter-layer stage
  // or reuse two buffers (ping-pong).
  // To match strict HLS dataflow, independent variables are safer initially.
  T buffer_seq[NUM_LAYERS + 1][SEQ_LEN][D_MODEL];
  // buffer_seq[0] is output of Conv1d
  // buffer_seq[1] is output of Layer 0...

public:
  RoFormer() {}

  // --- Weight Setters ---
  // Conv1d
  void set_conv_weights(const T w[D_MODEL][IN_CHANNELS][3]) {
    conv1.setWeights(w);
  }
  void set_conv_bias(const T b[D_MODEL]) { conv1.setBias(b); }

  // Layers
  // Note: In real HLS, we'd pass an index or verify this unrolls
  RoFormerBlock<T, D_MODEL, NUM_HEADS, SEQ_LEN, D_FF> &get_layer(int i) {
    return layers[i];
  }

  // Final FC
  void set_final_fc_weights(const T w[OUT_DIM][D_MODEL]) {
    fc_out.setWeights(w);
  }
  void set_final_fc_bias(const T b[OUT_DIM]) { fc_out.setBias(b); }

  // Forward Pass
  // Input: [IN_CHANNELS][SEQ_LEN]
  // Output: [OUT_DIM]
  void forward(const T input[IN_CHANNELS][SEQ_LEN], T output[OUT_DIM]) {
// REMOVED DATAFLOW for stability.
// Sequential execution avoids synthesis "multiple writer" errors on buffers.
#pragma HLS ARRAY_PARTITION variable = buffer_seq type = complete dim = 1

    // 1. Conv1d
    // Input -> buffer_seq[0]
    conv1.forward(input, buffer_seq[0]);

    // 2. Transformer Blocks Stack
    // Unroll loop for Dataflow
    for (int i = 0; i < NUM_LAYERS; ++i) {
#pragma HLS UNROLL
      layers[i].forward(buffer_seq[i], buffer_seq[i + 1]);
    }

    // 3. Global Average Pooling
    // Mean over SEQ_LEN dimension: [SEQ_LEN][D_MODEL] -> [D_MODEL]
    T pooled[D_MODEL];

    // Accumulate
    for (int d = 0; d < D_MODEL; ++d) {
#pragma HLS UNROLL
      T sum = 0;
      for (int t = 0; t < SEQ_LEN; ++t) {
#pragma HLS PIPELINE
        sum += buffer_seq[NUM_LAYERS][t][d];
      }
      pooled[d] = sum / (T)SEQ_LEN;
    }

    // 4. Final FC
    fc_out.forward(pooled, output);
  }
};

#endif // ROFORMER_H
