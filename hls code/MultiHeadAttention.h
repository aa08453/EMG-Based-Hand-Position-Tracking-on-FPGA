#ifndef MULTIHEADATTENTION_H
#define MULTIHEADATTENTION_H

#include "FCLayer.h"
#include "Softmax.h"
#ifdef __SYNTHESIS__
#include <hls_math.h>
#else
#include "hls_math.h"
#endif

// T: Data type
// D_MODEL: Hidden dimension (e.g. 128)
// NUM_HEADS: Number of heads (e.g. 4)
// SEQ_LEN: Sequence length (Max)
template <typename T, int D_MODEL, int NUM_HEADS, int SEQ_LEN>
class MultiHeadAttention {
private:
  static const int HEAD_DIM = D_MODEL / NUM_HEADS;

  // Projections
  FCLayer<T, D_MODEL, D_MODEL> w_q;
  FCLayer<T, D_MODEL, D_MODEL> w_k;
  FCLayer<T, D_MODEL, D_MODEL> w_v;
  FCLayer<T, D_MODEL, D_MODEL> w_o;

  // Internal buffers
  // Storing full sequence Q, K, V
  // Format: [SEQ_LEN][NUM_HEADS][HEAD_DIM]
  T Q[SEQ_LEN][NUM_HEADS][HEAD_DIM];
  T K[SEQ_LEN][NUM_HEADS][HEAD_DIM];
  T V[SEQ_LEN][NUM_HEADS][HEAD_DIM];

  // Attention Scores: [NUM_HEADS][SEQ_LEN][SEQ_LEN]
  // Note: This is memory intensive!
  // T scores[NUM_HEADS][SEQ_LEN][SEQ_LEN];

  // Softmax helper
  Softmax<T, SEQ_LEN> softmax;

public:
  MultiHeadAttention() {
    // Initialize weights if needed or provide setters
  }

  // Setters for weights (delegating to FCLayer)
  void set_w_q(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) {
    w_q.setWeights(w);
    w_q.setBias(b);
  }
  void set_w_k(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) {
    w_k.setWeights(w);
    w_k.setBias(b);
  }
  void set_w_v(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) {
    w_v.setWeights(w);
    w_v.setBias(b);
  }
  void set_w_o(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) {
    w_o.setWeights(w);
    w_o.setBias(b);
  }

  // Forward pass
  // input_q, input_k, input_v: [SEQ_LEN][D_MODEL]
  // output: [SEQ_LEN][D_MODEL]
  // We assume batch size 1 for HLS
  void forward(const T q_rot[SEQ_LEN][D_MODEL], const T k_rot[SEQ_LEN][D_MODEL],
               const T v_origin[SEQ_LEN][D_MODEL], T output[SEQ_LEN][D_MODEL]) {
#pragma HLS INLINE off

    // #pragma HLS DATAFLOW

    // 1. Linear Projections + Split Heads
    // Iterate over sequence length
    for (int t = 0; t < SEQ_LEN; ++t) {
      T q_flat[D_MODEL], k_flat[D_MODEL], v_flat[D_MODEL];

      // Project
      w_q.forward(q_rot[t], q_flat);
      w_k.forward(k_rot[t], k_flat);
      w_v.forward(v_origin[t], v_flat);

      // Reshape / Split Heads
      for (int h = 0; h < NUM_HEADS; ++h) {
        // #pragma HLS UNROLL
        for (int d = 0; d < HEAD_DIM; ++d) {
          // #pragma HLS UNROLL
          Q[t][h][d] = q_flat[h * HEAD_DIM + d];
          K[t][h][d] = k_flat[h * HEAD_DIM + d];
          V[t][h][d] = v_flat[h * HEAD_DIM + d];
        }
      }
    }

    // 2. Scaled Dot-Product Attention
    T scale = 1.0 / hls::sqrt((T)HEAD_DIM);

    for (int h = 0; h < NUM_HEADS; ++h) {
      // For each query position
      for (int i = 0; i < SEQ_LEN; ++i) {
        // Against all key positions
        T row_scores[SEQ_LEN];
        for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE
          T dot = 0;
          for (int d = 0; d < HEAD_DIM; ++d) {
            dot += Q[i][h][d] * K[j][h][d];
          }
          row_scores[j] = dot * scale;
        }

        // Apply Softmax
        T row_probs[SEQ_LEN];
        softmax.forward(row_scores, row_probs);

        // Multiply with Value
        // Verify: Attention(Q, K, V) = softmax(QK^T / d)V
        // Check if we need to store result or accumulate directly
        // We need to compute the weighted sum of V vectors
        // Result for this head, this position i is a vector of size HEAD_DIM
        // We can't write directly to Output yet because output is [D_MODEL]
        // We'll reconstruct the flat vector in step 3.
        // Wait, typically we store the attention output per head first.
        // Let's optimize: accumulate directly into a temporary buffer for this
        // position/head

        // Oops, I can't easily write to 'output' directly because it's
        // [SEQ_LEN][D_MODEL] interleaved I need an intermediate buffer for the
        // concatenated heads
      }
      // Wait, loop structure above is inefficient for writing V
    }

    // Better Loop Structure for Step 2 & 3 Combined
    // For each head, calculate all attention outputs
    T context[SEQ_LEN][D_MODEL]; // concatenated heads

    for (int h = 0; h < NUM_HEADS; ++h) {
      for (int i = 0; i < SEQ_LEN; ++i) {
        // Calculate Scores
        T row_scores[SEQ_LEN];
        for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE
          T dot = 0;
          for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS UNROLL
            dot += Q[i][h][d] * K[j][h][d];
          }
          row_scores[j] = dot * scale;
        }

        // Softmax
        T row_probs[SEQ_LEN];
        softmax.forward(row_scores, row_probs);

        // Weighted Sum of V
        for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS PIPELINE
          T val_acc = 0;
          for (int j = 0; j < SEQ_LEN; ++j) {
            val_acc += row_probs[j] * V[j][h][d];
          }
          // Place into context buffer
          context[i][h * HEAD_DIM + d] = val_acc;
        }
      }
    }

    // 4. Output Projection
    for (int t = 0; t < SEQ_LEN; ++t) {
#pragma HLS PIPELINE
      w_o.forward(context[t], output[t]);
    }
  }
};

#endif // MULTIHEADATTENTION_H
