#ifndef MULTIHEADATTENTION_H
#define MULTIHEADATTENTION_H

#include "FCLayer.h"
#include "Softmax.h"
#ifdef __SYNTHESIS__
#include <hls_math.h>
#else
#include "mock_hls_math.h"
#endif

// T: Data type
// D_MODEL: Hidden dimension (e.g. 128)
// NUM_HEADS: Number of heads (e.g. 4)
// SEQ_LEN: Sequence length (Max)
template <typename T, int D_MODEL, int NUM_HEADS, int SEQ_LEN>
class MultiHeadAttention {
private:
  static const int HEAD_DIM = D_MODEL / NUM_HEADS;
  // Precomputed 1/sqrt(HEAD_DIM) — avoids synthesizing hls::sqrt hardware
  // 1/sqrt(32) = 0.176777 for HEAD_DIM=32

  // Projections
  FCLayer<T, D_MODEL, D_MODEL> w_q;
  FCLayer<T, D_MODEL, D_MODEL> w_k;
  FCLayer<T, D_MODEL, D_MODEL> w_v;
  FCLayer<T, D_MODEL, D_MODEL> w_o;

  // Only store K and V (Q computed per-timestep on-the-fly)
  T K[SEQ_LEN][NUM_HEADS][HEAD_DIM];
  T V[SEQ_LEN][NUM_HEADS][HEAD_DIM];

  // Softmax helper
  Softmax<T, SEQ_LEN> softmax;

public:
  MultiHeadAttention() {}

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

  // Forward pass (streaming attention)
  // q_rot, k_rot: [SEQ_LEN][D_MODEL] (after RoPE)
  // v_origin:     [SEQ_LEN][D_MODEL] (original input for value)
  // output:       [SEQ_LEN][D_MODEL]
  void mha_forward(const T q_rot[SEQ_LEN][D_MODEL],
                   const T k_rot[SEQ_LEN][D_MODEL],
                   const T v_origin[SEQ_LEN][D_MODEL],
                   T output[SEQ_LEN][D_MODEL]) {
#pragma HLS INLINE off

  // Phase 1: Project K and V for all timesteps, store in K[] and V[]
  MHA_PROJECT_KV:
    for (int t = 0; t < SEQ_LEN; ++t) {
      T k_flat[D_MODEL], v_flat[D_MODEL];
      w_k.forward(k_rot[t], k_flat);
      w_v.forward(v_origin[t], v_flat);

    MHA_SPLIT_HEADS:
      for (int h = 0; h < NUM_HEADS; ++h) {
      MHA_SPLIT_DIM:
        for (int d = 0; d < HEAD_DIM; ++d) {
          K[t][h][d] = k_flat[h * HEAD_DIM + d];
          V[t][h][d] = v_flat[h * HEAD_DIM + d];
        }
      }
    }

    T scale = (T)0.176776695f; // precomputed 1/sqrt(HEAD_DIM)

  // Phase 2: For each query position, compute attention on-the-fly
  MHA_QUERY_LOOP:
    for (int i = 0; i < SEQ_LEN; ++i) {
      // Compute Q for this single timestep (no storage needed)
      T q_flat[D_MODEL];
      w_q.forward(q_rot[i], q_flat);

      // Temporary buffer for one concatenated output row
      T out_row[D_MODEL];

    MHA_HEAD_LOOP:
      for (int h = 0; h < NUM_HEADS; ++h) {
        // Extract Q head for this position
        T q_head[HEAD_DIM];
      MHA_EXTRACT_Q:
        for (int d = 0; d < HEAD_DIM; ++d)
          q_head[d] = q_flat[h * HEAD_DIM + d];

        // Compute dot-product scores against all keys
        T row_scores[SEQ_LEN];
      MHA_DOT_PRODUCT:
        for (int j = 0; j < SEQ_LEN; ++j) {
          T dot = 0;
        MHA_DOT_DIM:
          for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS PIPELINE
            dot += q_head[d] * K[j][h][d];
          }
          row_scores[j] = dot * scale;
        }

        // Softmax
        T row_probs[SEQ_LEN];
        softmax.softmax_forward(row_scores, row_probs);

      // Weighted sum of V -> directly into out_row
      MHA_WEIGHTED_SUM_DIM:
        for (int d = 0; d < HEAD_DIM; ++d) {
          T acc = 0;
        MHA_WEIGHTED_SUM_SEQ:
          for (int j = 0; j < SEQ_LEN; ++j) {
#pragma HLS PIPELINE
            acc += row_probs[j] * V[j][h][d];
          }
          out_row[h * HEAD_DIM + d] = acc;
        }
      }

      // Output projection immediately (no context buffer needed)
      w_o.forward(out_row, output[i]);
    }
  }
};

#endif // MULTIHEADATTENTION_H
