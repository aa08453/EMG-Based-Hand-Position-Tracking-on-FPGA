#ifndef ROFORMERBLOCK_H
#define ROFORMERBLOCK_H

#include "FeedForward.h"
#include "LayerNorm.h"
#include "MultiHeadAttention.h"
#include "RotaryEmbedding.h"

// T: Data type
// D_MODEL: Data dimension
// NUM_HEADS: Attention heads
// SEQ_LEN: Sequence Length
// D_FF: FeedForward hidden dimension
template <typename T, int D_MODEL, int NUM_HEADS, int SEQ_LEN, int D_FF>
class RoFormerBlock {
private:
  MultiHeadAttention<T, D_MODEL, NUM_HEADS, SEQ_LEN> mha;
  FeedForward<T, D_MODEL, D_FF> ff;
  LayerNorm<T, D_MODEL> ln1;
  LayerNorm<T, D_MODEL> ln2;
  RotaryEmbedding<T, D_MODEL, SEQ_LEN> rope;

  // Intermediate Buffers (minimized)
  // q_rot doubles as attn_out (safe: MHA streaming reads q_rot[i] before
  // writing output[i])
  T q_rot[SEQ_LEN][D_MODEL];
  T k_rot[SEQ_LEN][D_MODEL];
  // Removed: attn_out (reuse q_rot), temp, ff_out

public:
  RoFormerBlock() {}

  // MHA Weights
  void set_mha_q(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) {
    mha.set_w_q(w, b);
  }
  void set_mha_k(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) {
    mha.set_w_k(w, b);
  }
  void set_mha_v(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) {
    mha.set_w_v(w, b);
  }
  void set_mha_o(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) {
    mha.set_w_o(w, b);
  }

  // FF Weights
  void set_ff_1(const T w[D_FF][D_MODEL], const T b[D_FF]) {
    ff.set_fc1_weights(w, b);
  }
  void set_ff_2(const T w[D_MODEL][D_FF], const T b[D_MODEL]) {
    ff.set_fc2_weights(w, b);
  }

  // LN Weights
  void set_ln1(const T g[D_MODEL], const T b[D_MODEL]) {
    ln1.setGamma(g);
    ln1.setBeta(b);
  }
  void set_ln2(const T g[D_MODEL], const T b[D_MODEL]) {
    ln2.setGamma(g);
    ln2.setBeta(b);
  }

  // Forward pass
  // input_seq: [SEQ_LEN][D_MODEL] (src)
  // output_seq: [SEQ_LEN][D_MODEL]
  void block_forward(T input_seq[SEQ_LEN][D_MODEL],
                     T output_seq[SEQ_LEN][D_MODEL]) {
#pragma HLS INLINE off

    // 1. RoPE Application (Q, K)
    rope.rope_forward(input_seq, q_rot);
    rope.rope_forward(input_seq, k_rot);

    // 2. Multi-Head Attention
    // Output written to q_rot (reused — safe because streaming MHA
    // reads q_rot[i] before writing output[i] for each timestep)
    mha.mha_forward(q_rot, k_rot, input_seq, q_rot);

  // 3. Add & Norm 1 (Post-Norm) — no temp buffer needed
  BLOCK_ADD_NORM1:
    for (int t = 0; t < SEQ_LEN; ++t) {
      T ln_in[D_MODEL];
      T ln_out[D_MODEL];

    BLOCK_RESIDUAL1:
      for (int i = 0; i < D_MODEL; ++i) {
#pragma HLS PIPELINE
        ln_in[i] = input_seq[t][i] + q_rot[t][i]; // q_rot now holds attn_out
      }

      ln1.ln_forward(ln_in, ln_out);

    BLOCK_LN1_COPY_OUT:
      for (int i = 0; i < D_MODEL; i++) {
#pragma HLS PIPELINE
        output_seq[t][i] = ln_out[i];
      }
    }

  // 4+5. Feed Forward + Add & Norm 2 (fused per-timestep — no ff_out buffer)
  BLOCK_FF_AND_NORM2:
    for (int t = 0; t < SEQ_LEN; ++t) {
      T ff_in[D_MODEL];
      T ff_res[D_MODEL];
      T ln_in[D_MODEL];
      T ln_out[D_MODEL];

    BLOCK_FF_COPY_IN:
      for (int i = 0; i < D_MODEL; i++) {
#pragma HLS PIPELINE
        ff_in[i] = output_seq[t][i];
      }

      ff.ff_forward(ff_in, ff_res);

    // Fused: residual add + LayerNorm2 immediately (no ff_out storage)
    BLOCK_RESIDUAL2:
      for (int i = 0; i < D_MODEL; ++i) {
#pragma HLS PIPELINE
        ln_in[i] = output_seq[t][i] + ff_res[i];
      }

      ln2.ln_forward(ln_in, ln_out);

    BLOCK_LN2_COPY_OUT:
      for (int i = 0; i < D_MODEL; ++i) {
#pragma HLS PIPELINE
        output_seq[t][i] = ln_out[i];
      }
    }
  }
};

#endif // ROFORMERBLOCK_H
