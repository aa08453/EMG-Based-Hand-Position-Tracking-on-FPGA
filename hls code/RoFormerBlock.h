#ifndef ROFORMERBLOCK_H
#define ROFORMERBLOCK_H

#include "MultiHeadAttention.h"
#include "FeedForward.h"
#include "LayerNorm.h"
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

    // Intermediate Buffers
    T q_rot[SEQ_LEN][D_MODEL];
    T k_rot[SEQ_LEN][D_MODEL];
    T attn_out[SEQ_LEN][D_MODEL];
    T ff_out[SEQ_LEN][D_MODEL];
    T temp[SEQ_LEN][D_MODEL]; // For residual calculations

public:
    RoFormerBlock() {}

    // Expose setters for weights
    // (In a real implementation, we would grouping these better, e.g., struct params)
    
    // MHA Weights
    void set_mha_q(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) { mha.set_w_q(w, b); }
    void set_mha_k(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) { mha.set_w_k(w, b); }
    void set_mha_v(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) { mha.set_w_v(w, b); }
    void set_mha_o(const T w[D_MODEL][D_MODEL], const T b[D_MODEL]) { mha.set_w_o(w, b); }

    // FF Weights
    void set_ff_1(const T w[D_FF][D_MODEL], const T b[D_FF]) { ff.set_fc1_weights(w, b); }
    void set_ff_2(const T w[D_MODEL][D_FF], const T b[D_MODEL]) { ff.set_fc2_weights(w, b); }

    // LN Weights
    void set_ln1(const T g[D_MODEL], const T b[D_MODEL]) { ln1.setGamma(g); ln1.setBeta(b); }
    void set_ln2(const T g[D_MODEL], const T b[D_MODEL]) { ln2.setGamma(g); ln2.setBeta(b); }

    // Forward pass
    // input_seq: [SEQ_LEN][D_MODEL] (src)
    // output_seq: [SEQ_LEN][D_MODEL]
    void forward(T input_seq[SEQ_LEN][D_MODEL], T output_seq[SEQ_LEN][D_MODEL]) {
        #pragma HLS DATAFLOW

        // 1. RoPE Application (Q, K)
        // In PyTorch code: q = k = src. apply_rope(q, k).
        // Note: We use the local 'rope' instance which has its own cos/sin cache.
        rope.forward(input_seq, q_rot); 
        rope.forward(input_seq, k_rot);

        // 2. Multi-Head Attention
        // src2 = self_attn(q_rot, k_rot, src)
        mha.forward(q_rot, k_rot, input_seq, attn_out);

        // 3. Add & Norm 1 (Post-Norm)
        // src = norm1(src + dropout(src2))
        // We omit dropout for inference/synthesis usually, or treat as Identity.
        for(int t=0; t<SEQ_LEN; ++t) {
            #pragma HLS PIPELINE
            for(int i=0; i<D_MODEL; ++i) {
                temp[t][i] = input_seq[t][i] + attn_out[t][i];
            }
            // Norm in place (reads from temp, writes to output_seq momentarily to save space or use distinct buffer)
            // Let's use 'output_seq' as intermediate 'src' buffer logic
            T ln_in[D_MODEL];
            T ln_out[D_MODEL];
            for(int i=0; i<D_MODEL; i++) ln_in[i] = temp[t][i];
            
            ln1.forward(ln_in, ln_out);
            
            for(int i=0; i<D_MODEL; i++) output_seq[t][i] = ln_out[i]; 
        }

        // Current state of 'src' is in 'output_seq'
        
        // 4. Feed Forward
        // src2 = linear2(activation(linear1(src)))
        for(int t=0; t<SEQ_LEN; ++t) {
            #pragma HLS PIPELINE
            T ff_in[D_MODEL];
            T ff_res[D_MODEL];
            for(int i=0; i<D_MODEL; i++) ff_in[i] = output_seq[t][i];
            
            ff.forward(ff_in, ff_res);
            
            for(int i=0; i<D_MODEL; i++) ff_out[t][i] = ff_res[i];
        }

        // 5. Add & Norm 2 (Post-Norm)
        // src = norm2(src + src2)
        for(int t=0; t<SEQ_LEN; ++t) {
             #pragma HLS PIPELINE
            T ln_in[D_MODEL];
            T ln_out[D_MODEL]; 
            
            for(int i=0; i<D_MODEL; ++i) {
                ln_in[i] = output_seq[t][i] + ff_out[t][i];
            }

            ln2.forward(ln_in, ln_out);

            for(int i=0; i<D_MODEL; ++i) {
                output_seq[t][i] = ln_out[i];
            }
        }
    }
};

#endif // ROFORMERBLOCK_H
