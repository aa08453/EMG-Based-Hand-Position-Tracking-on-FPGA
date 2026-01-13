#ifndef ROTARYEMBEDDING_H
#define ROTARYEMBEDDING_H

#include <cmath>
#include <iostream>

// T: Data type
// DIM: Dimension of the embedding (d_model)
// MAX_SEQ_LEN: Maximum sequence length
template <typename T, int DIM, int MAX_SEQ_LEN>
class RotaryEmbedding {
private:
    T cos_cache[MAX_SEQ_LEN][DIM];
    T sin_cache[MAX_SEQ_LEN][DIM];
    
    // Config
    int base = 10000;

public:
    RotaryEmbedding() {
        init_cache();
    }

    void init_cache() {
        // Precompute Cos and Sin tables
        // Logic matches RoFormer.py:
        // inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
        // freqs = outer(t, inv_freq)
        // emb = cat(freqs, freqs)
        
        for (int t = 0; t < MAX_SEQ_LEN; ++t) {
            for (int i = 0; i < DIM; i += 2) {
                // Determine frequency for this pair of dimensions
                // i goes 0, 2, 4... 
                // exponent = i / DIM
                float exponent = (float)i / (float)DIM;
                float freq_val = 1.0f / std::pow((float)base, exponent);
                
                float angle = t * freq_val;
                
                T c_val = std::cos(angle);
                T s_val = std::sin(angle);
                
                // RoFormer.py: emb = cat(freqs, freqs)
                // The indices in the final embedding:
                // First half (0 to DIM/2 - 1) gets freqs corresponding to 0..DIM/2
                // Second half (DIM/2 to DIM - 1) gets the SAME freqs?
                // Wait, RoFormer.py Logic:
                // inv_freq size: DIM/2
                // freqs shape: [seq, DIM/2]
                // emb = cat((freqs, freqs), dim=-1) -> shape [seq, DIM]
                
                // So:
                // cos_cache[t][i/2]           = cos(angle)
                // cos_cache[t][i/2 + DIM/2]   = cos(angle)
                
                // But wait, the 'i' in the loop is stepping by 2.
                // The python code uses `arange(0, dim, 2)`.
                // Let's index strictly:
                
                // "freas" vector has indices 0..(DIM/2 - 1). 
                // Entry k in "freqs" corresponds to angle based on 2*k/DIM.
                
                int half_dim = DIM / 2;
                int freq_idx = i / 2; // 0, 1, 2...
                
                // In the concatenated output:
                // Index 'freq_idx' gets cos(angle)
                // Index 'freq_idx + half_dim' gets cos(angle)
                
                cos_cache[t][freq_idx] = c_val;
                cos_cache[t][freq_idx + half_dim] = c_val;
                
                sin_cache[t][freq_idx] = s_val;
                sin_cache[t][freq_idx + half_dim] = s_val;
            }
        }
    }

    // Forward Pass
    // input: [SEQ_LEN][DIM]
    // output: [SEQ_LEN][DIM]
    void forward(const T input[MAX_SEQ_LEN][DIM], T output[MAX_SEQ_LEN][DIM]) {
        // x1 = x[..., :half]
        // x2 = x[..., half:]
        // rotate_half(x) = cat(-x2, x1)
        // result = x * cos + rotate_half(x) * sin
        
        int half_dim = DIM / 2;

        for (int t = 0; t < MAX_SEQ_LEN; ++t) {
            #pragma HLS PIPELINE
            for (int i = 0; i < DIM; ++i) {
                #pragma HLS UNROLL
                
                T x_val = input[t][i];
                T cos_val = cos_cache[t][i];
                T sin_val = sin_cache[t][i];
                
                T rot_val; // This is the value from "rotate_half(x)" at index i
                
                // Construct rotate_half logic:
                // if i < half_dim: component comes from -x2 (which is -input[t][i + half_dim])
                // if i >= half_dim: component comes from x1 (which is input[t][i - half_dim])
                
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
