#ifndef CONV1DLAYER_H
#define CONV1DLAYER_H

#include <iostream>

// Template parameters:
// T: Data type (float, fixed-point)
// IN_CHANNELS: Number of input channels
// OUT_CHANNELS: Number of output channels
// SEQ_LEN: Length of the sequence
// KERNEL_SIZE: Kernel size (fixed to 3 for this implementation, but templated for clarity)
// PADDING: fixed to 1
template <typename T, int IN_CHANNELS, int OUT_CHANNELS, int SEQ_LEN, int KERNEL_SIZE=3>
class Conv1dLayer {
private:
    T weights[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE];
    T bias[OUT_CHANNELS];

public:
    Conv1dLayer() {
        // Init weights/bias to 0
        for(int o=0; o<OUT_CHANNELS; ++o) {
            bias[o] = 0;
            for(int i=0; i<IN_CHANNELS; ++i) {
                for(int k=0; k<KERNEL_SIZE; ++k) {
                    weights[o][i][k] = 0;
                }
            }
        }
    }

    void setWeights(const T w[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE]) {
         for(int o=0; o<OUT_CHANNELS; ++o) {
            for(int i=0; i<IN_CHANNELS; ++i) {
                for(int k=0; k<KERNEL_SIZE; ++k) {
                    weights[o][i][k] = w[o][i][k];
                }
            }
        }
    }

    void setBias(const T b[OUT_CHANNELS]) {
        for(int o=0; o<OUT_CHANNELS; ++o) {
            bias[o] = b[o];
        }
    }

    // Forward pass
    // Input format: [IN_CHANNELS][SEQ_LEN] (Matches PyTorch logical layout before transpose)
    // Output format: [SEQ_LEN][OUT_CHANNELS] (Transposed, ready for Transformer/Linear layers)
    void forward(const T input[IN_CHANNELS][SEQ_LEN], T output[SEQ_LEN][OUT_CHANNELS]) {
        #pragma HLS INLINE off

        // Padding = 1 for Kernel = 3 means:
        // t=0 uses: [padding, input[t], input[t+1]] -> conceptually [0, input[0], input[1]]
        // t=i uses: [input[i-1], input[i], input[i+1]]
        // t=last uses: [input[last-1], input[last], padding] -> [input[last-1], input[last], 0]

        for (int t = 0; t < SEQ_LEN; ++t) {
            #pragma HLS PIPELINE
            
            for (int o = 0; o < OUT_CHANNELS; ++o) {
                #pragma HLS UNROLL
                T sum = bias[o];
                
                for (int i = 0; i < IN_CHANNELS; ++i) {
                     #pragma HLS UNROLL
                    
                    // Kernel loop unrolled manually or by HLS for K=3
                    // k=0 corresponds to t-1
                    // k=1 corresponds to t
                    // k=2 corresponds to t+1
                    
                    // k=0 (Left neighbor)
                    if (t > 0) {
                        sum += input[i][t-1] * weights[o][i][0];
                    }
                    // else padded with 0 (do nothing)

                    // k=1 (Center)
                    sum += input[i][t] * weights[o][i][1];

                    // k=2 (Right neighbor)
                    if (t < SEQ_LEN - 1) {
                        sum += input[i][t+1] * weights[o][i][2];
                    }
                    // else padded with 0
                }
                output[t][o] = sum;
            }
        }
    }
};

#endif // CONV1DLAYER_H
