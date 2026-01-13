#ifndef RELU_H
#define RELU_H

#include <algorithm>

// Template parameters:
// T: Data type
// DIM: Dimension of the input/output array
template <typename T, int DIM>
class ReLU {
public:
    ReLU() {}

    // Forward pass: y = max(0, x)
    // inplace: if true, modifies input array directly
    void forward(T input[DIM], T output[DIM]) {
        #pragma HLS PIPELINE
        
        for (int i = 0; i < DIM; ++i) {
            #pragma HLS UNROLL
            if (input[i] > 0) {
                output[i] = input[i];
            } else {
                output[i] = 0;
            }
        }
    }

    // In-place version
    void forward(T data[DIM]) {
        #pragma HLS PIPELINE
        for (int i = 0; i < DIM; ++i) {
            #pragma HLS UNROLL
            if (data[i] < 0) {
                data[i] = 0;
            }
        }
    }
};

#endif // RELU_H
