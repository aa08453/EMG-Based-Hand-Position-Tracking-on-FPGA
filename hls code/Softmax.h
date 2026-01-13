#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cmath>
#include <limits>

template <typename T, int DIM>
class Softmax {
public:
    void forward(T input[DIM], T output[DIM]) {
        #pragma HLS INLINE
        
        // Find max for numerical stability
        T max_val = -std::numeric_limits<T>::max();
        for (int i = 0; i < DIM; ++i) {
            #pragma HLS PIPELINE
            if (input[i] > max_val) max_val = input[i];
        }

        T sum_exp = 0;
        for (int i = 0; i < DIM; ++i) {
            #pragma HLS PIPELINE
            T exp_val = std::exp(input[i] - max_val);
            output[i] = exp_val;
            sum_exp += exp_val;
        }

        T inv_sum = 1.0 / sum_exp;
        for (int i = 0; i < DIM; ++i) {
            #pragma HLS PIPELINE
            output[i] *= inv_sum;
        }
    }
};

#endif // SOFTMAX_H
