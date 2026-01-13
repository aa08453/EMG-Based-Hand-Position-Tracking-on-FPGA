#ifndef LAYERNORM_H
#define LAYERNORM_H

#include <cmath>

// Template parameters:
// T: Data type
// DIM: Dimension of the input/output array
template <typename T, int DIM>
class LayerNorm {
private:
    T gamma[DIM];
    T beta[DIM];
    T epsilon;

public:
    LayerNorm(T eps = 1e-5) : epsilon(eps) {
        // Initialize gamma to 1 and beta to 0 (params usually loaded later)
        for(int i=0; i<DIM; ++i) {
            gamma[i] = 1.0;
            beta[i] = 0.0;
        }
    }

    void setGamma(const T g[DIM]) {
        for(int i=0; i<DIM; ++i) gamma[i] = g[i];
    }

    void setBeta(const T b[DIM]) {
        for(int i=0; i<DIM; ++i) beta[i] = b[i];
    }

    // Forward pass: y = ((x - mean) / sqrt(var + eps)) * gamma + beta
    void forward(T input[DIM], T output[DIM]) {
        #pragma HLS INLINE // Inline this function for better optimization in larger blocks

        T sum = 0;
        T sum_sq = 0;

        // 1. Calculate Mean
        for (int i = 0; i < DIM; ++i) {
            #pragma HLS PIPELINE
            sum += input[i];
        }
        T mean = sum / DIM;

        // 2. Calculate Variance
        for (int i = 0; i < DIM; ++i) {
            #pragma HLS PIPELINE
            T diff = input[i] - mean;
            sum_sq += diff * diff;
        }
        T variance = sum_sq / DIM;
        T std_dev_inv = 1.0 / std::sqrt(variance + epsilon);

        // 3. Normalize and Scale/Shift
        for (int i = 0; i < DIM; ++i) {
            #pragma HLS PIPELINE
            T norm = (input[i] - mean) * std_dev_inv;
            output[i] = norm * gamma[i] + beta[i];
        }
    }
};

#endif // LAYERNORM_H
