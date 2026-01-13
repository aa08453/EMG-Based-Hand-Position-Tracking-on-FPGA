#include <iostream>
#include <cmath>
#include "RoFormerBlock.h"

// Dimensions
#define DIM 4
#define HEADS 2
#define SEQ 2
#define D_FF 4

int main() {
    std::cout << "Starting RoFormer Block Test..." << std::endl;

    RoFormerBlock<float, DIM, HEADS, SEQ, D_FF> block;

    // Initialization check (should run without crashing)
    // We will leave weights as default 0 (from FCLayer constructor) for a "Zero Test" first,
    // or set them to Identity.
    
    // Setting Identity Weights for MHA Projections
    float eye[DIM][DIM] = {0};
    for(int i=0; i<DIM; ++i) eye[i][i] = 1.0f;
    float zero_b[DIM] = {0};

    block.set_mha_q(eye, zero_b);
    block.set_mha_k(eye, zero_b);
    block.set_mha_v(eye, zero_b);
    block.set_mha_o(eye, zero_b);

    // Setting FF weights
    // FC1 (Dim -> FF): Identity (padded if D_FF > DIM)
    float w_ff1[D_FF][DIM] = {0};
    for(int i=0; i<DIM; ++i) w_ff1[i][i] = 1.0f;
    float b_ff1[D_FF] = {0};
    block.set_ff_1(w_ff1, b_ff1);

    // FC2 (FF -> Dim): Identity
    float w_ff2[DIM][D_FF] = {0};
    for(int i=0; i<DIM; ++i) w_ff2[i][i] = 1.0f;
    float b_ff2[DIM] = {0};
    block.set_ff_2(w_ff2, b_ff2);

    // LN Weights (Gamma=1, Beta=0 is default in constructor, but let's be explicit if needed)
    // Actually LayerNorm constructor sets them.

    // Input Data
    float input[SEQ][DIM] = {
        {1.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 1.0f}
    };
    float output[SEQ][DIM];

    block.forward(input, output);
    
    // Verify meaningful output exists (not NaN, not Inf)
    bool sane = true;
    std::cout << "Output:" << std::endl;
    for(int t=0; t<SEQ; ++t) {
        std::cout << "t=" << t << ": ";
        for(int i=0; i<DIM; ++i) {
            std::cout << output[t][i] << " ";
            if(std::isnan(output[t][i]) || std::isinf(output[t][i])) sane = false;
        }
        std::cout << std::endl;
    }

    if(sane) {
        std::cout << "Test Block PASSED (Sanity Check Only)!" << std::endl;
    } else {
        std::cout << "Test Block FAILED (Output is NaN/Inf)!" << std::endl;
        return 1;
    }

    return 0;
}
