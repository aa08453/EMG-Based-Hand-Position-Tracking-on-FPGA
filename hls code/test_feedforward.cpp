#include <iostream>
#include <cmath>
#include "FeedForward.h"

#define TEST_D_MODEL 2
#define TEST_D_FF 2

int main() {
    std::cout << "Starting FeedForward Test..." << std::endl;

    FeedForward<float, TEST_D_MODEL, TEST_D_FF> ff;

    // We want to test Input -> FC1 -> ReLU -> FC2 -> Output

    // 1. Setup FC1 (D_MODEL -> D_FF)
    // We want to produce some negative values to test ReLU.
    // Weights: Identity. Bias: [-5, 5]
    // Input: [2, 2]
    // FC1 Out: [2-5, 2+5] = [-3, 7]
    float w1[TEST_D_FF][TEST_D_MODEL] = {
        {1.0f, 0.0f},
        {0.0f, 1.0f}
    };
    float b1[TEST_D_FF] = {-5.0f, 5.0f};

    ff.set_fc1_weights(w1, b1);

    // 2. Setup ReLU
    // Input: [-3, 7]
    // Output: [0, 7]

    // 3. Setup FC2 (D_FF -> D_MODEL)
    // Weights: Identity. Bias: [1, 1]
    // Input: [0, 7]
    // FC2 Out: [0+1, 7+1] = [1, 8]
    float w2[TEST_D_MODEL][TEST_D_FF] = {
        {1.0f, 0.0f},
        {0.0f, 1.0f}
    };
    float b2[TEST_D_MODEL] = {1.0f, 1.0f};

    ff.set_fc2_weights(w2, b2);

    // Run Forward
    float input[TEST_D_MODEL] = {2.0f, 2.0f};
    float output[TEST_D_MODEL];

    ff.forward(input, output);

    // Verify
    float expected[TEST_D_MODEL] = {1.0f, 8.0f};
    bool pass = true;

    for(int i=0; i<TEST_D_MODEL; ++i) {
        if(std::abs(output[i] - expected[i]) > 0.001f) {
            pass = false;
            std::cout << "Mismatch at " << i << ": Got " << output[i] << " Exp " << expected[i] << std::endl;
        }
    }

    if(pass) std::cout << "Test FeedForward PASSED!" << std::endl;
    else { std::cout << "Test FeedForward FAILED!" << std::endl; return 1; }

    return 0;
}
