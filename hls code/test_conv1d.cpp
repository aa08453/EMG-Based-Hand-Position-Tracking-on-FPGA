#include <iostream>
#include <cmath>
#include "Conv1dLayer.h"

// Test Dimensions
#define TEST_IN_CHANNELS 2
#define TEST_OUT_CHANNELS 2
#define TEST_SEQ_LEN 4
#define KERNEL 3

int main() {
    std::cout << "Starting Conv1d Layer Test..." << std::endl;

    // Instantiate
    Conv1dLayer<float, TEST_IN_CHANNELS, TEST_OUT_CHANNELS, TEST_SEQ_LEN> conv;

    // --- Weights Setup ---
    // Kernel Size 3: [k0, k1, k2]
    // OutCh 0, InCh 0: [0, 1, 0] (Identity Center)
    // OutCh 0, InCh 1: [0, 0, 0]
    // Result: OutCh 0 should just copy InCh 0
    
    // OutCh 1, InCh 0: [0, 0, 0]
    // OutCh 1, InCh 1: [1, 0, 0] (Left Neighbor / Previous Time Step)
    // Result: OutCh 1 should be InCh 1 shifted right (delay)
    
    float w[TEST_OUT_CHANNELS][TEST_IN_CHANNELS][KERNEL];
    
    // Initialize all to 0
    for(int o=0; o<TEST_OUT_CHANNELS; ++o)
        for(int i=0; i<TEST_IN_CHANNELS; ++i)
            for(int k=0; k<KERNEL; ++k) w[o][i][k] = 0.0f;

    // Set specific kernels
    w[0][0][1] = 1.0f; // Out0 <- In0 Center
    w[1][1][0] = 1.0f; // Out1 <- In1 Left (k=0 is t-1)

    conv.setWeights(w);

    float b[TEST_OUT_CHANNELS] = {0.0f, 0.0f};
    conv.setBias(b);

    // --- Input Setup ---
    // In0: [1, 2, 3, 4]
    // In1: [10, 20, 30, 40]
    float input[TEST_IN_CHANNELS][TEST_SEQ_LEN] = {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {10.0f, 20.0f, 30.0f, 40.0f}
    };
    
    float output[TEST_SEQ_LEN][TEST_OUT_CHANNELS];

    // Run
    conv.forward(input, output);

    // --- Verification ---
    // Expected Output:
    // Out0 (Copy In0): [1, 2, 3, 4]
    
    // Out1 (Shift In1 Left / Delay):
    // t=0: uses In1[-1] (pad=0) -> 0
    // t=1: uses In1[0]  -> 10
    // t=2: uses In1[1]  -> 20
    // t=3: uses In1[2]  -> 30
    // So: [0, 10, 20, 30]

    bool pass = true;
    
    // Check Out0
    float expected_0[TEST_SEQ_LEN] = {1.0f, 2.0f, 3.0f, 4.0f};
    for(int t=0; t<TEST_SEQ_LEN; ++t) {
        if(output[t][0] != expected_0[t]) {
            pass = false;
            std::cout << "Mismatch Out0 at " << t << ": Got " << output[t][0] << " Exp " << expected_0[t] << std::endl;
        }
    }

    // Check Out1
    float expected_1[TEST_SEQ_LEN] = {0.0f, 10.0f, 20.0f, 30.0f};
    for(int t=0; t<TEST_SEQ_LEN; ++t) {
        if(output[t][1] != expected_1[t]) {
            pass = false;
            std::cout << "Mismatch Out1 at " << t << ": Got " << output[t][1] << " Exp " << expected_1[t] << std::endl;
        }
    }

    if(pass) std::cout << "Test Conv1d PASSED!" << std::endl;
    else {
        std::cout << "Test Conv1d FAILED!" << std::endl;
        return 1;
    }

    return 0;
}
