#include <iostream>
#include <cmath>
#include "Conv1dLayer.h"

#define TEST_SEQ 10
#define TEST_IN 2
#define TEST_OUT 2

int main() {
    std::cout << "Starting Conv1d Layer Test..." << std::endl;

    // 1. Setup Wrapper/Layer
    //    IN_CH=2, OUT_CH=2, SEQ=10
    Conv1dLayer<float, TEST_IN, TEST_OUT, TEST_SEQ> conv;

    // 2. Setup Weights (Identity-ish)
    //    Weights: [OUT][IN][KERNEL]
    float weights[TEST_OUT][TEST_IN][3] = {0};
    
    // Channel 0 -> Channel 0 (Center pass through)
    weights[0][0][1] = 1.0f; 
    
    // Channel 1 -> Channel 1 (Center pass through)
    weights[1][1][1] = 1.0f;

    // Bias
    float bias[TEST_OUT] = {0.0f, 0.0f};

    conv.setWeights(weights);
    conv.setBias(bias);

    // 3. Setup Input
    float input[TEST_IN][TEST_SEQ];
    for(int i=0; i<TEST_SEQ; ++i) {
        input[0][i] = (float)i;       // Ch0: 0, 1, 2...
        input[1][i] = (float)(i*10);  // Ch1: 0, 10, 20...
    }

    // 4. Run Forward
    float output[TEST_SEQ][TEST_OUT];
    conv.forward(input, output);

    // 5. Verify (Identity conv should preserve values, ignoring padding effect for now)
    bool pass = true;
    for(int t=0; t<TEST_SEQ; ++t) {
        // Expected: Same as input
        float expected0 = input[0][t];
        float expected1 = input[1][t];
        
        // Output format is [Time][Channel]
        if(std::abs(output[t][0] - expected0) > 0.001) {
            std::cout << "Mismatch Ch0 at " << t << ": " << output[t][0] << " != " << expected0 << std::endl;
            pass = false;
        }
        if(std::abs(output[t][1] - expected1) > 0.001) {
            std::cout << "Mismatch Ch1 at " << t << ": " << output[t][1] << " != " << expected1 << std::endl;
            pass = false;
        }
    }

    // Test Padding (Shift)
    // Set kernel to look at previous sample (Left shift / Delay)
    // Reset weights
    for(int o=0; o<TEST_OUT; ++o) 
        for(int i=0; i<TEST_IN; ++i) 
            for(int k=0; k<3; ++k) weights[o][i][k] = 0.0f;
            
    // Output 0 looks at Input 0's previous time (Index 0 in kernel? No. Kernel: [0, 1, 2] -> [t-1, t, t+1])
    // So to see t-1, we activate index 0?
    // formula: sum += w[k] * in[t + k - 1]
    // if k=0: in[t - 1]. Correct.
    weights[0][0][0] = 1.0f; 
    conv.setWeights(weights);
    
    conv.forward(input, output);
    
    // Check t=5. Should see Input[0][4] = 4.0
    if(std::abs(output[5][0] - 4.0f) > 0.001) {
         std::cout << "Shift Test Mismatch: " << output[5][0] << " != 4.0" << std::endl;
         pass = false;
    }
    // Check t=0. Should see 0 (padding)
    if(std::abs(output[0][0] - 0.0f) > 0.001) {
         std::cout << "Padding Test Mismatch: " << output[0][0] << " != 0.0" << std::endl;
         pass = false;
    }

    if(pass) std::cout << "Test Conv1d PASSED!" << std::endl;
    else std::cout << "Test Conv1d FAILED!" << std::endl;

    return 0;
}
