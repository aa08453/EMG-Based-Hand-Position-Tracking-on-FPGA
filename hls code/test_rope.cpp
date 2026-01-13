#include <iostream>
#include <cmath>
#include "RotaryEmbedding.h"

#define TEST_DIM 4
#define TEST_SEQ 2

int main() {
    std::cout << "Starting Rotary Embedding Test..." << std::endl;

    RotaryEmbedding<float, TEST_DIM, TEST_SEQ> rope;

    float input[TEST_SEQ][TEST_DIM] = {
        {1.0f, 2.0f, 3.0f, 4.0f}, // t=0
        {1.0f, 2.0f, 3.0f, 4.0f}  // t=1
    };
    float output[TEST_SEQ][TEST_DIM];

    rope.forward(input, output);

    bool pass = true;

    // --- Check t=0 ---
    // Angle = 0 for all dims. Cos=1, Sin=0.
    // Result should be Identity.
    std::cout << "Checking t=0 (Identity Rotation)..." << std::endl;
    for(int i=0; i<TEST_DIM; ++i) {
        if(std::abs(output[0][i] - input[0][i]) > 0.001f) {
            pass = false;
            std::cout << "Mismatch at t=0, dim=" << i << ": Got " << output[0][i] << " Exp " << input[0][i] << std::endl; 
        }
    }

    // --- Check t=1 ---
    std::cout << "Checking t=1 (Active Rotation)..." << std::endl;
    // Base=10000. Dims=4.
    // Pair 0 (Indices 0, 1): Freq = 1 / 10000^(0/4) = 1.0. 
    //   Angle = 1*1 = 1.0 rad.
    //   Cos(1.0) = 0.5403, Sin(1.0) = 0.8414
    
    // Pair 1 (Indices 2, 3): Freq = 1 / 10000^(2/4) = 1/100 = 0.01.
    //   Angle = 1*0.01 = 0.01 rad.
    //   Cos(0.01) = 0.9999, Sin(0.01) = 0.0099

    // Correct Logic based on cat(freqs, freqs):
    // Indices [0, 2] use Freq0 (Angle 1.0)
    // Indices [1, 3] use Freq1 (Angle 0.01)

    // Out[0] (Idx 0, Freq0): x=1, rot=-3.
    // 1 * Cos(1) + (-3) * Sin(1) = 0.5403 - 2.5242 = -1.9839
    
    // Out[1] (Idx 1, Freq1): x=2, rot=-4.
    // 2 * Cos(0.01) + (-4) * Sin(0.01) = 2 * 0.99995 - 4 * 0.00999 
    // = 1.9999 - 0.0399 = 1.9600

    // Out[2] (Idx 2, Freq0): x=3, rot=1.
    // 3 * Cos(1) + 1 * Sin(1) = 3 * 0.5403 + 0.8414 
    // = 1.6209 + 0.8414 = 2.4623

    // Out[3] (Idx 3, Freq1): x=4, rot=2.
    // 4 * Cos(0.01) + 2 * Sin(0.01) = 4 * 0.99995 + 2 * 0.00999
    // = 3.9998 + 0.0199 = 4.0197

    float expected_t1[TEST_DIM] = {-1.9839f, 1.9600f, 2.4623f, 4.0197f};
    
    for(int i=0; i<TEST_DIM; ++i) {
        if(std::abs(output[1][i] - expected_t1[i]) > 0.001f) {
            pass = false;
            std::cout << "Mismatch at t=1, dim=" << i << ": Got " << output[1][i] << " Exp " << expected_t1[i] 
                      << " Diff: " << std::abs(output[1][i] - expected_t1[i]) << std::endl;
        }
    }

    if(pass) std::cout << "Test RoPE PASSED!" << std::endl;
    else { std::cout << "Test RoPE FAILED!" << std::endl; return 1; }

    return 0;
}
