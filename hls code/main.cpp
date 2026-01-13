#include <iostream>
#include <cmath>
#include "FCLayer.h"
#include "ReLU.h"
#include "LayerNorm.h"
#include "MultiHeadAttention.h"

// Define dimensions for the test
#define TEST_IN_DIM 3
#define TEST_OUT_DIM 2

int main() {
    std::cout << "Starting Fully Connected Layer Test..." << std::endl;

    // Instantiate the layer
    FCLayer<float, TEST_IN_DIM, TEST_OUT_DIM> fc1;

    // --- Test Case 1: Basic Positive Values ---
    std::cout << "Running Test Case 1 (Basic)..." << std::endl;
    // Weights:
    // [0.1, 0.2, 0.3]
    // [0.4, 0.5, 0.6]
    float w[TEST_OUT_DIM][TEST_IN_DIM] = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f}
    };
    fc1.setWeights(w);

    // Biases: [0.01, 0.02]
    float b[TEST_OUT_DIM] = {0.01f, 0.02f};
    fc1.setBias(b);

    // Input: [1.0, 2.0, 3.0]
    float input[TEST_IN_DIM] = {1.0f, 2.0f, 3.0f};
    float output[TEST_OUT_DIM];

    // Run forward pass
    fc1.forward(input, output);

    // Expected Output:
    // Neuron 0: 1.41
    // Neuron 1: 3.22
    float expected[TEST_OUT_DIM] = {1.41f, 3.22f};
    bool pass = true;
    for(int i=0; i<TEST_OUT_DIM; i++) {
        if(std::abs(output[i] - expected[i]) > 0.001f) {
            pass = false;
            std::cout << "Mismatch at index " << i << ": Expected " << expected[i] << ", Got " << output[i] << std::endl;
        }
    }

    if(pass) {
        std::cout << "Test Case 1 (Basic): PASSED!" << std::endl;
    } else {
        std::cout << "Test Case 1 (Basic): FAILED!" << std::endl;
        return 1;
    }

    // --- Test Case 2: Zero Input ---
    std::cout << "\nRunning Test Case 2 (Zero Input)..." << std::endl;
    float input_zero[TEST_IN_DIM] = {0.0f, 0.0f, 0.0f};
    
    fc1.forward(input_zero, output);
    
    // Expected Output: Just bias
    // Neuron 0: 0 + 0.01 = 0.01
    // Neuron 1: 0 + 0.02 = 0.02
    float expected_zero[TEST_OUT_DIM] = {0.01f, 0.02f};
    
    for(int i=0; i<TEST_OUT_DIM; i++) {
        if(std::abs(output[i] - expected_zero[i]) > 0.001f) {
            pass = false;
            std::cout << "Mismatch at index " << i << ": Expected " << expected_zero[i] << ", Got " << output[i] << std::endl;
        }
    }
    
    if(pass) std::cout << "Test Case 2 (Zero Input): PASSED!" << std::endl;
    else { std::cout << "Test Case 2: FAILED!" << std::endl; return 1; }

    // --- Test Case 3: Negative Input/Weights ---
    std::cout << "\nRunning Test Case 3 (Negative Values)..." << std::endl;
    // Update weights to include negatives
    float w_neg[TEST_OUT_DIM][TEST_IN_DIM] = {
        {-0.1f, 0.2f, -0.3f},
        {0.4f, -0.5f, 0.6f}
    };
    fc1.setWeights(w_neg);
    
    // Input with negatives
    float input_neg[TEST_IN_DIM] = {-1.0f, 2.0f, -3.0f};
    
    fc1.forward(input_neg, output);
    
    // Expected Output:
    // Neuron 0: (-0.1*-1) + (0.2*2) + (-0.3*-3) + 0.01 
    //         = 0.1 + 0.4 + 0.9 + 0.01 = 1.41
    // Neuron 1: (0.4*-1) + (-0.5*2) + (0.6*-3) + 0.02 
    //         = -0.4 - 1.0 - 1.8 + 0.02 = -3.18
    float expected_neg[TEST_OUT_DIM] = {1.41f, -3.18f};
    
    for(int i=0; i<TEST_OUT_DIM; i++) {
        if(std::abs(output[i] - expected_neg[i]) > 0.001f) {
            pass = false;
            std::cout << "Mismatch at index " << i << ": Expected " << expected_neg[i] << ", Got " << output[i] << std::endl;
        }
    }

    if(pass) std::cout << "Test Case 3 (Negatives): PASSED!" << std::endl;
    else { std::cout << "Test Case 3: FAILED!" << std::endl; return 1; }

    // --- Test Case 4: ReLU ---
    std::cout << "\nRunning Test Case 4 (ReLU)..." << std::endl;
    ReLU<float, TEST_IN_DIM> relu_layer;
    float input_relu[TEST_IN_DIM] = {-1.0f, 0.5f, -0.2f};
    float output_relu[TEST_IN_DIM];
    
    relu_layer.forward(input_relu, output_relu);
    
    // Expected: [0.0, 0.5, 0.0]
    float expected_relu[TEST_IN_DIM] = {0.0f, 0.5f, 0.0f};
    pass = true;
    for(int i=0; i<TEST_IN_DIM; i++) {
        if(std::abs(output_relu[i] - expected_relu[i]) > 0.001f) {
            pass = false;
            std::cout << "ReLU Mismatch at " << i << ": Expected " << expected_relu[i] << ", Got " << output_relu[i] << std::endl;
        }
    }
    if(pass) std::cout << "Test Case 4 (ReLU): PASSED!" << std::endl;
    else { std::cout << "Test Case 4: FAILED!" << std::endl; return 1; }

    // --- Test Case 5: LayerNorm ---
    std::cout << "\nRunning Test Case 5 (LayerNorm)..." << std::endl;
    LayerNorm<float, TEST_IN_DIM> ln_layer;
    // Input: [1.0, 2.0, 3.0] -> Mean=2.0, Var=2/3=0.666
    // StdDev = sqrt(0.666) ~= 0.816
    // Norm: [-1.224, 0.0, 1.224]
    float input_ln[TEST_IN_DIM] = {1.0f, 2.0f, 3.0f};
    float output_ln[TEST_IN_DIM];
    
    ln_layer.forward(input_ln, output_ln);
    
    // Check against expected values
    // Gamma=1, Beta=0 by default
    float expected_ln[TEST_IN_DIM] = {-1.2247f, 0.0f, 1.2247f};
    
    pass = true;
    for(int i=0; i<TEST_IN_DIM; i++) {
        // increased tolerance slightly for float precision
        if(std::abs(output_ln[i] - expected_ln[i]) > 0.01f) {
            pass = false;
            std::cout << "LayerNorm Mismatch at " << i << ": Expected " << expected_ln[i] << ", Got " << output_ln[i] << std::endl;
        }
    }
     if(pass) std::cout << "Test Case 5 (LayerNorm): PASSED!" << std::endl;
    else { std::cout << "Test Case 5: FAILED!" << std::endl; return 1; }

    // --- Test Case 6: MultiHead Attention (Tiny) ---
    std::cout << "\nRunning Test Case 6 (MultiHeadAttention)..." << std::endl;
    // Params: T=float, D_MODEL=4, NUM_HEADS=2, SEQ_LEN=2
    // HEAD_DIM = 2
    const int MHA_DIM = 4;
    const int MHA_HEADS = 2;
    const int MHA_SEQ = 2;
    
    MultiHeadAttention<float, MHA_DIM, MHA_HEADS, MHA_SEQ> mha;
    
    // Initialize weights to Identity to make checking easier
    // If W_q, W_k, W_v, W_o are identity:
    // Q = Input, K = Input, V = Input
    // Scores = QK^T / sqrt(2)
    // Softmax(Scores)
    // Output = Scores * V
    
    float eye[MHA_DIM][MHA_DIM] = {0};
    for(int i=0; i<MHA_DIM; ++i) eye[i][i] = 1.0f;
    float zero_bias[MHA_DIM] = {0};
    
    mha.set_w_q(eye, zero_bias);
    mha.set_w_k(eye, zero_bias);
    mha.set_w_v(eye, zero_bias);
    mha.set_w_o(eye, zero_bias);
    
    // Input: two identical token vectors -> Attention should match perfectly
    // [1, 0, 1, 0]
    // [0, 1, 0, 1]
    float input_seq[MHA_SEQ][MHA_DIM] = {
        {1.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 1.0f}
    };
    float output_mha[MHA_SEQ][MHA_DIM];
    
    // In our simplified test (Q=K=V=Input):
    // For Head 0 (dim 0-1): Input=[1,0], [0,1]
    // For Head 1 (dim 2-3): Input=[1,0], [0,1]
    // Scale = 1/sqrt(2) = 0.707
    // Dot products:
    // H0:
    //   pos0 . pos0 = 1*1 + 0*0 = 1
    //   pos0 . pos1 = 1*0 + 0*1 = 0
    //   pos1 . pos0 = 0
    //   pos1 . pos1 = 1
    // Scores = [[0.707, 0], [0, 0.707]]
    // Softmax([0.707, 0]) -> exp(0.707)/(exp(0.707)+exp(0)) = 2.02 / (2.02 + 1) = 0.67
    // Softmax([0, 0.707]) -> exp(0)/(...) = 0.33
    // So attn weights are roughly [0.67, 0.33] for diagonal, [0.33, 0.67] off-diagonal?
    // Wait, row 0 scores: [0.707, 0]. Softmax p0=0.67, p1=0.33.
    // Context 0 = 0.67 * V[0] + 0.33 * V[1]
    // V[0] (Head0) = [1, 0]
    // V[1] (Head0) = [0, 1]
    // C0 = [0.67, 0.33]
    
    // Same for Head 1.
    // Concat: [0.67, 0.33, 0.67, 0.33]
    // Output Projection (Identity) -> Output = [0.67, 0.33, 0.67, 0.33]
    
    mha.forward(input_seq, input_seq, input_seq, output_mha);
    
    // Check output for pos 0
    // Expected rough: 0.669, 0.331, 0.669, 0.331
    // Check output for pos 1 (Symmetric)
    // Scores [0, 0.707] -> Softmax p0=0.331, p1=0.669
    // C1 = 0.33*V[0] + 0.67*V[1] = [0.33, 0.67]
    // Output 1: [0.331, 0.669, 0.331, 0.669]
    
    float exp_p = std::exp(1.0f / std::sqrt(2.0f)); 
    float exp_0 = std::exp(0.0f);
    float sum_e = exp_p + exp_0;
    float p_high = exp_p / sum_e; // ~0.669
    float p_low = exp_0 / sum_e;  // ~0.331
    
    float expected_mha_0[MHA_DIM] = {p_high, p_low, p_high, p_low};
    float expected_mha_1[MHA_DIM] = {p_low, p_high, p_low, p_high};
    
    pass = true;
    for(int j=0; j<MHA_DIM; ++j) {
        if(std::abs(output_mha[0][j] - expected_mha_0[j]) > 0.001f) {
            pass = false;
            std::cout << "MHA Pos0 Mismatch at " << j << ": Exp " << expected_mha_0[j] << " Got " << output_mha[0][j] << std::endl;
        }
        if(std::abs(output_mha[1][j] - expected_mha_1[j]) > 0.001f) {
            pass = false;
            std::cout << "MHA Pos1 Mismatch at " << j << ": Exp " << expected_mha_1[j] << " Got " << output_mha[1][j] << std::endl;
        }
    }
    
    if(pass) std::cout << "Test Case 6 (MHA): PASSED!" << std::endl;
    else { std::cout << "Test Case 6: FAILED!" << std::endl; return 1; }

    // --- Test Case 7: MHA Distinct (Sharp vs Soft Attention) ---
    std::cout << "\nRunning Test Case 7 (MHA Distinct Heads)..." << std::endl;
    // We reuse the 'mha' instance. Weights are still Identity.
    // Logic:
    // Head 0 (Dim 0-1): Input values are large (10, 0). 
    //      Dot product ~70.7. Softmax will be ~1.0 for the matching token, 0 for others.
    //      Behaves like "Hard Attention" (Copying the vector).
    // Head 1 (Dim 2-3): Input values are small (1, 0).
    //      Dot product ~0.7. Softmax will be mixed (~0.67, 0.33).
    //      Behaves like "Soft Attention" (Mixing vectors).
    
    float input_sharp_soft[MHA_SEQ][MHA_DIM] = {
        {10.0f, 0.0f,  1.0f, 0.0f}, // Pos 0
        { 0.0f, 10.0f, 0.0f, 1.0f}  // Pos 1
    };
    // Expected Output:
    // Pos 0:
    //   Head 0: Attends strictly to Pos 0 (itself). V=[10, 0]. -> [10, 0]
    //   Head 1: Attends Softly (0.67 to self, 0.33 to other). V=[1, 0] and [0, 1]. -> [0.67, 0.33]
    //   Result: [10.0, 0.0, 0.67, 0.33]
    // Pos 1:
    //   Head 0: Attends strictly to Pos 1. V=[0, 10]. -> [0, 10]
    //   Head 1: Attends Softly. -> [0.33, 0.67]
    //   Result: [0.0, 10.0, 0.33, 0.67]
    
    mha.forward(input_sharp_soft, input_sharp_soft, input_sharp_soft, output_mha);
    
    // Check Pos 0
    float expected_7_p0[MHA_DIM] = {10.0f, 0.0f, p_high, p_low};
    float expected_7_p1[MHA_DIM] = {0.0f, 10.0f, p_low, p_high};
    
    pass = true;
    for(int j=0; j<MHA_DIM; ++j) {
        // Use wider tolerance for Head 0 logic since 1.0 vs 0.9999...
        if(std::abs(output_mha[0][j] - expected_7_p0[j]) > 0.01f) {
            pass = false;
            std::cout << "MHA TC7 Pos0 Mismatch at " << j << ": Exp " << expected_7_p0[j] << " Got " << output_mha[0][j] << std::endl;
        }
        if(std::abs(output_mha[1][j] - expected_7_p1[j]) > 0.01f) {
            pass = false;
            std::cout << "MHA TC7 Pos1 Mismatch at " << j << ": Exp " << expected_7_p1[j] << " Got " << output_mha[1][j] << std::endl;
        }
    }

    if(pass) std::cout << "Test Case 7 (MHA Distinct): PASSED!" << std::endl;
    else { std::cout << "Test Case 7: FAILED!" << std::endl; return 1; }

    std::cout << "\nAll Test Cases PASSED!" << std::endl;
    return 0;
}
