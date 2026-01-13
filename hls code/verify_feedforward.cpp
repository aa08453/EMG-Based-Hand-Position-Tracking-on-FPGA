#include <iostream>
#include "FeedForward.h"
#include "TestUtils.h"

#define V_SEQ_LEN 100
#define V_D_MODEL 128
#define V_D_FF 256

int main() {
    std::cout << "--- Verifying FeedForward ---" << std::endl;

    // Weights
    static float w1[V_D_FF][V_D_MODEL];
    static float b1[V_D_FF];
    static float w2[V_D_MODEL][V_D_FF];
    static float b2[V_D_MODEL];

    static float input[V_SEQ_LEN][V_D_MODEL]; // Python processes batch x seq x dim. FF is pointwise.
    static float output[V_SEQ_LEN][V_D_MODEL];
    static float expected[V_SEQ_LEN][V_D_MODEL];

    // Load Data
    // Note: Python saves linear weights as [Out, In]. C++ FCLayer expects [Out][In]. So it matches directly.
    if (!load_data("verification_data/ff_w1.txt", (float*)w1, V_D_FF*V_D_MODEL)) return 1;
    if (!load_data("verification_data/ff_b1.txt", b1, V_D_FF)) return 1;
    if (!load_data("verification_data/ff_w2.txt", (float*)w2, V_D_MODEL*V_D_FF)) return 1;
    if (!load_data("verification_data/ff_b2.txt", b2, V_D_MODEL)) return 1;
    
    // Input/Output
    if (!load_data("verification_data/ff_input.txt", (float*)input, V_SEQ_LEN*V_D_MODEL)) return 1;
    if (!load_data("verification_data/ff_output.txt", (float*)expected, V_SEQ_LEN*V_D_MODEL)) return 1;


    // Instantiate
    FeedForward<float, V_D_MODEL, V_D_FF> ff;
    ff.set_fc1_weights(w1, b1);
    ff.set_fc2_weights(w2, b2);

    // Forward
    // FF is position-wise, so we loop over sequence length here 
    // because FeedForward class processes ONE vector [D_MODEL] at a time (as defined in my header).
    // wait, let me check FeedForward.h again.
    // Yes: void forward(const T input[D_MODEL], T output[D_MODEL])
    
    // But PyTorch processes the whole sequence [1, 100, 128].
    // So we loop.
    
    float max_diff = 0;
    bool all_pass = true;

    for(int t=0; t<V_SEQ_LEN; ++t) {
        ff.forward(input[t], output[t]);
    }

    if (compare_data((float*)output, (float*)expected, V_SEQ_LEN*V_D_MODEL, 1e-4f)) {
        return 0;
    } else {
        return 1;
    }
}
