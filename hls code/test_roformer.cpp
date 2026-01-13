#include <iostream>
#include <cmath>
#include "RoFormer.h"

// Tiny RoFormer Configuration for Testing
#define T_IN_CH 2
#define T_SEQ 4
#define T_D_MODEL 4
#define T_HEADS 2
#define T_D_FF 4
#define T_LAYERS 2
#define T_OUT_DIM 2

int main() {
    std::cout << "Starting Full RoFormer Model Test..." << std::endl;

    RoFormer<float, T_IN_CH, T_SEQ, T_D_MODEL, T_HEADS, T_D_FF, T_LAYERS, T_OUT_DIM> roformer;

    // --- Weight Initialization ---
    // For this test, we care about data flowing through.
    // We will initialize:
    // Conv1d: Identity-ish. Channel 0 -> Dim 0. Channel 1 -> Dim 1. (Rest 0).
    // Blocks: Identity. (Weights = Eye, Bias = 0)
    // Final FC: Identity-ish. Dim 0 -> Out 0. Dim 1 -> Out 1. Global Pool will average the seq.

    // 1. Conv1d Weights
    // [D_MODEL][IN_CH][KERNEL]
    // Out0 <- In0 Center
    // Out1 <- In1 Center
    // Out2, Out3 = 0
    float w_conv[T_D_MODEL][T_IN_CH][3] = {0};
    w_conv[0][0][1] = 1.0f;
    w_conv[1][1][1] = 1.0f;
    float b_conv[T_D_MODEL] = {0};
    
    roformer.set_conv_weights(w_conv);
    roformer.set_conv_bias(b_conv);

    // 2. Layers (Set Identity for both layers)
    for(int l=0; l<T_LAYERS; ++l) {
        // MHA
        float eye[T_D_MODEL][T_D_MODEL] = {0};
        for(int i=0; i<T_D_MODEL; ++i) eye[i][i] = 1.0f;
        float zero_b[T_D_MODEL] = {0};
        
        roformer.get_layer(l).set_mha_q(eye, zero_b);
        roformer.get_layer(l).set_mha_k(eye, zero_b);
        roformer.get_layer(l).set_mha_v(eye, zero_b);
        roformer.get_layer(l).set_mha_o(eye, zero_b);
        
        // FF
        float w_ff1[T_D_FF][T_D_MODEL] = {0};
        for(int i=0; i<T_D_MODEL; ++i) w_ff1[i][i] = 1.0f;
        float b_ff1[T_D_FF] = {0};
        roformer.get_layer(l).set_ff_1(w_ff1, b_ff1);
        
        float w_ff2[T_D_MODEL][T_D_FF] = {0};
        for(int i=0; i<T_D_MODEL; ++i) w_ff2[i][i] = 1.0f;
        float b_ff2[T_D_MODEL] = {0};
        roformer.get_layer(l).set_ff_2(w_ff2, b_ff2);
        
        // LNs (Gamma=1, Beta=0 is default)
    }

    // 3. Final FC
    // We want Output 0 to match Pooled 0.
    float w_fc[T_OUT_DIM][T_D_MODEL] = {0};
    w_fc[0][0] = 1.0f;
    w_fc[1][1] = 1.0f;
    float b_fc[T_OUT_DIM] = {0};

    roformer.set_final_fc_weights(w_fc);
    roformer.set_final_fc_bias(b_fc);

    // --- Input Data ---
    // Ch0: [1, 1, 1, 1]
    // Ch1: [2, 2, 2, 2]
    float input[T_IN_CH][T_SEQ] = {
        {1.0f, 1.0f, 1.0f, 1.0f},
        {2.0f, 2.0f, 2.0f, 2.0f}
    };
    float output[T_OUT_DIM];

    // --- Expected Flow ---
    // Conv Output:
    //   Seq t: [1, 2, 0, 0]
    
    // Layer 0:
    //   Identity-ish pass through.
    //   Note: RoPE will rotate [1, 2]. Q/K rotation. 
    //   But since Q=K=V=Input and weights are identity, MHA usually preserves "something" close to input feature wise if attention is uniform.
    //   With uniform input [1, 2, 0, 0] across all time steps, attention will be perfectly uniform (0.25 weight each).
    //   So MHA Output = Weighted Sum of V. V is constant [1, 2, 0, 0]. Sum is [1, 2, 0, 0].
    //   Residual 1: Input + MHA = [2, 4, 0, 0].
    //   LN1: Normalizes [2, 4, 0, 0]. Mean=1.5. Var=(0.25+6.25+2.25+2.25)/4 = 11/4 = 2.75. Std=1.65.
    //        Norm = [0.3, 1.5, -0.9, -0.9] (Roughly).
    //   FF: Linear -> ReLU -> Linear.
    //   Residual 2 + LN2.
    
    // As inputs are constant across time, the "Global Average Pooling" will just equal the value of any time step.
    
    roformer.forward(input, output);

    std::cout << "Computed Output: [ " << output[0] << ", " << output[1] << " ]" << std::endl;

    if(!std::isnan(output[0]) && !std::isnan(output[1])) {
        std::cout << "Test RoFormer PASSED (Sanity: Output is valid number)" << std::endl;
    } else {
        std::cout << "Test RoFormer FAILED (NaN)" << std::endl;
        return 1;
    }

    return 0;
}
