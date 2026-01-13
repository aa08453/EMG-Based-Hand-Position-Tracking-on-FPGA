#include <iostream>
#include "RoFormerBlock.h"
#include "TestUtils.h"

// Config
#define V_SEQ_LEN 100
#define V_D_MODEL 128
#define V_NUM_HEADS 4
#define V_D_FF 256

int main() {
    std::cout << "--- Verifying RoFormerBlock ---" << std::endl;

    // Weights buffers
    static float mha_q_w[V_D_MODEL][V_D_MODEL], mha_q_b[V_D_MODEL];
    static float mha_k_w[V_D_MODEL][V_D_MODEL], mha_k_b[V_D_MODEL];
    static float mha_v_w[V_D_MODEL][V_D_MODEL], mha_v_b[V_D_MODEL];
    static float mha_o_w[V_D_MODEL][V_D_MODEL], mha_o_b[V_D_MODEL];
    
    static float ln1_g[V_D_MODEL], ln1_b[V_D_MODEL];
    static float ln2_g[V_D_MODEL], ln2_b[V_D_MODEL];

    static float ff_w1[V_D_FF][V_D_MODEL], ff_b1[V_D_FF];
    static float ff_w2[V_D_MODEL][V_D_FF], ff_b2[V_D_MODEL];

    static float input[V_SEQ_LEN][V_D_MODEL];
    static float output[V_SEQ_LEN][V_D_MODEL];
    static float expected[V_SEQ_LEN][V_D_MODEL];

    // Load Data
    // MHA
    if (!load_data("verification_data/blk_mha_q_w.txt", (float*)mha_q_w, V_D_MODEL*V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_mha_q_b.txt", mha_q_b, V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_mha_k_w.txt", (float*)mha_k_w, V_D_MODEL*V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_mha_k_b.txt", mha_k_b, V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_mha_v_w.txt", (float*)mha_v_w, V_D_MODEL*V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_mha_v_b.txt", mha_v_b, V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_mha_o_w.txt", (float*)mha_o_w, V_D_MODEL*V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_mha_o_b.txt", mha_o_b, V_D_MODEL)) return 1;

    // LN
    if (!load_data("verification_data/blk_ln1_g.txt", ln1_g, V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_ln1_b.txt", ln1_b, V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_ln2_g.txt", ln2_g, V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_ln2_b.txt", ln2_b, V_D_MODEL)) return 1;

    // FF
    if (!load_data("verification_data/blk_ff_w1.txt", (float*)ff_w1, V_D_FF*V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_ff_b1.txt", ff_b1, V_D_FF)) return 1;
    if (!load_data("verification_data/blk_ff_w2.txt", (float*)ff_w2, V_D_MODEL*V_D_FF)) return 1;
    if (!load_data("verification_data/blk_ff_b2.txt", ff_b2, V_D_MODEL)) return 1;

    // IO
    if (!load_data("verification_data/blk_input.txt", (float*)input, V_SEQ_LEN*V_D_MODEL)) return 1;
    if (!load_data("verification_data/blk_output.txt", (float*)expected, V_SEQ_LEN*V_D_MODEL)) return 1;

    // Instantiate
    RoFormerBlock<float, V_D_MODEL, V_NUM_HEADS, V_SEQ_LEN, V_D_FF> block;
    
    // Set Weights
    block.set_mha_q(mha_q_w, mha_q_b);
    block.set_mha_k(mha_k_w, mha_k_b);
    block.set_mha_v(mha_v_w, mha_v_b);
    block.set_mha_o(mha_o_w, mha_o_b);

    block.set_ln1(ln1_g, ln1_b);
    block.set_ln2(ln2_g, ln2_b);

    block.set_ff_1(ff_w1, ff_b1);
    block.set_ff_2(ff_w2, ff_b2);

    // Forward
    block.forward(input, output);

    // Compare
    // Due to fp accumulation differences in MHA and LayerNorm, tolerance might need to be e.g. 1e-3 or 1e-4
    if (compare_data((float*)output, (float*)expected, V_SEQ_LEN*V_D_MODEL, 1e-3f)) {
        return 0;
    } else {
        return 1;
    }
}
