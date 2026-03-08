#include "RoFormer.h"
#include "types.h"

// Define the precise configuration for hardware
// These MUST match your desired FPGA resource constraints and Python Model
#define IN_CH 12
#define SEQ_LEN 400
#define D_MODEL 128
#define NUM_HEADS 4
#define D_FF 256
#define NUM_LAYERS 1
#define OUT_DIM 10

// Instantiate the static model
// Making it static ensures weights might be treated as ROM/BRAM if initialized
static RoFormer<data_t, IN_CH, SEQ_LEN, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS,
                OUT_DIM>
    roformer_inst;

// Helper to load weights for verification
// This function allows the testbench to inject weights into the static instance
void init_roformer_weights(
    const data_t conv_w[D_MODEL][IN_CH][3], const data_t conv_b[D_MODEL],
    // Layers (We will pass pointers or handle inside? Passing massive args is
    // bad) Better: Allow testbench to loop and set. We can expose the instance
    // reference? No, static. Let's make a function that takes an index and
    // weights for that layer.
    const data_t fc_w[OUT_DIM][D_MODEL], const data_t fc_b[OUT_DIM]) {
  roformer_inst.set_conv_weights(conv_w);
  roformer_inst.set_conv_bias(conv_b);
  roformer_inst.set_final_fc_weights(fc_w);
  roformer_inst.set_final_fc_bias(fc_b);
}

// Function to set layer weights
// l: layer index
void set_roformer_layer_weights(
    int l, const data_t q_w[D_MODEL][D_MODEL], const data_t q_b[D_MODEL],
    const data_t k_w[D_MODEL][D_MODEL], const data_t k_b[D_MODEL],
    const data_t v_w[D_MODEL][D_MODEL], const data_t v_b[D_MODEL],
    const data_t o_w[D_MODEL][D_MODEL], const data_t o_b[D_MODEL],
    const data_t ln1_g[D_MODEL], const data_t ln1_b[D_MODEL],
    const data_t ln2_g[D_MODEL], const data_t ln2_b[D_MODEL],
    const data_t ff_w1[D_FF][D_MODEL], const data_t ff_b1[D_FF],
    const data_t ff_w2[D_MODEL][D_FF], const data_t ff_b2[D_MODEL]) {
  auto &layer = roformer_inst.get_layer(l);
  layer.set_mha_q(q_w, q_b);
  layer.set_mha_k(k_w, k_b);
  layer.set_mha_v(v_w, v_b);
  layer.set_mha_o(o_w, o_b);
  layer.set_ln1(ln1_g, ln1_b);
  layer.set_ln2(ln2_g, ln2_b);
  layer.set_ff_1(ff_w1, ff_b1);
  layer.set_ff_2(ff_w2, ff_b2);
}

// The Top-Level Function
// This is what Vitis will synthesize into RTL (Verilog/VHDL)
// Arguments become the hardware ports (AXI, Bram, etc using #pragma HLS
// INTERFACE)
void RoFormer_Top(const data_t input[IN_CH][SEQ_LEN], data_t output[OUT_DIM]) {
#pragma HLS INTERFACE mode = m_axi port = input bundle = gmem0 depth =         \
    4800 offset = slave
#pragma HLS INTERFACE mode = m_axi port = output bundle = gmem1 depth =        \
    10 offset = slave
#pragma HLS INTERFACE mode = s_axilite port = input
#pragma HLS INTERFACE mode = s_axilite port = output
#pragma HLS INTERFACE mode = s_axilite port = return

  // Call the class method
  roformer_inst.roformer_forward(input, output);
}
