#include "Conv1dLayer.h"
#include "TestUtils.h"
#include <iostream>

// Config matching Python script
#define V_SEQ_LEN 400
#define V_D_MODEL 128
#define V_IN_CHANNELS 12

int main() {
  std::cout << "--- Verifying Conv1dLayer ---" << std::endl;

  // Allocate Buffers
  // Weights: [D_MODEL][IN_CH][3]
  static float weights[V_D_MODEL][V_IN_CHANNELS][3];
  static float bias[V_D_MODEL];

  // Input: [IN_CH][SEQ_LEN]
  static float input[V_IN_CHANNELS][V_SEQ_LEN];

  // Output: [SEQ_LEN][D_MODEL] (Transposed in C++)
  static float output[V_SEQ_LEN][V_D_MODEL];
  static float expected[V_SEQ_LEN][V_D_MODEL];

  // Load Data
  if (!load_data("verification_data/conv1d_weights.txt", (float *)weights,
                 V_D_MODEL * V_IN_CHANNELS * 3))
    return 1;
  if (!load_data("verification_data/conv1d_bias.txt", bias, V_D_MODEL))
    return 1;
  if (!load_data("verification_data/conv1d_input.txt", (float *)input,
                 V_IN_CHANNELS * V_SEQ_LEN))
    return 1;
  if (!load_data("verification_data/conv1d_output.txt", (float *)expected,
                 V_SEQ_LEN * V_D_MODEL))
    return 1;

  // Instantiate & Init
  Conv1dLayer<float, V_IN_CHANNELS, V_D_MODEL, V_SEQ_LEN> conv;
  conv.setWeights(weights);
  conv.setBias(bias);

  // Forward
  conv.forward(input, output);

  // Compare
  if (compare_data((float *)output, (float *)expected, V_SEQ_LEN * V_D_MODEL,
                   1e-4f)) {
    return 0;
  } else {
    return 1;
  }
}
