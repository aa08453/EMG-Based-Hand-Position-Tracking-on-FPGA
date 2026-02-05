#include "RotaryEmbedding.h"
#include "TestUtils.h"
#include <iostream>


// Config matching Python script
#define V_SEQ_LEN 400
#define V_D_MODEL 128

int main() {
  std::cout << "--- Verifying RotaryEmbedding ---" << std::endl;

  // Allocate Buffers
  static float input[V_SEQ_LEN][V_D_MODEL];
  static float output[V_SEQ_LEN][V_D_MODEL];
  static float expected[V_SEQ_LEN][V_D_MODEL];

  // Load Data
  if (!load_data("verification_data/rope_input.txt", (float *)input,
                 V_SEQ_LEN * V_D_MODEL))
    return 1;
  if (!load_data("verification_data/rope_output.txt", (float *)expected,
                 V_SEQ_LEN * V_D_MODEL))
    return 1;

  // Instantiate
  RotaryEmbedding<float, V_D_MODEL, V_SEQ_LEN> rope;

  // Forward
  rope.forward(input, output);

  // Compare
  if (compare_data((float *)output, (float *)expected, V_SEQ_LEN * V_D_MODEL,
                   1e-4f)) {
    return 0;
  } else {
    return 1;
  }
}
