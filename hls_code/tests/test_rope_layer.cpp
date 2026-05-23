#include "../config.h"
#include "../types.h"
#include <iostream>
#include <cmath>

#ifdef __ROPE__

void rope_forward(const data_t input[], data_t output[]); 

int main() {
  static data_t input[SEQ_LEN * ROPE_HALF_DIM];
  static data_t output[SEQ_LEN * ROPE_HALF_DIM];

  for (int i = 0; i < SEQ_LEN * ROPE_HALF_DIM; ++i) {
    input[i] = (data_t)((float)std::cos(i * 0.02f));
  }

  rope_forward(input, output);

  float max_val = -1e9;
  for (int i = 0; i < SEQ_LEN * ROPE_HALF_DIM; ++i) {
    float v = (float)output[i];
    if (std::isnan(v)) {
      std::cerr << "FAILED: NaN in rope output\n";
      return 1;
    }
    if (v > max_val) max_val = v;
  }

  std::cout << "rope_layer: OK max=" << max_val << "\n";
  return 0;
}

#endif