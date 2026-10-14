#include "../config.h"
#include "../types.h"
#include <iostream>
#include <cmath>

#ifdef __FF__

void ff_forward(
    const data_t input[], // d_model
    data_t output[] // d_model
);

int main() {
  static data_t input[D_MODEL];
  static data_t output[D_MODEL];

  for (int i = 0; i < D_MODEL; ++i) {
    input[i] = (data_t)((float)(i % 7 - 3) * 0.1f);
  }

  ff_forward(input, output);

  float sum = 0;
  for (int i = 0; i < D_MODEL; ++i) {
    float v = (float)output[i];
    if (std::isnan(v)) {
      std::cerr << "FAILED: NaN in feedforward output\n";
      return 1;
    }
    sum += v;
  }

  std::cout << "feedforward_layer: OK sum=" << sum << "\n";
  return 0;
}

#endif