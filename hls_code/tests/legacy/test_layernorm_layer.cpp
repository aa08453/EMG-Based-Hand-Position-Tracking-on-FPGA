#include "../config.h"
#include "../types.h"
#include <iostream>
#include <cmath>

#ifdef __LN__

void ln_forward(
    const data_t input[], 
    data_t output[],
    const data_t gamma[],
    const data_t beta[]
);

int main() {
  static data_t input[D_MODEL];
  static data_t output[D_MODEL];
  static data_t gamma[D_MODEL];
  static data_t beta[D_MODEL];

  for (int i = 0; i < D_MODEL; ++i) {
    input[i] = (data_t)((float)(i % 5 - 2) * 0.2f);
    gamma[i] = (data_t)1.0f;
    beta[i] = (data_t)0.0f;
  }

  ln_forward(input, output, gamma, beta);

  float s = 0;
  for (int i = 0; i < D_MODEL; ++i) {
    float v = (float)output[i];
    if (std::isnan(v)) {
      std::cerr << "FAILED: NaN in layernorm output\n";
      return 1;
    }
    s += v;
  }

  std::cout << "layernorm_layer: OK avg=" << s / D_MODEL << "\n";
  return 0;
}

#endif