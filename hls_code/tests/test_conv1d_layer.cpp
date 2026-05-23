#include "../config.h"
#include "../types.h"
#include <iostream>
#include <cmath>

#ifdef __CONV__


void conv1d_forward(const data_t input[], data_t output[]);

int main() {
  static data_t input[IN_CHANNELS * SEQ_LEN];
  static data_t output[OUT_CHANNELS * SEQ_LEN];

  for (int i = 0; i < IN_CHANNELS * SEQ_LEN; ++i) {
    input[i] = (data_t)((float)std::sin(i * 0.01f));
  }

  conv1d_forward(input, output);

  float sum = 0;
  for (int i = 0; i < OUT_CHANNELS * SEQ_LEN; ++i) {
    float v = (float)output[i];
    if (std::isnan(v)) {
      std::cerr << "FAILED: NaN in conv1d output\n";
      return 1;
    }
    sum += v;
  }

  std::cout << "conv1d_layer: OK sum=" << sum << "\n";
  return 0;
}


#endif