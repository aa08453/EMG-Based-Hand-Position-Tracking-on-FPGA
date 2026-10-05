#include "../config.h"
#include "../types.h"
#include <iostream>
#include <cmath>

#ifdef __MHA__

void mha_forward(
    const data_t q_rot_flat[],   // size [SEQ_LEN * D_MODEL]
    const data_t k_rot_flat[],   // size [SEQ_LEN * D_MODEL]
    const data_t v_origin_flat[],// size [SEQ_LEN * D_MODEL]
    data_t output_flat[]         // size [SEQ_LEN * D_MODEL]
);

int main() {
  static data_t q[SEQ_LEN * D_MODEL];
  static data_t k[SEQ_LEN * D_MODEL];
  static data_t v[SEQ_LEN * D_MODEL];
  static data_t out[SEQ_LEN * D_MODEL];

  for (int i = 0; i < SEQ_LEN * D_MODEL; ++i) {
    q[i] = (data_t)((float)std::sin(i * 0.001f));
    k[i] = (data_t)((float)std::cos(i * 0.001f));
    v[i] = (data_t)((float)((i % 7) - 3) * 0.1f);
  }

  mha_forward(q, k, v, out);

  float s = 0;
  for (int i = 0; i < SEQ_LEN * D_MODEL; ++i) {
    float vout = (float)out[i];
    if (std::isnan(vout)) {
      std::cerr << "FAILED: NaN in mha output\n";
      return 1;
    }
    s += vout;
  }
  std::cout << "mha_layer: OK sum=" << s << "\n";
  return 0;
}

#endif
