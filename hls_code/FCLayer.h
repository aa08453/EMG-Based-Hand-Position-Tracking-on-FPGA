#ifndef FCLAYER_H
#define FCLAYER_H

// Template parameters:
// T: Data type (e.g., float, double, or fixed-point type if using HLS libs)
// IN_DIM: Input dimension size
// OUT_DIM: Output dimension size
template <typename T, int IN_DIM, int OUT_DIM> class FCLayer {
private:
  T weights[OUT_DIM][IN_DIM];
  T bias[OUT_DIM];

public:
  FCLayer() {}

  // Function to load weights manually (for testing/setup)
  void setWeights(const T w[OUT_DIM][IN_DIM]) {
  FC_SET_WEIGHTS_ROW:
    for (int i = 0; i < OUT_DIM; ++i) {
    FC_SET_WEIGHTS_COL:
      for (int j = 0; j < IN_DIM; ++j) {
        weights[i][j] = w[i][j];
      }
    }
  }

  // Function to load bias manually
  void setBias(const T b[OUT_DIM]) {
  FC_SET_BIAS:
    for (int i = 0; i < OUT_DIM; ++i) {
      bias[i] = b[i];
    }
  }

  // Forward pass: y = Wx + b
  // input: array of size IN_DIM
  // output: array of size OUT_DIM
  void forward(const T input[IN_DIM], T output[OUT_DIM]) {
#pragma HLS INLINE off

  FC_OUTPUT:
    for (int i = 0; i < OUT_DIM; ++i) {
      T sum = bias[i];
    FC_MAC:
      for (int j = 0; j < IN_DIM; ++j) {
#pragma HLS PIPELINE
#pragma HLS BIND_OP variable = sum op = mul impl = dsp
#pragma HLS BIND_OP variable = sum op = add impl = dsp
        sum += weights[i][j] * input[j];
      }
      output[i] = sum;
    }
  }
};

#endif // FCLAYER_H