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
  // Constructor (optional, can receive weights/bias or load them separately)
  FCLayer() {
    // Initialize weights and bias to 0 or random for testing if needed
    // In a real HLS scenario, these might be loaded from BRAM or Memory
    // for (int i = 0; i < OUT_DIM; ++i) {
    //     bias[i] = 0;
    //     for (int j = 0; j < IN_DIM; ++j) {
    //         weights[i][j] = 0;
    //     }
    // }
  }

  // Function to load weights manually (for testing/setup)
  void setWeights(const T w[OUT_DIM][IN_DIM]) {
    for (int i = 0; i < OUT_DIM; ++i) {
      for (int j = 0; j < IN_DIM; ++j) {
        weights[i][j] = w[i][j];
      }
    }
  }

  // Function to load bias manually
  void setBias(const T b[OUT_DIM]) {
    for (int i = 0; i < OUT_DIM; ++i) {
      bias[i] = b[i];
    }
  }

  // Forward pass: y = Wx + b
  // input: array of size IN_DIM
  // output: array of size OUT_DIM
  void forward(const T input[IN_DIM], T output[OUT_DIM]) {
#pragma HLS INLINE off
    // #pragma HLS ARRAY_PARTITION variable=weights complete dim=2

    for (int i = 0; i < OUT_DIM; ++i) {
      // #pragma HLS UNROLL
      T sum = bias[i];
      for (int j = 0; j < IN_DIM; ++j) {
#pragma HLS PIPELINE
        sum += weights[i][j] * input[j];
      }
      output[i] = sum;
    }
  }
};

#endif // FCLAYER_H
