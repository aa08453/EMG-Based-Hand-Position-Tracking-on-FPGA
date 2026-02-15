#ifndef RELU_H
#define RELU_H

// Template parameters:
// T: Data type
// DIM: Dimension of the input/output array
template <typename T, int DIM> class ReLU {
public:
  ReLU() {}

  // Forward pass: y = max(0, x)
  void relu_forward(T input[DIM], T output[DIM]) {
#pragma HLS INLINE off

  RELU_LOOP:
    for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE
      if (input[i] > 0) {
        output[i] = input[i];
      } else {
        output[i] = 0;
      }
    }
  }

  // In-place version
  void relu_forward_inplace(T data[DIM]) {
#pragma HLS INLINE off
  RELU_INPLACE_LOOP:
    for (int i = 0; i < DIM; ++i) {
#pragma HLS PIPELINE
      if (data[i] < 0) {
        data[i] = 0;
      }
    }
  }
};

#endif // RELU_H
