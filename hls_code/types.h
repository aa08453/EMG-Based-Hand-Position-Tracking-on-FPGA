#ifndef TYPES_H
#define TYPES_H
#ifdef __SYNTHESIS__
#include <ap_fixed.h>
#else
#include "mock_ap_fixed.h"
#endif

// Define fixed-point type: 8 bits total, 4 bits integer
// Range: [-16, 16), precision: 0.0625
typedef ap_fixed<8, 4> data_t;

#endif
