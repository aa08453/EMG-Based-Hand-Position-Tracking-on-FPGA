#ifndef TYPES_H
#define TYPES_H

// In local builds: compile with -DUSE_MOCK_HLS and the local ap_fixed.h proxy
// intercepts the include, routing to mock_ap_fixed.h.
// In Vitis HLS: the real <ap_fixed.h> is found via toolchain include paths.
// #ifndef _SYNTHESIS_
// #include "mock_ap_fixed.h"
// #else
#include <ap_fixed.h>
// #endif

// Define fixed-point types
typedef ap_fixed<10, 3> data_t;
typedef ap_fixed<8, 2> rope_t;
typedef ap_fixed<16, 2> pool_scale_t;
typedef ap_fixed<24,8> conv_sum_t;
typedef ap_fixed<28,10> mha_sum_t;
typedef ap_fixed<24, 12> ln_calc_t;

#endif
