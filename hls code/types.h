#ifndef TYPES_H
#define TYPES_H
#ifdef __SYNTHESIS__
#include <ap_fixed.h>
#else
#include "ap_fixed.h"
#endif

// Define fixed-point type: 16 bits total, 6 bits integer
typedef ap_fixed<16, 6> data_t;

#endif
