#ifndef MOCK_HLS_MATH_H
#define MOCK_HLS_MATH_H

// Mock for hls_math.h — provides software equivalents of HLS math functions.
// Used when compiling on Windows/VS Code with USE_MOCK_HLS defined.

#include <cmath>

namespace hls {
    inline float rsqrt(float x) {
        return 1.0f / std::sqrt(x);
    }
    inline float exp(float x) {
        return std::exp(x);
    }
    inline float sqrt(float x) {
        return std::sqrt(x);
    }
    inline float log(float x) {
        return std::log(x);
    }
}

#endif // MOCK_HLS_MATH_H