#ifndef TESTUTILS_H
#define TESTUTILS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

// Helper to load data from file
template <typename T>
bool load_data(const std::string& filename, T* buffer, int size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < size; ++i) {
        if (!(file >> buffer[i])) {
            std::cerr << "Error: Not enough data in " << filename << " (Expected " << size << ")" << std::endl;
            return false;
        }
    }
    return true;
}

// Compare arrays
template <typename T>
bool compare_data(const T* output, const T* expected, int size, float tolerance = 1e-4) {
    float max_diff = 0.0f;
    float mse = 0.0f;
    bool pass = true;

    for (int i = 0; i < size; ++i) {
        float diff = std::abs((float)output[i] - (float)expected[i]);
        if (diff > max_diff) max_diff = diff;
        mse += diff * diff;
        
        if (diff > tolerance) {
            pass = false;
        }
    }
    mse /= size;

    std::cout << "  Size: " << size << std::endl;
    std::cout << "  Max Diff: " << max_diff << " (Tolerance: " << tolerance << ")" << std::endl;
    std::cout << "  MSE: " << mse << std::endl;
    
    if (pass) std::cout << "  Result: PASSED" << std::endl;
    else std::cout << "  Result: FAILED" << std::endl;

    return pass;
}

#endif // TESTUTILS_H
