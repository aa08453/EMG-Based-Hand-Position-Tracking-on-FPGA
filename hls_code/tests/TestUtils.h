#ifndef TESTUTILS_H
#define TESTUTILS_H

#include "../types.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

// Helper to load binary float data into your ap_fixed array
bool load_binary_weights(const std::string& filepath, data_t* weight_buffer, int num_elements) {
    std::ifstream file(filepath.c_str(), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << std::endl;
        return false;
    }

    // Read 32-bit floats exactly as MATLAB saved them
    std::vector<float> temp_buffer(num_elements);
    file.read(reinterpret_cast<char*>(temp_buffer.data()), num_elements * sizeof(float));
    
    // Cast and store into the ap_fixed hardware buffer
    for (int i = 0; i < num_elements; ++i) {
        weight_buffer[i] = (data_t)temp_buffer[i];
    }
    
    return true;
}
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

// Compare arrays using Max Diff and MSE
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

// Calculate Cosine Similarity
template <typename T1, typename T2>
float calculate_cosine_similarity(const T1* computed, const T2* expected, int size) {
    float dot_product = 0.0f;
    float norm_computed = 0.0f;
    float norm_expected = 0.0f;

    for (int i = 0; i < size; ++i) {
        float val_c = (float)computed[i];
        float val_e = (float)expected[i];
        
        dot_product += val_c * val_e;
        norm_computed += val_c * val_c;
        norm_expected += val_e * val_e;
    }

    if (norm_computed == 0.0f || norm_expected == 0.0f) {
        return 0.0f; // Prevent division by zero if an array is completely empty
    }

    return dot_product / (std::sqrt(norm_computed) * std::sqrt(norm_expected));
}

#endif // TESTUTILS_H