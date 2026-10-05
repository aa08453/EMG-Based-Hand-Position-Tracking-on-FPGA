/**
 * test_vitis_roformer.cpp
 * -----------------------
 * Automated layer-by-layer verification testbench for the RoFormer HLS kernel.
 *
 * How it works:
 *   1. Reads config_debug.h to determine which layer checkpoint is active
 *      (DEBUG_CONV1D, DEBUG_ROPE, DEBUG_MHA, DEBUG_FFN, DEBUG_POOL, or DEBUG_FULL).
 *   2. Loads the FULL-PRECISION input from verification_data/full_input.txt
 *      (matching what PyTorch used to generate the reference data).
 *   3. Calls RoFormer_Top() and compares against PyTorch reference arrays
 *      stored in weights/results.h.
 *   4. Reports MAE, RMSE, Max Error, Cosine Similarity, and pass/fail.
 *
 * Build (local mock):
 *   g++ -std=c++14 -O2 -DUSE_MOCK_HLS -I. -I./tests \
 *       -o test_vitis tests/test_vitis_roformer.cpp RoFormer_Top.cpp
 *
 * Build (Vitis C-sim):
 *   Vitis HLS will compile this as the testbench with RoFormer_Top as the DUT.
 *
 * To test a different layer:
 *   - Edit config_debug.h → uncomment the desired #define
 *   - Recompile and run
 */
#include "TestUtils.h"
#include "../types.h"
#include "../config.h"
#include "../weights/results.h"
#include "../weights/input_output.h"
#include "../config_debug.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <cstdio>
#include <sstream>

#ifdef _WIN32
  #include <direct.h>
  #define getcwd _getcwd
#else
  #include <unistd.h>
#endif

// Forward-declare the top-level kernel
void RoFormer_Top(const data_t input_axi[], data_t output_axi[]);

// ============================================================================
// Load input from verification_data/full_input.txt (full-precision floats)
// ============================================================================
static bool load_input_from_file(const std::string& filepath, data_t* buffer, int expected_count) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: Could not open input file: " << filepath << std::endl;
        return false;
    }
    
    int count = 0;
    float val;
    while (ifs >> val && count < expected_count) {
        buffer[count++] = (data_t)val;
    }
    
    if (count != expected_count) {
        std::cerr << "WARNING: Read " << count << " values, expected " << expected_count << std::endl;
    }
    
    std::cout << "Loaded " << count << " input values from " << filepath << std::endl;
    return count > 0;
}

// ============================================================================
// Mapping from DEBUG_LABEL → reference array pointer + tolerance
// ============================================================================
struct LayerConfig {
    const float* ref_ptr;
    int          num_elements;
    float        mae_threshold;
    float        cosine_threshold;
    const char*  description;
};

static LayerConfig get_layer_config(const char* label) {
    LayerConfig cfg = {nullptr, 0, 0.05f, 0.95f, "Unknown"};

    std::string s(label);
    if (s == "conv1d") {
        cfg.ref_ptr          = ref_conv1d_out;
        cfg.num_elements     = SEQ_LEN * OUT_CHANNELS;  // 25600
        cfg.mae_threshold    = 0.02f;    // Conv1D should be tight
        cfg.cosine_threshold = 0.98f;
        cfg.description      = "Conv1D Layer Output";
    }
    else if (s == "rope") {
        cfg.ref_ptr          = ref_l0_rope_out;
        cfg.num_elements     = SEQ_LEN * D_MODEL;       // 25600
        cfg.mae_threshold    = 0.05f;
        cfg.cosine_threshold = 0.95f;
        cfg.description      = "Conv1D + RoPE Output";
    }
    else if (s == "mha") {
        cfg.ref_ptr          = ref_l0_add_norm1_out;
        cfg.num_elements     = SEQ_LEN * D_MODEL;       // 25600
        cfg.mae_threshold    = 0.10f;    // MHA accumulates quantization error
        cfg.cosine_threshold = 0.90f;
        cfg.description      = "Conv1D + RoPE + MHA + AddNorm1 Output";
    }
    else if (s == "ffn") {
        cfg.ref_ptr          = ref_l0_add_norm2_out;
        cfg.num_elements     = SEQ_LEN * D_MODEL;       // 25600
        cfg.mae_threshold    = 0.15f;    // Further accumulation expected
        cfg.cosine_threshold = 0.85f;
        cfg.description      = "Full Transformer Block (AddNorm2) Output";
    }
    else if (s == "pool") {
        cfg.ref_ptr          = ref_pool_out;
        cfg.num_elements     = D_MODEL;                  // 128
        cfg.mae_threshold    = 0.15f;
        cfg.cosine_threshold = 0.85f;
        cfg.description      = "Mean Pooling Output";
    }
    else if (s == "full") {
        cfg.ref_ptr          = output;
        cfg.num_elements     = OUT_DIM;                  // 10
        cfg.mae_threshold    = 0.20f;    // Full pipeline error
        cfg.cosine_threshold = 0.80f;
        cfg.description      = "Final FC Output (10 joints)";
    }

    return cfg;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "====================================================" << std::endl;
    std::cout << " RoFormer HLS Verification Testbench" << std::endl;
    std::cout << "====================================================" << std::endl;
    std::cout << " DEBUG CHECKPOINT  : " << DEBUG_LABEL << std::endl;
    std::cout << " EXPECTED ELEMENTS : " << DEBUG_OUT_LEN << std::endl;

    // ---- Get layer configuration ----
    LayerConfig cfg = get_layer_config(DEBUG_LABEL);

    if (cfg.ref_ptr == nullptr) {
        std::cerr << "ERROR: Unknown DEBUG_LABEL '" << DEBUG_LABEL << "'" << std::endl;
        return 1;
    }

    std::cout << " LAYER DESCRIPTION : " << cfg.description << std::endl;
    std::cout << " MAE THRESHOLD     : " << cfg.mae_threshold << std::endl;
    std::cout << " COSINE THRESHOLD  : " << cfg.cosine_threshold << std::endl;
    std::cout << "====================================================" << std::endl;

    // ---- Sanity check ----
    if (cfg.num_elements != DEBUG_OUT_LEN) {
        std::cerr << "WARNING: num_elements mismatch! Config says " 
                  << cfg.num_elements << " but DEBUG_OUT_LEN is " 
                  << DEBUG_OUT_LEN << std::endl;
    }

    // ---- Load full-precision input ----
    // The input must match what PyTorch used to generate the reference data.
    // full_input.txt has shape [1, IN_CHANNELS, SEQ_LEN] = 2400 floats
    // static data_t hw_input[IN_CHANNELS * SEQ_LEN] = {0};
    
    // // Try multiple paths to find the input file
    // const char* input_paths[] = {
    //     "verification_data/full_input.txt",
    //     "../verification_data/full_input.txt",
    //     "./verification_data/full_input.txt",
    //     "/home/abdullah/Desktop/FYP/EMG-Based-Hand-Pose-Tracking-on-FPGA/updated_hls_code/hls_code/verification_data/full_input.txt",
    //     nullptr
    // };
    
    // bool input_loaded = false;
    // for (int p = 0; input_paths[p] != nullptr; ++p) {
    //     if (load_input_from_file(input_paths[p], hw_input, IN_CHANNELS * SEQ_LEN)) {
    //         input_loaded = true;
    //         break;
    //     }
    // }
    
    // if (!input_loaded) {
    //     std::cerr << "FATAL: Could not load input data. Ensure verification_data/full_input.txt exists." << std::endl;
    //     return 1;
    // }

    // ---- Output buffer ----
    static data_t hw_output[SEQ_LEN * D_MODEL] = {0};

    // ---- Run the model ----
    std::cout << "\nRunning RoFormer_Top hardware simulation..." << std::endl;
    RoFormer_Top(input, hw_output);
    std::cout << "Simulation complete.\n" << std::endl;

    // ---- Perform Comparison ----
    std::cout << "--- ELEMENT-WISE COMPARISON (first 20 + outliers) ---" << std::endl;
    printf("%-6s | %-12s | %-12s | %-12s | %s\n", 
           "Idx", "HW Output", "Ref (PyT)", "Diff", "Status");
    std::cout << "------------------------------------------------------" << std::endl;

    float sum_sq = 0.0f, sum_abs = 0.0f, max_err = 0.0f;
    int n_outliers = 0;
    int max_err_idx = 0;

    for (int i = 0; i < cfg.num_elements; ++i) {
        float val  = (float)hw_output[i];
        float ref  = cfg.ref_ptr[i];
        float diff = std::abs(val - ref);

        sum_abs += diff;
        sum_sq  += diff * diff;
        if (diff > max_err) {
            max_err = diff;
            max_err_idx = i;
        }

        // Print first 20 elements and any major discrepancies
        bool is_outlier = (diff > cfg.mae_threshold * 2.0f);
        if (is_outlier) ++n_outliers;

        if (i < 20 || is_outlier) {
            const char* status = "";
            if (diff > cfg.mae_threshold * 2.0f) status = "[!!]";
            else if (diff > cfg.mae_threshold) status = "[!]";
            printf("%-6d | %12.6f | %12.6f | %12.6f | %s\n",
                   i, val, ref, diff, status);
        }
    }

    // ---- Cosine Similarity ----
    float dot_product = 0.0f, norm_hw = 0.0f, norm_ref = 0.0f;
    for (int i = 0; i < cfg.num_elements; ++i) {
        float h = (float)hw_output[i];
        float r = cfg.ref_ptr[i];
        dot_product += h * r;
        norm_hw     += h * h;
        norm_ref    += r * r;
    }
    float cosine_sim = 0.0f;
    if (norm_hw > 0.0f && norm_ref > 0.0f) {
        cosine_sim = dot_product / (std::sqrt(norm_hw) * std::sqrt(norm_ref));
    }

    // ---- Final Metrics ----
    float mae  = sum_abs / cfg.num_elements;
    float rmse = std::sqrt(sum_sq / cfg.num_elements);

    std::cout << "\n=====================================================" << std::endl;
    std::cout << "              VERIFICATION SUMMARY" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "Layer           : " << cfg.description << std::endl;
    std::cout << "Debug Label     : " << DEBUG_LABEL << std::endl;
    std::cout << "Total Compared  : " << cfg.num_elements << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    printf(     "MAE             : %.6f  (threshold: %.4f)\n", mae, cfg.mae_threshold);
    printf(     "RMSE            : %.6f\n", rmse);
    printf(     "Max Abs Error   : %.6f  (at index %d)\n", max_err, max_err_idx);
    printf(     "Cosine Sim      : %.6f  (threshold: %.4f)\n", cosine_sim, cfg.cosine_threshold);
    printf(     "Outliers (>2x)  : %d\n", n_outliers);

    // ---- Pass/Fail logic ----
    bool mae_pass    = (mae < cfg.mae_threshold);
    bool cosine_pass = (cosine_sim > cfg.cosine_threshold);
    bool overall     = mae_pass && cosine_pass;

    std::cout << "-----------------------------------------------------" << std::endl;
    printf(     "MAE Check       : %s\n", mae_pass ? "PASSED" : "FAILED");
    printf(     "Cosine Check    : %s\n", cosine_pass ? "PASSED" : "FAILED");
    std::cout << "=====================================================" << std::endl;
    std::cout << "OVERALL RESULT  : " << (overall ? "PASSED" : "FAILED") << std::endl;
    std::cout << "=====================================================" << std::endl;
    // Right after RoFormer_Top returns, before comparison:

    // ---- Dump HW output to file for Python cross-validation ----
    {
        std::string dump_file = std::string("hw_output_") + DEBUG_LABEL + ".txt";
        std::ofstream ofs(dump_file);
        if (ofs.is_open()) {
            for (int i = 0; i < cfg.num_elements; ++i) {
                ofs << (float)hw_output[i];
                if (i < cfg.num_elements - 1) ofs << " ";
            }
            ofs << std::endl;
            ofs.close();
            std::cout << "\nHW output saved to: " << dump_file << std::endl;
        }
    }



    return overall ? 0 : 1;
}