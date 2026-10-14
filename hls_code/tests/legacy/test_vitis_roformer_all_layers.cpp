/**
 * test_vitis_roformer_all_layers.cpp
 * -----------------------------------
 * Dumps outputs for each major layer stage (conv1d, rope, mha+add&norm1,
 * ffn+add&norm2, pool, full) into vitis_outputs/*.txt.
 *
 * This allows Python to verify layer-by-layer against:
 *   verification_data/{conv1d_out,l0_rope_out,l0_add_norm1_out,...}.txt
 *
 * Build (local mock):
 *   g++ -std=c++14 -O2 -DUSE_MOCK_HLS -I. -I./tests \
 *     -o test_vitis_roformer_all_layers \
 *     tests/test_vitis_roformer_all_layers.cpp
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "Conv1dLayer.h"
#include "FFNorm.h"
#include "FCLayer.h"
#include "MHANorm.h"
#include "Pooling.h"
#include "RotaryEmbedding.h"
#include "types.h"
#include "config.h"

#include "./weights/fc_weights.h"
#include "./weights/input_output.h"

static bool load_input_from_file(const std::string &filepath, data_t *buffer, int expected_count) {
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
  return count > 0;
}

template <typename T>
static void write_vector_1_per_line(const std::string &filepath, const T *buffer, int n) {
  std::ofstream ofs(filepath);
  if (!ofs.is_open()) {
    std::cerr << "ERROR: Could not open output file for writing: " << filepath << std::endl;
    return;
  }

  ofs << std::setprecision(8) << std::fixed;
  for (int i = 0; i < n; ++i) {
    ofs << (double)buffer[i] << "\n";
  }
}

int main(int argc, char **argv) {
  std::string out_dir = "vitis_outputs";
  if (argc >= 2 && argv[1] != nullptr) out_dir = argv[1];

  std::cout << "RoFormer HLS all-layer dump\n";
  std::cout << "Output dir: " << out_dir << std::endl;

  // static data_t hw_input[IN_CHANNELS * SEQ_LEN] = {0};

  // // Match test_vitis_roformer.cpp relative paths.
  // const char *input_paths[] = {
  //     "verification_data/full_input.txt",
  //     "../verification_data/full_input.txt",
  //     "./verification_data/full_input.txt",
  //     nullptr};

  // bool input_loaded = false;
  // for (int p = 0; input_paths[p] != nullptr; ++p) {
  //   if (load_input_from_file(input_paths[p], hw_input, IN_CHANNELS * SEQ_LEN)) {
  //     input_loaded = true;
  //     break;
  //   }
  // }
  // if (!input_loaded) {
  //   std::cerr << "FATAL: Could not load input data." << std::endl;
  //   return 1;
  // }

  // Internal activation buffer layout matches RoFormer_Top.cpp:
  // We use separate ping-pong buffers per stage to avoid in-place aliasing hazards
  // in plain g++ simulation (Vitis banking/partitioning can hide these hazards).
  static data_t act0[D_MODEL * SEQ_LEN];
  static data_t act1[D_MODEL * SEQ_LEN];

  // READ_INPUT (same as RoFormer_Top.cpp)
  for (int i = 0; i < D_MODEL * SEQ_LEN; ++i) {
    act0[i] = (i < IN_CHANNELS * SEQ_LEN) ? input[i] : (data_t)0;
  }

  // 1) Conv1D dump
  conv1d_forward(act0, act1);
  write_vector_1_per_line(out_dir + "/conv1d_out.txt", act1, SEQ_LEN * OUT_CHANNELS);

  // 2) RoPE dump
  rope_forward(act1, act0);
  write_vector_1_per_line(out_dir + "/l0_rope_out.txt", act0, SEQ_LEN * D_MODEL);

  // 3) MHA + AddNorm1 dump
  mha_norm_forward(act0, act1);
  write_vector_1_per_line(out_dir + "/l0_add_norm1_out.txt", act1, SEQ_LEN * D_MODEL);

  // 4) FFN + AddNorm2 dump
  ffn_norm_forward(act1, act0);
  write_vector_1_per_line(out_dir + "/l0_add_norm2_out.txt", act0, SEQ_LEN * D_MODEL);

  // 5) Pool dump
  static data_t pooled[D_MODEL];
  pooling(act0, pooled);
  write_vector_1_per_line(out_dir + "/pool_out.txt", pooled, D_MODEL);

  // 6) Full dump (final FC)
  static data_t local_output[OUT_DIM];
  linear(pooled, local_output, fc_w_flat, fc_b_flat, D_MODEL, OUT_DIM);
  write_vector_1_per_line(out_dir + "/full_output.txt", local_output, OUT_DIM);

  std::cout << "Dump complete." << std::endl;
  return 0;
}

