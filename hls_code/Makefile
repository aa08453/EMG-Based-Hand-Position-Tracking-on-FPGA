# Bit-accurate C simulation of the RoFormer (any number of encoder layers) without Vitis.
# third_party/ap_types holds Xilinx's open-source ap_fixed headers (Apache-2.0); sim/ shims hls_math.h.
# The installed model (model_config.h, weights/) comes from scripts/install_export.py.
# Weight arrays are float in C simulation (wrom_t, types.h), so a build takes seconds; -O0 keeps it that way.
#   make              build both tests
#   make run          run them (layer-by-layer + 40 HIL windows, then the Vitis testbench)
#   make MAC_UNROLL=8 build with a different parallelism setting (results must be identical)
CXX      ?= g++
MAC_UNROLL ?= 4
CXXFLAGS = -std=c++14 -O0 -w -Isim -Ithird_party/ap_types/include -I. -DMAC_UNROLL=$(MAC_UNROLL)
SRCS     = $(wildcard *.h) $(wildcard weights/*.h) RoFormer_Top.cpp tests/check.h

all: test_layers test_top

test_layers: tests/test_layers.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ tests/test_layers.cpp

test_top: tests/test_vitis_roformer.cpp $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ tests/test_vitis_roformer.cpp RoFormer_Top.cpp

run: all
	./test_layers ../hil_verification_samples
	./test_top

clean:
	rm -f test_layers test_top

.PHONY: all run clean
