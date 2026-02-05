# Compiler
CXX = g++
CXXFLAGS = -O3 -std=c++14

# Executable Names
# Executable Names
# Executable Names
TESTS = verify_conv1d verify_rope verify_feedforward verify_block verify_vitis \
        test_conv1d test_rope test_feedforward test_block test_roformer

# Targets
all: $(TESTS)

# --- Verification Suite (PyTorch Data) ---
verify_conv1d: verify_conv1d.cpp Conv1dLayer.h TestUtils.h
	$(CXX) $(CXXFLAGS) verify_conv1d.cpp -o verify_conv1d

verify_rope: verify_rope.cpp RotaryEmbedding.h TestUtils.h
	$(CXX) $(CXXFLAGS) verify_rope.cpp -o verify_rope

verify_feedforward: verify_feedforward.cpp FeedForward.h TestUtils.h
	$(CXX) $(CXXFLAGS) verify_feedforward.cpp -o verify_feedforward

verify_block: verify_block.cpp RoFormerBlock.h TestUtils.h
	$(CXX) $(CXXFLAGS) verify_block.cpp -o verify_block

verify_vitis: test_vitis_roformer.cpp RoFormer_Top.cpp RoFormer.h TestUtils.h
	$(CXX) $(CXXFLAGS) test_vitis_roformer.cpp RoFormer_Top.cpp -o verify_vitis

# --- Legacy Unit Tests (Self-Contained) ---
test_conv1d: test_conv1d.cpp Conv1dLayer.h
	$(CXX) $(CXXFLAGS) test_conv1d.cpp -o test_conv1d

test_rope: test_rope.cpp RotaryEmbedding.h
	$(CXX) $(CXXFLAGS) test_rope.cpp -o test_rope

test_feedforward: test_feedforward.cpp FeedForward.h
	$(CXX) $(CXXFLAGS) test_feedforward.cpp -o test_feedforward

test_block: test_block.cpp RoFormerBlock.h
	$(CXX) $(CXXFLAGS) test_block.cpp -o test_block

test_roformer: test_roformer.cpp RoFormer.h
	$(CXX) $(CXXFLAGS) test_roformer.cpp -o test_roformer

# Helper to run all verifications
run: all
	@echo "--- Running Conv1d Verification ---"
	./verify_conv1d
	@echo "\n--- Running RoPE Verification ---"
	./verify_rope
	@echo "\n--- Running FeedForward Verification ---"
	./verify_feedforward
	@echo "\n--- Running RoFormerBlock Verification ---"
	./verify_block
	@echo "\n--- Running Vitis RoFormer Top Verification ---"
	./verify_vitis

# Clean up binaries
# Explicitly list binaries to avoid wildcard deleting .cpp files
clean:
	rm -f verify_conv1d verify_rope verify_feedforward verify_block verify_vitis \
	      test_conv1d test_rope test_feedforward test_block test_roformer \
	      *.o *.exe
