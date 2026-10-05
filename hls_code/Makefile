# Compiler
CXX = g++
CXXFLAGS = -O3 -std=c++14

SRC_DIR = .
TEST_DIR = tests

TEST_VITIS_ALL = test_vitis_roformer_all_layers
TEST_VITIS_SIMPLE = test_vitis

all: $(TEST_VITIS_ALL) $(TEST_VITIS_SIMPLE)

$(TEST_VITIS_ALL): $(TEST_DIR)/test_vitis_roformer_all_layers.cpp $(SRC_DIR)/types.h $(SRC_DIR)/config.h
	$(CXX) $(CXXFLAGS) -DUSE_MOCK_HLS -I. -I./tests -o $(TEST_VITIS_ALL) $(TEST_DIR)/test_vitis_roformer_all_layers.cpp

$(TEST_VITIS_SIMPLE): $(TEST_DIR)/test_vitis_roformer.cpp $(SRC_DIR)/RoFormer_Top.cpp $(SRC_DIR)/types.h $(SRC_DIR)/config.h
	$(CXX) $(CXXFLAGS) -DUSE_MOCK_HLS -I. -I./tests -o $(TEST_VITIS_SIMPLE) $(TEST_DIR)/test_vitis_roformer.cpp $(SRC_DIR)/RoFormer_Top.cpp

run: all
	@echo "--- Running tests ---"

clean-all-layers:
	rm -f $(TEST_VITIS_ALL)

clean-simple-test:
	rm -f $(TEST_VITIS_SIMPLE)

clean:
	rm -f $(TESTS) $(TEST_VITIS_ALL) $(TEST_VITIS_SIMPLE) *.o *.exe
