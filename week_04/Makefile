# Makefile for Week 04 FlashAttention
CXX  = g++
NVCC ?= nvcc

CXXFLAGS = -O2 -std=c++17
NVFLAGS  = -O3 -std=c++17 -arch=sm_80

BUILD = build
CPU_OBJS = $(BUILD)/naive_attention.o $(BUILD)/flash_attn_ref.o $(BUILD)/test_runner.o

all: cpu

$(BUILD):
	mkdir -p $(BUILD)

# -------- CPU build & test --------
cpu: $(BUILD) $(CPU_OBJS)
	$(CXX) $(CXXFLAGS) -o $(BUILD)/test_cpu $(CPU_OBJS) -lm

$(BUILD)/naive_attention.o: naive_attention.c | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/flash_attn_ref.o: flash_attn_ref.c | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/test_runner.o: test_runner.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# -------- CUDA build (compile only) --------
# We only verify that the CUDA kernel builds with nvcc in the GPU environment.
cuda: $(BUILD)
	$(NVCC) $(NVFLAGS) -c flash_attn_cuda.cu -o $(BUILD)/flash_attn_cuda.o

clean:
	rm -rf $(BUILD)
