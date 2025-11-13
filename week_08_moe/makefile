# Makefile for Week 8 DeepSeek MoE CUDA assignment
# Usage:
#   make        # build the binary
#   make run    # build and run
#   make clean  # remove the binary

NVCC ?= nvcc
TARGET := deepseek_moe
SRC := main.cu

# You can tweak SM architecture if needed (e.g., sm_80 for A100)
NVCC_FLAGS := -O3 -std=c++17

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
