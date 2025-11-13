# Simple Makefile for Week 4 FlashAttention
# Usage:
#   make        # builds the flash_attention binary
#   make run    # builds and runs it
#   make clean  # removes the binary

NVCC ?= nvcc
NVCCFLAGS = -O3 -std=c++17

TARGET = flash_attention
SRC = flash_attention.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
