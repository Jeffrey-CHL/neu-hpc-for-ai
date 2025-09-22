# Makefile for week_02/mat_mul
NVCC ?= nvcc
ARCH ?= -arch=sm_80   # adjust to your GPU (sm_70 V100, sm_80 A100, sm_89 L4, sm_90 H100/B200)
OPT  ?= -O3

# default TILE=16; you can override: make TILE=32
TILE ?= 16
DEFS := -DTILE=$(TILE)

all: gemm

gemm: gemm.cu
	$(NVCC) $(OPT) $(ARCH) $(DEFS) -o gemm gemm.cu

clean:
	rm -f gemm
