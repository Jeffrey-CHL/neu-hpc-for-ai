# Makefile for Week 02 GEMM
NVCC ?= nvcc
ARCH ?= -arch=sm_80
OPT  ?= -O3
TILE ?= 16
DEFS := -DTILE=$(TILE)

all: gemm

gemm: gemm.cu
	$(NVCC) $(OPT) $(ARCH) $(DEFS) -o gemm gemm.cu

clean:
	rm -f gemm