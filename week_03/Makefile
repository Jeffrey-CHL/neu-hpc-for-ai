NVCC ?= nvcc
ARCH ?= -arch=sm_80
CFLAGS ?= -O3 $(ARCH)

all: gemm

gemm: gemm.cu
	$(NVCC) $(CFLAGS) $< -o $@

clean:
	rm -f gemm *.o
