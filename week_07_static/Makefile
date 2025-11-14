# ==== 配置区 ====

CUDA_HOME ?= /usr/local/cuda
NVCC      := nvcc
# 注意：sm_80 是 A100 的算力架构，和 Modal 给你的 GPU 匹配
NVCCFLAGS := -O3 -arch=sm_80

# Modal 容器里 OpenMPI 的头文件和库路径
MPI_INC   := /usr/include/x86_64-linux-gnu/mpi
MPI_LIB   := /usr/lib/x86_64-linux-gnu/openmpi/lib

INCLUDES  := -I$(MPI_INC)
LIBS      := -L$(MPI_LIB) -lmpi

SRC_DIR   := src
BIN_DIR   := bin
TARGET    := $(BIN_DIR)/flash_attn
SRCS      := $(SRC_DIR)/flash_attn.cu

# ==== 规则区 ====

all: $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(BIN_DIR) $(SRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SRCS) $(LIBS)

clean:
	rm -rf $(BIN_DIR)