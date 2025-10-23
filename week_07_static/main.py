
import modal
import os
from textwrap import dedent

app = modal.App("week07-flashattn-static")

# Single CUDA devel image so nvcc is available where we run
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("openmpi-bin", "libopenmpi-dev", "git", "make")
)

PROJECT_DIR = "/root/week_07/project"

README_MD = dedent("""
# Week 07 — FlashAttention (Minimal CUDA Demo)

This is a minimal CUDA program launched via MPI (`mpirun -np 4`) to emulate
a FlashAttention-style workload (matrix ops + softmax). Each MPI process
performs the same GPU computation independently; using `mpirun` just spawns
multiple workers.

## Run locally with Modal
```bash
modal run main.py::auto_run
```

That will:
1) Create the project files *inside* the Modal container
2) Build with `nvcc`
3) Run with `mpirun -np 4` on an A100
""")

MAKEFILE = dedent("""
BIN = bin
SRC = src/flash_attn.cu
OUT = $(BIN)/flash_attn

NVCC = nvcc
NVFLAGS = -O3 -arch=sm_80

.PHONY: all clean

all: $(OUT)

$(OUT): $(SRC) | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $<

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
""")

FLASH_ATTN_CU = dedent(r"""
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \ 
} while (0)

__global__ void matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += A[row*K + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}

__global__ void row_softmax(float* S, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        float maxv = -1e30f;
        for (int j = 0; j < N; ++j) maxv = fmaxf(maxv, S[row*N + j]);
        float sum = 0.0f;
        for (int j = 0; j < N; ++j) {
            float v = expf(S[row*N + j] - maxv);
            S[row*N + j] = v;
            sum += v;
        }
        for (int j = 0; j < N; ++j) S[row*N + j] /= sum;
    }
}

__global__ void transposeKD(const float* K, float* KT, int N, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N*D) {
        int n = idx / D;
        int d = idx % D;
        KT[d*N + n] = K[n*D + d];
    }
}

__global__ void scale_kernel(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

int main() {
    int M = 256, N = 256, D = 64;
    float scale = 1.0f / sqrtf((float)D);

    size_t q_bytes = M * D * sizeof(float);
    size_t k_bytes = N * D * sizeof(float);
    size_t v_bytes = N * D * sizeof(float);
    size_t s_bytes = M * N * sizeof(float);
    size_t o_bytes = M * D * sizeof(float);

    std::vector<float> hQ(M*D), hK(N*D), hV(N*D);
    for (int i = 0; i < M*D; ++i) hQ[i] = (i % 13) * 0.01f;
    for (int i = 0; i < N*D; ++i) hK[i] = (i % 17) * 0.01f;
    for (int i = 0; i < N*D; ++i) hV[i] = (i % 19) * 0.01f;

    float *dQ, *dK, *dV, *dS, *dO, *dKT;
    CUDA_CHECK(cudaMalloc(&dQ, q_bytes));
    CUDA_CHECK(cudaMalloc(&dK, k_bytes));
    CUDA_CHECK(cudaMalloc(&dV, v_bytes));
    CUDA_CHECK(cudaMalloc(&dS, s_bytes));
    CUDA_CHECK(cudaMalloc(&dO, o_bytes));
    CUDA_CHECK(cudaMalloc(&dKT, k_bytes));

    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), q_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), k_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), v_bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N*D + threads - 1) / threads;
    transposeKD<<<blocks, threads>>>(dK, dKT, N, D);
    CUDA_CHECK(cudaGetLastError());

    dim3 block(16, 16);
    dim3 grid1((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);
    matmul<<<grid1, block>>>(dQ, dKT, dS, M, N, D);
    CUDA_CHECK(cudaGetLastError());

    int total = M*N;
    int blocks2 = (total + threads - 1) / threads;
    scale_kernel<<<blocks2, threads>>>(dS, scale, total);
    CUDA_CHECK(cudaGetLastError());

    int blocks3 = (M + threads - 1) / threads;
    row_softmax<<<blocks3, threads>>>(dS, M, N);
    CUDA_CHECK(cudaGetLastError());

    dim3 grid3((D + block.x - 1)/block.x, (M + block.y - 1)/block.y);
    matmul<<<grid3, block>>>(dS, dV, dO, M, D, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hO(M*D);
    CUDA_CHECK(cudaMemcpy(hO.data(), dO, o_bytes, cudaMemcpyDeviceToHost));
    double checksum = 0.0;
    for (int i = 0; i < M*D; i += 97) checksum += hO[i];
    printf("OK - checksum: %.6f\n", checksum);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dS); cudaFree(dO); cudaFree(dKT);
    return 0;
}
""")

def _write(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

@app.function(image=image, timeout=1200)
def build_binary():
    os.makedirs(PROJECT_DIR, exist_ok=True)
    os.chdir(PROJECT_DIR)
    _write(f"{PROJECT_DIR}/README.md", README_MD)
    _write(f"{PROJECT_DIR}/Makefile", MAKEFILE)
    _write(f"{PROJECT_DIR}/src/flash_attn.cu", FLASH_ATTN_CU)
    print("Sources written to", PROJECT_DIR)
    os.system("nvcc --version || true")
    print("Building...")
    rc = os.system("make -j")
    if rc != 0:
        raise RuntimeError("nvcc build failed")
    print("Build complete. Files:", os.listdir("."))

@app.function(image=image, gpu="A100", timeout=1200)
def build_and_run():
    os.makedirs(PROJECT_DIR, exist_ok=True)
    os.chdir(PROJECT_DIR)

    # 写入文件
    def _write(path, content):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)

    _write(f"{PROJECT_DIR}/README.md", README_MD)
    _write(f"{PROJECT_DIR}/Makefile", MAKEFILE)
    _write(f"{PROJECT_DIR}/src/flash_attn.cu", FLASH_ATTN_CU)

    print("✅ Sources written to", PROJECT_DIR)
    os.system("nvcc --version || true")
    print("🔧 Building...")
    os.system("make -j")

    print("🚀 Running FlashAttention...")
    os.system("nvidia-smi || true")
    cmd = (
        "OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 "
        "mpirun --allow-run-as-root -np 4 ./bin/flash_attn"
    )
    os.system(cmd)

@app.local_entrypoint()
def auto_run():
    build_and_run.remote()