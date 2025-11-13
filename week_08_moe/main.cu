// main.cu
// Week 8 – Unfused DeepSeek-style Mixture-of-Experts (MoE) operator in CUDA.
//
// This file implements a simple, single-GPU MoE forward pass that simulates
// expert-parallel routing logic:
//
//   1) Gate / Router:
//        - compute gate logits
//        - softmax over experts
//        - Top-1 expert selection per token (Switch-style)
//   2) Dispatch tokens into per-expert mini-batches
//   3) Per-expert MLP (two-layer FFN with GELU activation)
//   4) Combine expert outputs back to the original token order
//
// The implementation is intentionally "unfused": each logical step is a
// separate CUDA kernel call. This matches the assignment requirement of an
// unfused, distributed, data-parallel, expert-parallel MoE operator.
//
// The code runs on a single GPU but captures the same routing behavior that
// a multi-GPU expert-parallel implementation would have.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <random>

// -------------- Utility macro for CUDA error checking -------------- //

#define CHECK_CUDA(cmd)                                                     \
    do {                                                                    \
        cudaError_t e = (cmd);                                              \
        if (e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(e));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)


// -------------- Device-side helpers -------------- //

// GELU activation function (approximation used by many transformers).
__device__ __forceinline__ float gelu(float x) {
    // Approximation from Hendrycks & Gimpel:
    // gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x^3 ) ) )
    const float k0 = 0.7978845608f;   // sqrt(2 / pi)
    const float k1 = 0.044715f;
    float x3 = x * x * x;
    float inner = k0 * (x + k1 * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}


// -------------- Kernel 1: compute gate logits -------------- //
// x:            [batch_size, hidden_dim]
// W_gate:       [hidden_dim, num_experts]
// b_gate:       [num_experts]
// gate_logits:  [batch_size, num_experts]
__global__ void compute_gate_logits_kernel(
    const float* __restrict__ x,
    const float* __restrict__ W_gate,
    const float* __restrict__ b_gate,
    float* __restrict__ gate_logits,
    int batch_size,
    int hidden_dim,
    int num_experts
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= batch_size) return;

    // For each token, compute one logit per expert.
    for (int e = 0; e < num_experts; ++e) {
        float acc = 0.0f;
        // Dot product between token embedding and gate weight column.
        // W_gate is stored as [hidden_dim, num_experts] in row-major:
        // index(h, e) = h * num_experts + e.
        for (int h = 0; h < hidden_dim; ++h) {
            float x_val = x[token_idx * hidden_dim + h];
            float w_val = W_gate[h * num_experts + e];
            acc += x_val * w_val;
        }
        acc += b_gate[e];
        gate_logits[token_idx * num_experts + e] = acc;
    }
}


// -------------- Kernel 2: softmax + top-1 routing -------------- //
// gate_logits: [batch_size, num_experts]
// expert_indices: [batch_size]  (top-1 expert index per token)
// expert_probs:   [batch_size]  (corresponding probability)
__global__ void softmax_top1_kernel(
    const float* __restrict__ gate_logits,
    int* __restrict__ expert_indices,
    float* __restrict__ expert_probs,
    int batch_size,
    int num_experts
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= batch_size) return;

    // 1) Find max logit for numerical stability.
    float max_logit = -1e30f;
    for (int e = 0; e < num_experts; ++e) {
        float val = gate_logits[token_idx * num_experts + e];
        if (val > max_logit) {
            max_logit = val;
        }
    }

    // 2) Compute exp(logit - max_logit) and sum.
    float sum_exp = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
        float val = gate_logits[token_idx * num_experts + e];
        float e_val = expf(val - max_logit);
        sum_exp += e_val;
    }

    // 3) Compute probabilities, select Top-1 expert.
    int best_idx = 0;
    float best_prob = 0.0f;
    for (int e = 0; e < num_experts; ++e) {
        float val = gate_logits[token_idx * num_experts + e];
        float prob = expf(val - max_logit) / sum_exp;
        if (prob > best_prob) {
            best_prob = prob;
            best_idx = e;
        }
    }

    expert_indices[token_idx] = best_idx;
    expert_probs[token_idx] = best_prob;
}


// -------------- Kernel 3: count tokens per expert -------------- //
// expert_indices: [batch_size]
// expert_counts:  [num_experts] (must be zero-initialized before launch)
__global__ void count_expert_assignments_kernel(
    const int* __restrict__ expert_indices,
    int* __restrict__ expert_counts,
    int batch_size
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= batch_size) return;

    int e = expert_indices[token_idx];
    atomicAdd(&expert_counts[e], 1);
}


// -------------- Kernel 4: assign position within each expert -------------- //
// We reuse expert_counts as "cursors" to give each token a position in its
// expert's mini-batch.
// expert_indices:       [batch_size]
// expert_cursors:       [num_experts] (must be zero-initialized before launch)
// position_in_expert:   [batch_size]
__global__ void assign_positions_kernel(
    const int* __restrict__ expert_indices,
    int* __restrict__ expert_cursors,
    int* __restrict__ position_in_expert,
    int batch_size
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= batch_size) return;

    int e = expert_indices[token_idx];
    // atomicAdd returns the old value, which is the position for this token.
    int pos = atomicAdd(&expert_cursors[e], 1);
    position_in_expert[token_idx] = pos;
}


// -------------- Kernel 5: dispatch tokens to expert input buffer -------------- //
// x:                  [batch_size, hidden_dim]
// expert_indices:     [batch_size]
// position_in_expert: [batch_size]
// expert_offsets:     [num_experts]  (prefix sum of expert_counts)
// expert_input:       [total_expert_tokens, hidden_dim]
__global__ void dispatch_tokens_kernel(
    const float* __restrict__ x,
    const int* __restrict__ expert_indices,
    const int* __restrict__ position_in_expert,
    const int* __restrict__ expert_offsets,
    float* __restrict__ expert_input,
    int batch_size,
    int hidden_dim
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= batch_size) return;

    int e = expert_indices[token_idx];
    int pos = position_in_expert[token_idx];
    int offset = expert_offsets[e];
    int dst_row = offset + pos;

    // Copy entire token embedding row into expert_input[dst_row, :]
    for (int h = 0; h < hidden_dim; ++h) {
        expert_input[dst_row * hidden_dim + h] =
            x[token_idx * hidden_dim + h];
    }
}


// -------------- Kernel 6: expert MLP forward -------------- //
// expert_input:   [total_expert_tokens, hidden_dim]
// expert_output:  [total_expert_tokens, hidden_dim]
// W1:             [num_experts, hidden_dim, ffn_dim]
// b1:             [num_experts, ffn_dim]
// W2:             [num_experts, ffn_dim, hidden_dim]
// b2:             [num_experts, hidden_dim]
// expert_offsets: [num_experts] (prefix sums)
// total_expert_tokens = sum_e expert_counts[e]
//
// Each thread handles one "expert token row" (one row in expert_input).
__global__ void expert_mlp_forward_kernel(
    const float* __restrict__ expert_input,
    float* __restrict__ expert_output,
    const float* __restrict__ W1,
    const float* __restrict__ b1,
    const float* __restrict__ W2,
    const float* __restrict__ b2,
    const int* __restrict__ expert_offsets,
    int total_expert_tokens,
    int num_experts,
    int hidden_dim,
    int ffn_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_expert_tokens) return;

    // Determine which expert this row belongs to using expert_offsets.
    int expert_id = 0;
    // expert_offsets is a prefix sum: offsets[e] = sum_{k < e} count[k].
    // We find the largest e such that row >= offsets[e].
    for (int e = 1; e < num_experts; ++e) {
        if (row >= expert_offsets[e]) {
            expert_id = e;
        }
    }

    // Shortcut pointers to this expert's parameters.
    // Layout:
    //   W1[e, h, f] = W1[e * hidden_dim * ffn_dim + h * ffn_dim + f]
    //   W2[e, f, h] = W2[e * ffn_dim * hidden_dim + f * hidden_dim + h]
    const float* W1_e = W1 + expert_id * hidden_dim * ffn_dim;
    const float* b1_e = b1 + expert_id * ffn_dim;
    const float* W2_e = W2 + expert_id * ffn_dim * hidden_dim;
    const float* b2_e = b2 + expert_id * hidden_dim;

    const float* x_row = expert_input + row * hidden_dim;
    float* y_row = expert_output + row * hidden_dim;

    // We implement y = W2 * gelu(W1 * x + b1) + b2
    // To avoid local arrays, we compute hidden activations inside nested loops.
    for (int h_out = 0; h_out < hidden_dim; ++h_out) {
        float acc_out = 0.0f;
        for (int f = 0; f < ffn_dim; ++f) {
            // Compute hidden activation for dimension f.
            float acc_hidden = 0.0f;
            for (int h_in = 0; h_in < hidden_dim; ++h_in) {
                float x_val = x_row[h_in];
                float w1_val = W1_e[h_in * ffn_dim + f];
                acc_hidden += x_val * w1_val;
            }
            acc_hidden += b1_e[f];
            float hidden_val = gelu(acc_hidden);

            float w2_val = W2_e[f * hidden_dim + h_out];
            acc_out += hidden_val * w2_val;
        }
        acc_out += b2_e[h_out];
        y_row[h_out] = acc_out;
    }
}


// -------------- Kernel 7: combine expert outputs back to original tokens -------------- //
// expert_output:      [total_expert_tokens, hidden_dim]
// expert_indices:     [batch_size]
// position_in_expert: [batch_size]
// expert_offsets:     [num_experts]
// expert_probs:       [batch_size]  (top-1 probability per token)
// y:                  [batch_size, hidden_dim]  (final MoE output)
__global__ void combine_outputs_kernel(
    const float* __restrict__ expert_output,
    const int* __restrict__ expert_indices,
    const int* __restrict__ position_in_expert,
    const int* __restrict__ expert_offsets,
    const float* __restrict__ expert_probs,
    float* __restrict__ y,
    int batch_size,
    int hidden_dim
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= batch_size) return;

    int e = expert_indices[token_idx];
    int pos = position_in_expert[token_idx];
    int offset = expert_offsets[e];
    int src_row = offset + pos;

    float gate_prob = expert_probs[token_idx];

    for (int h = 0; h < hidden_dim; ++h) {
        float val = expert_output[src_row * hidden_dim + h];
        // We weight the output by the router probability (Top-1 gate).
        y[token_idx * hidden_dim + h] = gate_prob * val;
    }
}


// -------------- Host-side helper: random initialization -------------- //

void init_random(std::vector<float>& v, float scale = 0.02f) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, scale);
    for (auto& x : v) {
        x = dist(rng);
    }
}


// -------------- Main: test the MoE operator -------------- //

int main() {
    // ---------------- Hyperparameters ---------------- //
    // You can adjust these numbers; for debugging, small sizes are easier.
    const int batch_size  = 8;   // number of tokens
    const int hidden_dim  = 16;  // token embedding dimension
    const int num_experts = 4;   // number of experts in the MoE layer
    const int ffn_dim     = 32;  // inner dimension of MLP

    std::cout << "Running simple DeepSeek-style MoE on CUDA..." << std::endl;

    // ---------------- Host memory allocation ---------------- //
    // Input tokens
    std::vector<float> h_x(batch_size * hidden_dim);

    // Gate parameters
    std::vector<float> h_W_gate(hidden_dim * num_experts);
    std::vector<float> h_b_gate(num_experts);

    // Expert MLP parameters
    std::vector<float> h_W1(num_experts * hidden_dim * ffn_dim);
    std::vector<float> h_b1(num_experts * ffn_dim);
    std::vector<float> h_W2(num_experts * ffn_dim * hidden_dim);
    std::vector<float> h_b2(num_experts * hidden_dim);

    // Initialize everything with small random values.
    init_random(h_x);
    init_random(h_W_gate);
    init_random(h_b_gate);
    init_random(h_W1);
    init_random(h_b1);
    init_random(h_W2);
    init_random(h_b2);

    // ---------------- Device memory allocation ---------------- //

    float* d_x = nullptr;
    float* d_W_gate = nullptr;
    float* d_b_gate = nullptr;
    float* d_gate_logits = nullptr;

    int*   d_expert_indices = nullptr;
    float* d_expert_probs   = nullptr;

    int*   d_expert_counts  = nullptr;
    int*   d_expert_offsets = nullptr;
    int*   d_position_in_expert = nullptr;

    float* d_expert_input   = nullptr;
    float* d_expert_output  = nullptr;

    float* d_W1 = nullptr;
    float* d_b1 = nullptr;
    float* d_W2 = nullptr;
    float* d_b2 = nullptr;

    float* d_y  = nullptr;

    // Allocate device memory for input and parameters
    CHECK_CUDA(cudaMalloc(&d_x, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W_gate, hidden_dim * num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b_gate, num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gate_logits, batch_size * num_experts * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_expert_indices, batch_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_probs, batch_size * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_expert_counts, num_experts * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_offsets, num_experts * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_position_in_expert, batch_size * sizeof(int)));

    CHECK_CUDA(cudaMalloc(&d_W1, num_experts * hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b1, num_experts * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_W2, num_experts * ffn_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b2, num_experts * hidden_dim * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_y, batch_size * hidden_dim * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(),
                          batch_size * hidden_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W_gate, h_W_gate.data(),
                          hidden_dim * num_experts * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b_gate, h_b_gate.data(),
                          num_experts * sizeof(float),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_W1, h_W1.data(),
                          num_experts * hidden_dim * ffn_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b1, h_b1.data(),
                          num_experts * ffn_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_W2, h_W2.data(),
                          num_experts * ffn_dim * hidden_dim * sizeof(float),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b2, h_b2.data(),
                          num_experts * hidden_dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---------------- 1) Gate: compute logits ---------------- //
    {
        int threads = 128;
        int blocks = (batch_size + threads - 1) / threads;
        compute_gate_logits_kernel<<<blocks, threads>>>(
            d_x, d_W_gate, d_b_gate, d_gate_logits,
            batch_size, hidden_dim, num_experts
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ---------------- 2) Gate: softmax + top-1 routing ---------------- //
    {
        int threads = 128;
        int blocks = (batch_size + threads - 1) / threads;
        softmax_top1_kernel<<<blocks, threads>>>(
            d_gate_logits,
            d_expert_indices,
            d_expert_probs,
            batch_size,
            num_experts
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ---------------- 3) Count tokens per expert ---------------- //
    // We must zero-initialize expert_counts before counting.
    CHECK_CUDA(cudaMemset(d_expert_counts, 0, num_experts * sizeof(int)));
    {
        int threads = 128;
        int blocks = (batch_size + threads - 1) / threads;
        count_expert_assignments_kernel<<<blocks, threads>>>(
            d_expert_indices,
            d_expert_counts,
            batch_size
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Bring counts back to host to compute prefix sums (offsets).
    std::vector<int> h_expert_counts(num_experts);
    CHECK_CUDA(cudaMemcpy(h_expert_counts.data(), d_expert_counts,
                          num_experts * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<int> h_expert_offsets(num_experts);
    int total_expert_tokens = 0;
    for (int e = 0; e < num_experts; ++e) {
        h_expert_offsets[e] = total_expert_tokens;
        total_expert_tokens += h_expert_counts[e];
    }

    if (total_expert_tokens == 0) {
        std::cerr << "Warning: no tokens were routed to any expert. "
                  << "Check gate initialization." << std::endl;
    }

    // Copy offsets back to device
    CHECK_CUDA(cudaMemcpy(d_expert_offsets, h_expert_offsets.data(),
                          num_experts * sizeof(int),
                          cudaMemcpyHostToDevice));

    // ---------------- 4) Assign position within each expert ---------------- //
    // Reuse d_expert_counts as "cursors"; we must reset to zero first.
    CHECK_CUDA(cudaMemset(d_expert_counts, 0, num_experts * sizeof(int)));
    {
        int threads = 128;
        int blocks = (batch_size + threads - 1) / threads;
        assign_positions_kernel<<<blocks, threads>>>(
            d_expert_indices,
            d_expert_counts,       // used as cursors now
            d_position_in_expert,
            batch_size
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ---------------- 5) Allocate expert input/output buffers ---------------- //
    if (total_expert_tokens == 0) {
        // Degenerate case: all outputs are zeros.
        CHECK_CUDA(cudaMemset(d_y, 0, batch_size * hidden_dim * sizeof(float)));
    } else {
        CHECK_CUDA(cudaMalloc(&d_expert_input,
                              total_expert_tokens * hidden_dim * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_expert_output,
                              total_expert_tokens * hidden_dim * sizeof(float)));

        // ---------------- 6) Dispatch tokens to expert input buffer ---------------- //
        {
            int threads = 128;
            int blocks = (batch_size + threads - 1) / threads;
            dispatch_tokens_kernel<<<blocks, threads>>>(
                d_x,
                d_expert_indices,
                d_position_in_expert,
                d_expert_offsets,
                d_expert_input,
                batch_size,
                hidden_dim
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // ---------------- 7) Expert MLP forward ---------------- //
        {
            int threads = 128;
            int blocks = (total_expert_tokens + threads - 1) / threads;
            expert_mlp_forward_kernel<<<blocks, threads>>>(
                d_expert_input,
                d_expert_output,
                d_W1,
                d_b1,
                d_W2,
                d_b2,
                d_expert_offsets,
                total_expert_tokens,
                num_experts,
                hidden_dim,
                ffn_dim
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        // ---------------- 8) Combine expert outputs back to original token order ---------------- //
        {
            int threads = 128;
            int blocks = (batch_size + threads - 1) / threads;
            combine_outputs_kernel<<<blocks, threads>>>(
                d_expert_output,
                d_expert_indices,
                d_position_in_expert,
                d_expert_offsets,
                d_expert_probs,
                d_y,
                batch_size,
                hidden_dim
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    // ---------------- 9) Copy final output back to host and print a few values ---------------- //
    std::vector<float> h_y(batch_size * hidden_dim);
    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y,
                          batch_size * hidden_dim * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::cout << "MoE forward finished. Showing first token output:" << std::endl;
    for (int h = 0; h < hidden_dim; ++h) {
        std::cout << h_y[h] << " ";
    }
    std::cout << std::endl;

    // ---------------- Cleanup ---------------- //
    cudaFree(d_x);
    cudaFree(d_W_gate);
    cudaFree(d_b_gate);
    cudaFree(d_gate_logits);

    cudaFree(d_expert_indices);
    cudaFree(d_expert_probs);
    cudaFree(d_expert_counts);
    cudaFree(d_expert_offsets);
    cudaFree(d_position_in_expert);

    cudaFree(d_W1);
    cudaFree(d_b1);
    cudaFree(d_W2);
    cudaFree(d_b2);

    if (d_expert_input)  cudaFree(d_expert_input);
    if (d_expert_output) cudaFree(d_expert_output);

    cudaFree(d_y);

    std::cout << "Done." << std::endl;
    return 0;
}
