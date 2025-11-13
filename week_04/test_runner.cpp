// test_runner.cpp
// CPU test runner for Week 04 FlashAttention
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Prototypes from naive_attention.c and flash_attn_ref.c
void naive_attention_forward(
    const float* Q, const float* K, const float* V,
    float* O, int N, int d);

void flash_attention_forward_ref(
    const float* Q, const float* K, const float* V,
    float* O, int N, int d, int Br, int Bc);

static void rand_fill(float* x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = (float)rand() / RAND_MAX * 2.f - 1.f;
    }
}

static float max_abs_diff(const float* a, const float* b, int n) {
    float m = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

int main() {
    srand(0);

    const int N  = 128;   // sequence length
    const int d  = 64;    // head dimension
    const int Br = 64;    // row tile
    const int Bc = 64;    // col tile

    float* Q  = (float*)malloc(N * d * sizeof(float));
    float* K  = (float*)malloc(N * d * sizeof(float));
    float* V  = (float*)malloc(N * d * sizeof(float));
    float* O1 = (float*)malloc(N * d * sizeof(float));
    float* O2 = (float*)malloc(N * d * sizeof(float));

    if (!Q || !K || !V || !O1 || !O2) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    rand_fill(Q, N * d);
    rand_fill(K, N * d);
    rand_fill(V, N * d);

    printf("Running naive CPU attention...\n");
    naive_attention_forward(Q, K, V, O1, N, d);

    printf("Running FlashAttention CPU reference...\n");
    flash_attention_forward_ref(Q, K, V, O2, N, d, Br, Bc);

    float err = max_abs_diff(O1, O2, N * d);
    printf("CPU naive vs CPU flash max-abs error: %.6f\n", err);

    if (err < 1e-4f) {
        printf("OK ✅\n");
    } else {
        printf("Mismatch ❌\n");
    }

    free(Q); free(K); free(V); free(O1); free(O2);
    return 0;
}// (same as before, CPU test runner)
