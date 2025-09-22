#include <stdio.h>
#include <stdlib.h>

// Single-Threaded Matrix Multiplication: C = A * B
// A: m x n, B: n x p, C: m x p
void matmul_single(int m, int n, int p, double **A, double **B, double **C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}
