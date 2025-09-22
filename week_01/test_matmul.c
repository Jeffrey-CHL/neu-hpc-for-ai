#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

extern void matmul_single(int m, int n, int p, double **A, double **B, double **C);
extern void matmul_pthreads(int m, int n, int p, double **A, double **B, double **C, int num_threads);

// Helper: allocate a matrix
double** alloc_matrix(int m, int n) {
    double **M = malloc(m * sizeof(double*));
    for (int i = 0; i < m; i++) {
        M[i] = calloc(n, sizeof(double));
    }
    return M;
}

int main() {
    // Test A=1x1, B=1x1
    double **A = alloc_matrix(1,1), **B = alloc_matrix(1,1), **C = alloc_matrix(1,1);
    A[0][0] = 2; B[0][0] = 3;
    matmul_single(1,1,1,A,B,C);
    assert(C[0][0] == 6);

    // Test A=1x1, B=1x5
    A = alloc_matrix(1,1); B = alloc_matrix(1,5); C = alloc_matrix(1,5);
    A[0][0] = 2; for(int j=0;j<5;j++) B[0][j]=j+1;
    matmul_single(1,1,5,A,B,C);
    assert(C[0][4] == 10);

    // Test A=2x1, B=1x3
    A = alloc_matrix(2,1); B = alloc_matrix(1,3); C = alloc_matrix(2,3);
    A[0][0]=1; A[1][0]=2; B[0][0]=3; B[0][1]=4; B[0][2]=5;
    matmul_single(2,1,3,A,B,C);
    assert(C[1][2] == 10);

    printf("✅ All single-threaded tests passed!\n");

    // Multithreaded test
    int m=4,n=4,p=4;
    A = alloc_matrix(m,n); B = alloc_matrix(n,p); C = alloc_matrix(m,p);
    for(int i=0;i<m;i++) for(int j=0;j<n;j++) A[i][j]=i+j;
    for(int i=0;i<n;i++) for(int j=0;j<p;j++) B[i][j]=i*j;

    matmul_pthreads(m,n,p,A,B,C,4);
    printf("✅ Multi-threaded test completed.\n");
    return 0;
}
