#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void matmul_single(int m, int n, int p, double **A, double **B, double **C);
extern void matmul_pthreads(int m, int n, int p, double **A, double **B, double **C, int num_threads);

double** alloc_matrix(int m, int n) {
    double **M = malloc(m * sizeof(double*));
    for (int i = 0; i < m; i++) {
        M[i] = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            M[i][j] = rand() % 10;
        }
    }
    return M;
}

double elapsed(clock_t start, clock_t end) {
    return (double)(end-start)/CLOCKS_PER_SEC;
}

int main() {
    int m=1000,n=1000,p=1000;
    double **A=alloc_matrix(m,n), **B=alloc_matrix(n,p), **C=alloc_matrix(m,p);

    clock_t t1=clock();
    matmul_single(m,n,p,A,B,C);
    clock_t t2=clock();
    printf("Single-thread time: %.2fs\n", elapsed(t1,t2));

    int threads[]={1,4,16,32,64,128};
    for(int i=0;i<6;i++){
        t1=clock();
        matmul_pthreads(m,n,p,A,B,C,threads[i]);
        t2=clock();
        printf("%d threads: %.2fs\n", threads[i], elapsed(t1,t2));
    }
    return 0;
}
