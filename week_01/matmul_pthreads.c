#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    int tid;
    int m, n, p;
    double **A, **B, **C;
    int row_start, row_end; // Range of rows assigned to this thread
} ThreadData;

void *worker(void *arg) {
    ThreadData *td = (ThreadData*)arg;
    for (int i = td->row_start; i < td->row_end; i++) {
        for (int j = 0; j < td->p; j++) {
            td->C[i][j] = 0.0;
            for (int k = 0; k < td->n; k++) {
                td->C[i][j] += td->A[i][k] * td->B[k][j];
            }
        }
    }
    return NULL;
}

void matmul_pthreads(int m, int n, int p, double **A, double **B, double **C, int num_threads) {
    pthread_t threads[num_threads];
    ThreadData td[num_threads];
    int rows_per_thread = m / num_threads;
    int extra = m % num_threads;

    int row = 0;
    for (int t = 0; t < num_threads; t++) {
        int start = row;
        int end = start + rows_per_thread + (t < extra ? 1 : 0);
        td[t] = (ThreadData){t, m, n, p, A, B, C, start, end};
        pthread_create(&threads[t], NULL, worker, &td[t]);
        row = end;
    }
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}
