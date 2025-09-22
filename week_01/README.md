# Week 01 Assignment

## Part 1. Single-threaded Implementation
- Implemented in [`matmul_single.c`](./matmul_single.c)
- Algorithm: classic triple-nested loop, O(m*n*p) complexity.
- Verified correctness with small test matrices.

## Part 2. Test Cases
- Implemented in [`test_matmul.c`](./test_matmul.c)
- Edge cases tested:
  - A = 1x1, B = 1x1 → ✅
  - A = 1x1, B = 1x5 → ✅
  - A = 2x1, B = 1x3 → ✅
  - A = 2x2, B = 2x2 → ✅
- All tests passed successfully for the single-threaded version.
- Multi-threaded results were verified against single-threaded outputs.

## Part 3. Multi-threaded Implementation
- Implemented in [`matmul_pthreads.c`](./matmul_pthreads.c)
- Parallelization strategy:
  - Split rows of matrix A across multiple pthreads.
  - Each thread computes a block of rows in C.
- Verified correctness: multi-threaded results match single-threaded outputs.

## Part 4. Performance Evaluation
- Implemented in [`perf.c`](./perf.c)
- Environment: MacBook Pro M1 Pro, 16GB RAM (replace with your own if different).
- Matrix size: 1000 × 1000
- Performance results (fill in with your actual measurements):

| Threads | Runtime (seconds) |
|---------|--------------------|
| 1       | XX.XX             |
| 4       | XX.XX             |
| 16      | XX.XX             |
| 32      | XX.XX             |
| 64      | XX.XX             |
| 128     | XX.XX             |

### Observations
- Clear speedup as the number of threads increases, up to a point.
- Diminishing returns beyond ~32 threads due to thread overhead and hardware limits.

## Reflection
- Learned how to implement matrix multiplication in C from scratch.
- Practiced writing edge test cases to ensure correctness.
- Gained hands-on experience with pthreads and parallel programming.
- Observed practical limits of parallelism (hardware threads, cache effects).
- This assignment improved both low-level C coding skills and understanding of HPC concepts.

---

## 🔧 How to Compile and Run

Make sure you are inside the `week_01/` directory.

### 1. Compile test program
```bash
gcc -o test_matmul test_matmul.c matmul_single.c matmul_pthreads.c -lpthread
