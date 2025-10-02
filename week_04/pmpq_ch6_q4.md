# PMPP Chapter 6 — Question 4 (Answer)

## Restatement of the Question
Evaluate the scalability and performance limitations of a parallel program with both serial and parallel components, under the frameworks of **Amdahl’s Law**, **Gustafson’s Law**, and the **Roofline model**. Identify when execution is bandwidth-bound versus compute-bound, and propose optimization strategies.

## Analysis

### 1. Amdahl’s vs. Gustafson’s Law
- **Amdahl’s Law**: With a serial fraction `s`, the maximum theoretical speedup on `p` processors is  
  \[ S_{Amdahl} = \frac{1}{s + \frac{1-s}{p}}. \]  
  This highlights diminishing returns when `s` > 0, even with large `p`.
- **Gustafson’s Law**: When problem size scales with processor count, the effective speedup is  
  \[ S_{Gustafson} = s + (1-s) \cdot p. \]  
  This suggests that for large workloads, nearly linear scaling can be achieved.

### 2. Roofline Model
- Performance is bounded by either peak compute or memory bandwidth:  
  \[ P = \min(P_{peak}, I \cdot B_w), \]  
  where `I` is arithmetic intensity (FLOPs per byte).  
- If `I < P_peak/Bw`, the kernel is **memory-bound**.  
- If `I > P_peak/Bw`, the kernel is **compute-bound**.

### 3. Latency Hiding and Optimization
- **Occupancy**: Ensure enough threadblocks to hide latency.  
- **Instruction-level parallelism (ILP)**: Unroll loops and overlap independent operations.  
- **Prefetching / double-buffering**: Overlap memory transfer with compute.  
- **Tiling**: Increase reuse of Q/K/V tiles to raise arithmetic intensity.

## Example (Symbolic)
- Assume `P_peak = 20 TFLOP/s`, `Bw = 1 TB/s`.  
- For arithmetic intensity `I = 10 FLOP/Byte`: attainable performance is  
  \[ P = \min(20, 10 \times 1) = 10 \; TFLOP/s. \]  
  → memory-bound.  
- For `I = 50 FLOP/Byte`:  
  \[ P = \min(20, 50 \times 1) = 20 \; TFLOP/s. \]  
  → compute-bound.

## Recommendations (FlashAttention Case)
1. **Raise arithmetic intensity**: reuse K/V tiles from shared memory, reduce global memory traffic.  
2. **Threadblock mapping**: map one block per output tile to avoid race conditions.  
3. **Efficient memory access**: ensure coalesced loads, apply double buffering.  
4. **Numerical stability**: adopt online softmax, minimizing memory footprint.  
5. **Maximize occupancy**: balance register allocation and shared memory to allow many resident warps.

## Conclusion
FlashAttention can approach the Roofline upper bound when tiling and loop restructuring are applied carefully. Under small `I`, memory bandwidth dominates, requiring optimization of data reuse. For large `I`, compute throughput dominates, requiring warp-level efficiency. Combined with Amdahl’s and Gustafson’s perspectives, the practical performance is shaped by both workload scaling and hardware limits.
