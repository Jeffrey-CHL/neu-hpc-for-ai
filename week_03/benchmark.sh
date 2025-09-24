#!/bin/bash
# =============================
# File: benchmark.sh
# =============================
# Benchmark script for Week03 GEMM
# Run different modes and TILE sizes and collect timing + verification results.

set -e

# matrix sizes
M=1024
N=1024
K=1024

# alpha and beta
ALPHA=1.0
BETA=1.0

# TILE sizes
tiles=(16 32)

# modes
modes=(naive tiled)

for mode in "${modes[@]}"; do
  for tile in "${tiles[@]}"; do
    echo "=============================================="
    echo "Running mode=$mode TILE=$tile"
    ./gemm $M $N $K --mode $mode --ta N --tb N --alpha $ALPHA --beta $BETA --tile $tile
  done
done