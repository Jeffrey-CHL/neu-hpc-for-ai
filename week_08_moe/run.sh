#!/usr/bin/env bash
set -e

# Simple convenience script to build and run the MoE example.
# This assumes you are already inside a CUDA-enabled environment (e.g., Modal GPU container).

echo "[run.sh] Building deepseek_moe..."
make

echo "[run.sh] Running deepseek_moe..."
./deepseek_moe
