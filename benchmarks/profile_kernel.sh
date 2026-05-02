#!/usr/bin/env bash
set -euo pipefail

mkdir -p benchmarks/results
ncu \
  --set full \
  --target-processes all \
  --export benchmarks/results/kernel_ncu \
  --force-overwrite \
  python benchmarks/bench_kernel.py \
    --dtype fp16 \
    --device cuda:0 \
    --warmup 3 \
    --iters 10

