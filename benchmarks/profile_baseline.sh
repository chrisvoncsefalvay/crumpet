#!/usr/bin/env bash
set -euo pipefail

mkdir -p benchmarks/results
nsys profile \
  --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi \
  --output=benchmarks/results/baseline_nsys \
  python benchmarks/bench_baseline.py \
    --shape btcv \
    --dtype fp16 \
    --device cuda:0 \
    --warmup 3 \
    --iters 10 \
    --output-json benchmarks/results/baseline_profile_results.json

