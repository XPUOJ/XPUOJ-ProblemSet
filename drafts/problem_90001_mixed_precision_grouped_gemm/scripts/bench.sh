#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:-baseline}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROBLEM="drafts/problem_90001_mixed_precision_grouped_gemm"
TRAJ_DIR="$ROOT/$PROBLEM/trajectory/$LABEL-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$TRAJ_DIR"
cp -r "$ROOT/$PROBLEM/solution" "$TRAJ_DIR/"

cd "$ROOT"
conda run -n matris311 python tools/local_cuda_runner.py \
    "$PROBLEM" \
    "$PROBLEM/solution/kernel.cu" \
    --case all \
    --gpu 0 \
    --keep-going \
    2>&1 | tee "$TRAJ_DIR/bench_output.txt"
