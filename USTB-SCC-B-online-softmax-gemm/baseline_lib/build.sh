#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

nvcc -O3 -std=c++17 -arch=sm_80 -shared -Xcompiler -fPIC \
  "${script_dir}/scc_b_baseline.cu" \
  -o "${script_dir}/libscc_b_baseline.so"
