# SCC B/C CUDA Baseline Shared Libraries

SCC B and SCC C use the statement starter CUDA code as the OJ scoring baseline.
Their `testcase_config.py` files keep the original PyTorch implementation as
`_torch_reference(...)`, while `baseline(...)` loads a precompiled shared library
from `baseline_lib/` and calls the exported `extern "C" run_kernel` symbol.

This keeps the scoring anchor aligned with the starter CUDA implementation: if
the platform times `baseline()` for `T_b`, the 50-point baseline is the starter
CUDA code rather than the PyTorch reference.

## Files

- `USTB-SCC-B-online-softmax-gemm/baseline_lib/scc_b_baseline.cu`
- `USTB-SCC-B-online-softmax-gemm/baseline_lib/build.sh`
- `USTB-SCC-B-online-softmax-gemm/baseline_lib/libscc_b_baseline.so`
- `USTB-SCC-C-mixed-precision-grouped-gemm/baseline_lib/scc_c_baseline.cu`
- `USTB-SCC-C-mixed-precision-grouped-gemm/baseline_lib/build.sh`
- `USTB-SCC-C-mixed-precision-grouped-gemm/baseline_lib/libscc_c_baseline.so`

## Build

Build on the target A800 / SM80 environment before deployment:

```bash
module load cuda/12.8
bash USTB-SCC-B-online-softmax-gemm/baseline_lib/build.sh
bash USTB-SCC-C-mixed-precision-grouped-gemm/baseline_lib/build.sh
```

Each script uses:

```bash
nvcc -O3 -std=c++17 -arch=sm_80 -shared -Xcompiler -fPIC
```

The shared libraries use only a C ABI with raw pointers and scalar arguments.
They do not depend on the PyTorch C++ extension ABI.

## A800 Verification

Verified on Slurm job `79732`, node `g07`, partition `a800`, GPU type
`NVIDIA A800 80GB PCIe`.

Command:

```bash
A800_TIME=02:00:00 A800_GRES=gpu:1 A800_JOB_NAME=scc-bc-allcheck \
  scripts/sbatch-a800.sh bash -lc 'module load cuda/12.8; \
  export PATH=/share/home/lianghaotian/miniforge3/envs/env/bin:$PATH; \
  bash USTB-SCC-B-online-softmax-gemm/baseline_lib/build.sh; \
  bash USTB-SCC-C-mixed-precision-grouped-gemm/baseline_lib/build.sh; \
  python experiments/scc_bc_baseline/verify_bc_baseline.py all'
```

Results:

```text
B case 1: correct=True
B case 2: correct=True
B case 3: correct=True
B case 4: correct=True
B case 5: correct=True
B case 6: correct=True
C case 1: correct=True
C case 2: correct=True
C case 3: correct=True
C case 4: correct=True
C case 5: correct=True
C case 6: correct=True
C case 7: correct=True
C case 8: correct=True
```

If the OJ package does not allow committing `.so` files, keep the `.cu` and
`build.sh` files in the problem package and run the build scripts on the OJ A800
environment before publishing the problems.
