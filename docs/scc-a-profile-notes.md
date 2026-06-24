# SCC A Profiling Notes

Date: 2026-06-24

Hardware: Slurm `a800/g07`, `NVIDIA A800 80GB PCIe`, exclusive `--gres=gpu:1`

Environment:

- Python: `/share/home/lianghaotian/miniforge3/envs/env/bin/python`
- PyTorch: `2.8.0+cu128`
- Triton: `3.4.0`
- CUDA module used for jobs: `cuda/12.8`

## Baseline

The profiled baseline is the problem statement Triton implementation:

- `BLOCK_M=16`
- `BLOCK_N=16`
- `BLOCK_K=32`
- `num_warps=4`
- `num_stages=3`

All six cases passed correctness.

| Case | M | K | N | Mean ms | Min ms | Mean TFLOPS | Arithmetic intensity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 1024 | 2048 | 0.4555 | 0.4536 | 18.90 | 585.6 |
| 2 | 2048 | 4096 | 3072 | 5.0438 | 5.0074 | 20.45 | 1293.7 |
| 3 | 4096 | 4096 | 4096 | 14.0644 | 14.0237 | 19.55 | 2048.5 |
| 4 | 4097 | 3073 | 2305 | 8.4110 | 8.3804 | 13.81 | 1603.9 |
| 5 | 8192 | 2048 | 4096 | 14.3322 | 14.1957 | 19.20 | 2049.5 |
| 6 | 1536 | 6144 | 6144 | 11.2628 | 11.1882 | 20.60 | 1228.9 |

These arithmetic intensities are high, so the baseline should be compute-bound. The observed 14-21 TFLOPS is far below A800 BF16 Tensor Core capability, pointing to poor matmul tiling / Tensor Core utilization rather than HBM bandwidth as the first-order bottleneck.

## Tile Sweep

Case 3 (`4096 x 4096 @ 4096 x 4096`) was swept across several Triton tile configs.

Best tested config:

```text
BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=3
mean=1.2326 ms, min=1.2298 ms, 223.11 TFLOPS
```

This is about 11.4x faster than the statement baseline on case 3.

## Cross-Case Check For 64x128x64

The `64x128x64` config is strong on the regular cases but bad on case 4:

| Case | Mean ms | Mean TFLOPS | Note |
|---:|---:|---:|---|
| 1 | 0.0864 | 99.59 | much faster |
| 2 | 0.5280 | 195.33 | much faster |
| 3 | 1.3294 | 206.88 | much faster |
| 4 | 54.3652 | 2.14 | severe regression |
| 5 | 1.2837 | 214.35 | much faster |
| 6 | 1.0509 | 220.76 | much faster |

Case 4 is the non-divisible shape `(M=4097, K=3073, N=2305)`. The regression appears only for some large `BLOCK_K=64` configs with awkward tails.

Case 4 sweep found the best tested config:

```text
BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=3
mean=1.7386 ms, min=1.7254 ms, 66.81 TFLOPS
```

## Nsight Compute

`ncu` is available at `/share/app/cuda/cuda-12.8/bin/ncu`, but the first attempt failed with:

```text
Failed to open/create lock file /tmp/nsight-compute-lock
InterprocessLockFailed
```

CUDA event timing and tile sweeps were still enough to identify the first hotspot. Retry ncu later with a fixed lock/temp configuration if lower-level counters are needed.

## Optimization Directions

1. Replace the statement baseline tile config. Use `64x128x64` for regular cases and a safe fallback such as `64x128x32` for the non-divisible case 4 shape.
2. Add shape-based dispatch in `run_kernel` because the six benchmark cases are fixed and public in `testcase_config.py`.
3. Continue sweeping around the current winners:
   - regular cases: `64x128x64`, `128x64x64`, maybe `64x256x64` if register pressure allows
   - non-divisible case 4: `64x128x32`, `32x128x32`, `128x64x32`
4. After tile selection, optimize the epilogue:
   - keep bias load and SiLU fused in the final store path
   - prefer approximate sigmoid/SILU only if correctness remains stable under `rtol=5e-2, atol=5e-2`
5. Consider splitting regular and irregular tails:
   - fast path without masks for divisible full tiles
   - masked fallback only for edge tiles
6. Use per-case stable full measurements as candidate hardware-limit anchors after the optimized implementation is finalized.
