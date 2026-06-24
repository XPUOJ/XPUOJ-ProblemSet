# SCC A AKO4ALL Iterations

## Summary

- Current best: iter 1 shape-based Triton tile dispatch.
- Best validated command path: Slurm `a800/g07`, exclusive `--gres=gpu:1`.
- Best result file: `experiments/scc_a_profile/iter1_dispatch_results.json`.

## Iteration 1

Label: `iter-1`

Change:

- Added shape-based dispatch in A-problem Triton `run_kernel`.
- Regular cases use `BLOCK_M=64, BLOCK_N=128, BLOCK_K=64`.
- Non-divisible case 4 `(M=4097, K=3073, N=2305)` uses `BLOCK_K=32` to avoid the severe `BK=64` tail regression found during profiling.

Benchmark:

- Command submitted through `scripts/sbatch-a800.sh`.
- Slurm job: `79722`
- Node: `g07`
- GPU: `NVIDIA A800 80GB PCIe`
- Correctness: all 6 cases passed.

Results:

| Case | Mean ms | Min ms | Mean TFLOPS | Baseline mean ms | Speedup vs profiled baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0893 | 0.0870 | 96.37 | 0.4555 | 5.10x |
| 2 | 0.5322 | 0.5274 | 193.79 | 5.0438 | 9.48x |
| 3 | 1.2959 | 1.2411 | 212.22 | 14.0644 | 10.85x |
| 4 | 1.8064 | 1.7654 | 64.30 | 8.4110 | 4.66x |
| 5 | 1.3151 | 1.2902 | 209.22 | 14.3322 | 10.90x |
| 6 | 1.0798 | 1.0721 | 214.85 | 11.2628 | 10.43x |

Analysis:

- The first bottleneck was confirmed to be poor Tensor Core utilization from tiny `16x16x32` tiles.
- Shape-based tile dispatch produces large speedups on all cases while keeping correctness stable.
- Case 4 remains the weakest case because its awkward non-divisible dimensions force heavier masking and lower effective Tensor Core utilization.

Next directions:

- Sweep per-case configs around the new winners instead of using only two dispatch configs.
- Focus especially on case 4 configs with `BK=32`.
- Explore larger `BN`/alternative `BM` for regular cases while watching register pressure.
