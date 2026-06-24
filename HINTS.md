# AKO4ALL Hints

- Target the three SCC problems in this workspace:
  - `USTB-SCC-A-fused-swiglu-up-projection`
  - `USTB-SCC-B-online-softmax-gemm`
  - `USTB-SCC-C-mixed-precision-grouped-gemm`
- Run all GPU benchmarks through Slurm on the A800 node `g07`; do not run GPU workloads on the login node.
- Use `scripts/sbatch-a800.sh` or an equivalent explicit `#!/bin/bash` sbatch script for GPU tests.
- The optimization goal is to find the best practical implementation for each SCC operator on NVIDIA A800, then use the measured best-case times to propose per-case hardware-limit times for OJ scoring.
- Treat `baseline` time as the lower scoring anchor and the optimized/hardware-limit time as the 100-point anchor. The exact OJ field names for these anchors still need confirmation from platform docs or maintainers.
- Do not reward-hack benchmark timing. Optimize real kernel latency while preserving the published interface and correctness checks.
- For final hardware-limit recommendations, run stable full measurements on exclusive GPU allocation (`--gres=gpu:1`) and record per-case mean/min/std when available.
