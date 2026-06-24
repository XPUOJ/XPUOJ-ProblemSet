# Agent Notes

## Slurm and GPU Testing

- This cluster uses Slurm. The login node is only for editing, compiling, and submitting jobs. Do not run GPU tests or benchmarks directly on the login node.
- For this workspace, use the A800 resources for GPU tests unless the user explicitly asks for another GPU.
- The A800 partition exists as `a800`, with node `g07`.
- Current Slurm query showed `g07` has `gpu:nvidia:4` and `shard:nvidia:64`.
- A smoke Slurm job has been verified on `g07`; `nvidia-smi` reported `NVIDIA A800 80GB PCIe`.
- Prefer exclusive GPU allocation for performance tests: `--partition=a800 --nodelist=g07 --gres=gpu:1`.
- Use shard mode only for light checks when performance isolation is not important: `--gres=shard:1`.
- The A800 partition time limit is currently `2-00:00:00`; keep routine test jobs much shorter unless needed.
- Load CUDA in Slurm jobs with `module load cuda/12.2` unless the task has a stronger project-specific requirement.
- When creating batch jobs, use a `#!/bin/bash` sbatch script. Plain `sbatch --wrap` may execute through `/bin/sh` on this cluster, which does not support `set -o pipefail`.

Use `scripts/sbatch-a800.sh` for ordinary test commands:

```bash
scripts/sbatch-a800.sh python your_test.py
```

The helper submits the command to `g07` through Slurm and writes logs under `slurm_logs/`.

See `docs/slurm-a800.md` for concrete commands and troubleshooting notes.

## SCC Optimization Goal

- The current project goal is to optimize the three SCC operators with AKO4ALL on A800:
  - `USTB-SCC-A-fused-swiglu-up-projection`
  - `USTB-SCC-B-online-softmax-gemm`
  - `USTB-SCC-C-mixed-precision-grouped-gemm`
- The outcome should be best stable per-case runtimes on A800, used to propose OJ hardware-limit timing anchors.
- See `docs/scc-ako4all-goal.md` and `HINTS.md` before starting or resuming optimization work.
