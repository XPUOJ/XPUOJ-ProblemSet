# Slurm A800 Usage

This workspace should run GPU tests through Slurm on the A800 node.

## Confirmed Resource

Checked with:

```bash
sinfo -p a800 -o '%P %a %l %D %G %N'
scontrol show node g07
```

Observed:

```text
PARTITION AVAIL TIMELIMIT NODES GRES NODELIST
a800 up 2-00:00:00 1 gpu:nvidia:4(S:0-1),shard:nvidia:64(S:0-1) g07
```

`g07` is the A800 node. It exposes 4 GPUs and 64 shards.

A smoke job submitted through `scripts/sbatch-a800.sh` completed on `g07` with:

```text
host=g07
cuda_visible_devices=0
GPU 0: NVIDIA A800 80GB PCIe
```

## Rules

- Do not run GPU programs on the login node.
- Submit GPU tests with Slurm to `--partition=a800 --nodelist=g07`.
- Use `--gres=gpu:1` for correctness plus performance tests.
- Use `--gres=shard:1` only for quick, light checks where isolated GPU performance does not matter.
- Cancel interactive allocations when finished with `scancel <job_id>`.
- Prefer an explicit `#!/bin/bash` sbatch script. On this cluster, `sbatch --wrap` may run under `/bin/sh`, so bash-only options such as `set -o pipefail` can fail.

## Batch Test Command

Use the helper script from the repository root:

```bash
scripts/sbatch-a800.sh python your_test.py
```

Useful environment overrides:

```bash
A800_TIME=00:20:00 scripts/sbatch-a800.sh python your_test.py
A800_GRES=gpu:4 A800_NTASKS=4 scripts/sbatch-a800.sh torchrun --nproc_per_node=4 train.py
A800_GRES=shard:1 scripts/sbatch-a800.sh python smoke_test.py
```

Defaults:

```text
partition: a800
nodelist:  g07
gres:      gpu:1
time:      01:00:00
cpus:      16
module:    cuda/12.2
logs:      slurm_logs/
```

## Interactive Shell

For an interactive shell on the A800 node:

```bash
srun -p a800 --nodelist=g07 --gres=gpu:1 --time=01:00:00 --pty /bin/bash
```

Inside the shell:

```bash
module load cuda/12.2
nvidia-smi
```

Exit the shell to release the allocation.

## Manual sbatch Template

```bash
#!/bin/bash
#SBATCH --partition=a800
#SBATCH --nodelist=g07
#SBATCH --job-name=xpuoj-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.2

cd /share/home/lianghaotian/xpuoj
echo "host=$(hostname)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python your_test.py
```

Submit with:

```bash
mkdir -p slurm_logs
sbatch script.sbatch
```

## Status Commands

```bash
sinfo -p a800 -o '%P %a %l %D %G %N'
squeue --format="%.18i %.9P %.30j %.15u %.8T %.10M %.12l %.6D %R"
scancel <job_id>
```
