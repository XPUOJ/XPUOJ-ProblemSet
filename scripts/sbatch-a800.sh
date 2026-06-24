#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/sbatch-a800.sh <command> [args...]

Submit a command to the A800 Slurm node g07. GPU tests for this workspace
should use this helper instead of running directly on the login node.

Environment overrides:
  A800_JOB_NAME          default: xpuoj-test
  A800_TIME              default: 01:00:00
  A800_GRES              default: gpu:1
  A800_CPUS_PER_TASK     default: 16
  A800_NTASKS            default: 1
  A800_PARTITION         default: a800
  A800_NODELIST          default: g07
  A800_CUDA_MODULE       default: cuda/12.2
  A800_LOG_DIR           default: slurm_logs

Examples:
  scripts/sbatch-a800.sh python test.py
  A800_TIME=00:20:00 scripts/sbatch-a800.sh python test.py
  A800_GRES=shard:1 scripts/sbatch-a800.sh python smoke_test.py
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit $([[ $# -eq 0 ]] && echo 2 || echo 0)
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
job_name="${A800_JOB_NAME:-xpuoj-test}"
time_limit="${A800_TIME:-01:00:00}"
gres="${A800_GRES:-gpu:1}"
cpus="${A800_CPUS_PER_TASK:-16}"
ntasks="${A800_NTASKS:-1}"
partition="${A800_PARTITION:-a800}"
nodelist="${A800_NODELIST:-g07}"
cuda_module="${A800_CUDA_MODULE:-cuda/12.2}"
log_dir="${A800_LOG_DIR:-slurm_logs}"

mkdir -p "${repo_root}/${log_dir}"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/xpuoj-a800-sbatch.XXXXXX")"
trap 'rm -rf "${tmp_dir}"' EXIT

quoted_cmd=""
for arg in "$@"; do
  printf -v quoted_arg "%q" "$arg"
  quoted_cmd+=" ${quoted_arg}"
done
quoted_cmd="${quoted_cmd# }"

batch_script="${tmp_dir}/job.sbatch"
cat > "${batch_script}" <<EOF
#!/bin/bash
#SBATCH --partition=${partition}
#SBATCH --nodelist=${nodelist}
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=${ntasks}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --gres=${gres}
#SBATCH --time=${time_limit}
#SBATCH --output=${repo_root}/${log_dir}/%x-%j.out
#SBATCH --error=${repo_root}/${log_dir}/%x-%j.err

set -euo pipefail
cd "${repo_root}"
source /etc/profile.d/modules.sh 2>/dev/null || true
if command -v module >/dev/null 2>&1; then
  module load "${cuda_module}"
fi
echo "job_id=\${SLURM_JOB_ID:-unknown}"
echo "host=\$(hostname)"
echo "cuda_visible_devices=\${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi
${quoted_cmd}
EOF

sbatch "${batch_script}"
