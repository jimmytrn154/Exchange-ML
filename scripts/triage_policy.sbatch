#!/usr/bin/env bash
#SBATCH --job-name=dual-triage
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/dual-triage-%j.log
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit using sbatch}"
mode=${1:-full}
[[ $mode == full || $mode == smoke || $mode == check ]] || exit 2
# CPU-only saved-table policy evaluation, despite the partition name.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_triage_policy.py "$mode"
