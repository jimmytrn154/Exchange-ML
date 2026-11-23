#!/usr/bin/env bash
#SBATCH --job-name=count-ablation
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/logs/count-ablation-%j.log

set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit this script using sbatch}"
mode=${1:-full}
[[ $mode == full || $mode == smoke || $mode == check ]] || {
    echo 'Usage: sbatch scripts/count_rank.sbatch [full|smoke|check]' >&2
    exit 2
}
# The partition is named gpu; this table-only job requests no GPUs.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_count_rank.py "$mode"
