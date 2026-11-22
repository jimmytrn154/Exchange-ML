#!/usr/bin/env bash
#SBATCH --job-name=partition-v3
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/partition-v3-%j.log

set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit this script using sbatch}"
mode=${1:-full}
[[ $mode == full || $mode == smoke || $mode == check ]] || {
    echo 'Usage: sbatch scripts/partition_v3.sbatch [full|smoke|check]' >&2
    exit 2
}

# The cluster's partition is named gpu; this job requests only CPUs, no GPU.
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_partition_v3.py "$mode"
