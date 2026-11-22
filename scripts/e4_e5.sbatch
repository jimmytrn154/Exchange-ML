#!/usr/bin/env bash
#SBATCH --job-name=e4-e5
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/e4-e5-%j.log

set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit this script using sbatch}"

args=()
case "${1:-full}" in
    full) ;;
    smoke) args=(--smoke) ;;
    check) args=(--check) ;;
    *) echo "Usage: sbatch scripts/e4_e5.sbatch [full|smoke|check]" >&2; exit 2 ;;
esac

# This is a CPU-only postprocessing job. Do not request or initialize CUDA.
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/exchange-e4-e5-mpl-${SLURM_JOB_ID}"
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_e4_e5.py "${args[@]}"
