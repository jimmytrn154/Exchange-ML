#!/usr/bin/env bash
#SBATCH --job-name=e3-eval
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/e3-eval-%j.log

set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit this script using sbatch}"

args=()
case "${1:-full}" in
    smoke) args=(--smoke) ;;
    full) ;;
    *) echo "Usage: sbatch scripts/e3_evaluate.sbatch [smoke|full]" >&2; exit 2 ;;
esac

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/exchange-e3-mpl
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u \
    scripts/run_e3.py evaluate "${args[@]}"
