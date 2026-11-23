#!/usr/bin/env bash
#SBATCH --job-name=eval-sensitivity
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/logs/evaluator-sensitivity-%j.log
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit using sbatch}"
mode=${1:-full}
[[ $mode == full || $mode == smoke || $mode == check ]] || { echo 'Usage: sbatch scripts/evaluator_sensitivity.sbatch [full|smoke|check]' >&2; exit 2; }
# CPU evaluation of saved predictions, no CUDA allocation or new inference.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/evaluator-sensitivity-mpl-${SLURM_JOB_ID}"
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_evaluator_sensitivity.py "$mode"
