#!/usr/bin/env bash
#SBATCH --job-name=e6-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/logs/e6-smoke-%j.log
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit using sbatch}"
export PYTHONHASHSEED=32001
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export MPLCONFIGDIR=/tmp/exchange-e6-mpl
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_e6.py smoke
