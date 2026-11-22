#!/usr/bin/env bash
#SBATCH --job-name=e6-preprocess
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/e6-preprocess-%j.log
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit using sbatch}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export MPLCONFIGDIR=/tmp/exchange-e6-mpl
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_e6.py preprocess --processes 8
