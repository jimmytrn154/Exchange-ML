#!/usr/bin/env bash
#SBATCH --job-name=e6-eval
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/e6-eval-%j.log
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit using sbatch}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_e6.py evaluate
