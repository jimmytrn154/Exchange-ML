#!/usr/bin/env bash
#SBATCH --job-name=ribfrac-cohorts
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/logs/ribfrac-cohorts-%j.log

set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit with sbatch after E6 evaluation completes}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/report_ribfrac_cohorts.py "$@"
