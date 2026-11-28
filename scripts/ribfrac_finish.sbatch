#!/usr/bin/env bash
#SBATCH --job-name=ribfrac-fast-finish
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/ribfrac-fast-finish-%j.log
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit with sbatch}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case ${1:?preflight or audit required} in
    preflight)
        exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/check_ribfrac_fast.py preflight
        ;;
    audit)
        /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/check_ribfrac_fast.py audit
        exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/report_ribfrac_cohorts.py
        ;;
    *) echo 'Expected preflight or audit.' >&2; exit 2 ;;
esac
