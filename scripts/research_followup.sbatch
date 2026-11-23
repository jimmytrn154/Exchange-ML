#!/usr/bin/env bash
#SBATCH --job-name=research-preflight
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=outputs/logs/research-preflight-%j.log

set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit with sbatch}"
# Partition name is gpu; this job requests CPU resources only.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python_bin=/home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python
"$python_bin" -B scripts/audit_e1_e5_evidence.py
"$python_bin" -B scripts/audit_reproducibility.py --output-root outputs/research_followup_v1/preflight
"$python_bin" -B scripts/run_count_rank.py check
"$python_bin" -B scripts/run_evaluator_sensitivity.py check
