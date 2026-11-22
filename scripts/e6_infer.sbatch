#!/usr/bin/env bash
#SBATCH --job-name=e6-infer
#SBATCH --partition=gpu
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/e6-infer-%A_%a.log
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Submit using sbatch}"
task=${SLURM_ARRAY_TASK_ID:?Submit as array}
[[ $task =~ ^[0-2]$ ]] || { echo 'Invalid member array index' >&2; exit 2; }
models=(member_01 member_02 member_03)
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export MPLCONFIGDIR=/tmp/exchange-e6-mpl
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_e6.py infer --model "${models[task]}"
