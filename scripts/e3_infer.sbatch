#!/usr/bin/env bash
#SBATCH --job-name=e3-infer
#SBATCH --partition=gpu
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/e3-%A_%a.log

set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML

models=(member_01 member_02 member_03 stage2)
task_id=${SLURM_ARRAY_TASK_ID:?Submit this script using sbatch}
if [[ ! $task_id =~ ^[0-3]$ ]]; then
    echo "Invalid E3 model index: $task_id" >&2
    exit 2
fi

args=()
case "${1:-full}" in
    smoke) args=(--smoke) ;;
    full) ;;
    *) echo "Usage: sbatch scripts/e3_infer.sbatch [smoke|full]" >&2; exit 2 ;;
esac

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/exchange-e3-mpl
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u \
    scripts/run_e3.py infer --model "${models[task_id]}" "${args[@]}"
