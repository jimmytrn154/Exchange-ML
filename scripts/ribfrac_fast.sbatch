#!/usr/bin/env bash
#SBATCH --job-name=ribfrac-fast
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/ribfrac-fast-%j.log
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
: "${SLURM_JOB_ID:?Use sbatch}"
stage=${1:?stage required}
shift
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export MPLCONFIGDIR=/tmp/ribfrac-fast-mpl
args=()
if [[ $stage == train || $stage == infer ]]; then
    task=${SLURM_ARRAY_TASK_ID:?member array required}
    [[ $task =~ ^[0-2]$ ]] || exit 2
    models=(member_01 member_02 member_03)
    args=(--model "${models[task]}")
    export PYTHONHASHSEED=$((32001 + task))
fi
exec /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -u scripts/run_ribfrac_fast.py "$stage" "${args[@]}" "$@"
