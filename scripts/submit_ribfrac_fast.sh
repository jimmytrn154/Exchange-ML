#!/usr/bin/env bash
# User-run revision: benchmark and cache must pass before original jobs stop.
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mode=${1:-full}
[[ $mode == benchmark || $mode == full ]] || { echo 'Usage: bash scripts/submit_ribfrac_fast.sh [benchmark|full]' >&2; exit 2; }
mkdir -p outputs/logs
bench=$(sbatch --parsable --gres=gpu:1 --time=00:15:00 scripts/ribfrac_fast.sbatch benchmark)
bench=${bench%%;*}; [[ $bench =~ ^[0-9]+$ ]] || exit 1
echo "GPU batch benchmark: $bench"
[[ $mode == full ]] || exit 0
cache=$(sbatch --parsable --dependency="afterok:$bench" scripts/ribfrac_fast.sbatch cache)
cache=${cache%%;*}; [[ $cache =~ ^[0-9]+$ ]] || exit 1
switch=$(sbatch --parsable --dependency="afterok:$cache" scripts/ribfrac_fast.sbatch switch)
switch=${switch%%;*}; [[ $switch =~ ^[0-9]+$ ]] || exit 1
train=$(sbatch --parsable --array=0-2%2 --gres=gpu:1 --time=7-00:00:00 --output=outputs/logs/ribfrac-fast-train-%A_%a.log --dependency="afterok:$switch" scripts/ribfrac_fast.sbatch train)
train=${train%%;*}; [[ $train =~ ^[0-9]+$ ]] || exit 1
infer=$(sbatch --parsable --array=0-2%2 --gres=gpu:1 --time=24:00:00 --output=outputs/logs/ribfrac-fast-infer-%A_%a.log --dependency="afterok:$train" scripts/ribfrac_fast.sbatch infer)
infer=${infer%%;*}; [[ $infer =~ ^[0-9]+$ ]] || exit 1
evaluate=$(sbatch --parsable --dependency="afterok:$infer" scripts/ribfrac_fast.sbatch evaluate)
evaluate=${evaluate%%;*}; [[ $evaluate =~ ^[0-9]+$ ]] || exit 1
echo "Fast E6 jobs: benchmark=$bench cache=$cache handoff=$switch train=$train infer=$infer evaluate=$evaluate"
echo 'After benchmark/cache pass, handoff cancels original jobs 2880/2881/2882 and archives their latest checkpoints.'
echo 'New root: /mnt/disk1/backup_user/dang.cpm/Exchange-ML-E6-fast-v1'
echo 'Training uses at most two one-GPU tasks. This is an explicit mixed-batch protocol revision, not the original fixed-batch run.'
