#!/usr/bin/env bash
# User-run only. Smoke is a synthetic implementation check, not E6 evidence.
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mode=${1:-full}
[[ $mode == smoke || $mode == full ]] || { echo 'Usage: bash scripts/submit_e6.sh [smoke|full]' >&2; exit 2; }
mkdir -p outputs/logs
# Keep smoke on its own root even when the full experiment uses another disk.
export E6_SMOKE_ROOT=${E6_SMOKE_ROOT:-/mnt/sdb/shared/dang.cpm/Exchange-ML/outputs/smoke_e6}
smoke=$(sbatch --parsable scripts/e6_smoke.sbatch)
smoke=${smoke%%;*}
echo "E6 synthetic smoke job: $smoke"
if [[ $mode == smoke ]]; then exit 0; fi
prepare=$(sbatch --parsable --dependency="afterok:$smoke" scripts/e6_prepare.sbatch)
prepare=${prepare%%;*}
preprocess=$(sbatch --parsable --dependency="afterok:$prepare" scripts/e6_preprocess.sbatch)
preprocess=${preprocess%%;*}
train=$(sbatch --parsable --array=0-2%2 --dependency="afterok:$preprocess" scripts/e6_train.sbatch)
train=${train%%;*}
infer=$(sbatch --parsable --array=0-2%2 --dependency="afterok:$train" scripts/e6_infer.sbatch)
infer=${infer%%;*}
evaluate=$(sbatch --parsable --dependency="afterok:$infer" scripts/e6_evaluate.sbatch)
evaluate=${evaluate%%;*}
echo "E6 jobs: smoke=$smoke prepare=$prepare preprocess=$preprocess train=$train infer=$infer evaluate=$evaluate"
echo "Full output: ${E6_OUTPUT_ROOT:-outputs/external_ribfrac}"
echo 'GPU arrays run at most two one-GPU tasks concurrently. Full runtime is not yet measured.'
echo 'After timeout/failure: inspect logs, cancel stale DependencyNeverSatisfied jobs, then resubmit this same command to resume verified stages.'
