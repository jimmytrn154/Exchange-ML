#!/usr/bin/env bash
# User-run CPU follow-up. All original experiments and E6 writers stay separate.
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mode=${1:-full}
[[ $# -le 1 && ( $mode == full || $mode == smoke || $mode == check ) ]] || {
    echo 'Usage: bash scripts/submit_research_followup.sh [full|smoke|check]' >&2
    exit 2
}
mkdir -p outputs/logs
submit() {
    local submitted
    submitted=$(sbatch --parsable "$@")
    submitted=${submitted%%;*}
    [[ $submitted =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job ID: $submitted" >&2; return 1; }
    echo "$submitted"
}
preflight=$(submit scripts/research_followup.sbatch)
echo "CPU reproducibility/source preflight: $preflight"
[[ $mode != check ]] || exit 0
count_smoke=$(submit --dependency="afterok:$preflight" scripts/count_rank.sbatch smoke)
eval_smoke=$(submit --dependency="afterok:$preflight" scripts/evaluator_sensitivity.sbatch smoke)
echo "CPU synthetic smokes: count=$count_smoke evaluator=$eval_smoke"
[[ $mode != smoke ]] || exit 0
count_full=$(submit --dependency="afterok:$count_smoke" scripts/count_rank.sbatch full)
eval_full=$(submit --dependency="afterok:$eval_smoke" scripts/evaluator_sensitivity.sbatch full)
echo "CPU full jobs: count=$count_full evaluator=$eval_full"
echo 'Read job logs and complete manifests, including negative/undefined results.'
echo 'Count ablation is validation development; evaluator sensitivity reuses inspected E1/E3 cases.'
echo 'No fresh confirmation, GPU allocation, E6 cancellation or E6 restart is performed.'
