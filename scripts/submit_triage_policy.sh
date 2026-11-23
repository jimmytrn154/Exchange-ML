#!/usr/bin/env bash
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mode=${1:-full}
[[ $# -le 1 && ( $mode == full || $mode == smoke || $mode == check ) ]] || { echo 'Usage: bash scripts/submit_triage_policy.sh [full|smoke|check]' >&2; exit 2; }
mkdir -p outputs/logs
submit(){
    local jid
    jid=$(sbatch --parsable "$@")
    jid=${jid%%;*}
    [[ $jid =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm job ID' >&2; return 1; }
    echo "$jid"
}
if [[ $mode == check ]]; then
    jid=$(submit scripts/triage_policy.sbatch check)
    echo "CPU metadata preflight: $jid";exit 0
fi
smoke=$(submit scripts/triage_policy.sbatch smoke)
echo "CPU synthetic smoke: $smoke"
[[ $mode != smoke ]] || exit 0
full=$(submit --dependency="afterok:$smoke" scripts/triage_policy.sbatch full)
echo "CPU full saved-table policy analysis: $full"
echo 'Results: outputs/triage_policy_development_v1; no new inference, E6 change, or fresh confirmation.'
