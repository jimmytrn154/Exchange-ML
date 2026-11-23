#!/usr/bin/env bash
# User-run full development; smoke is synthetic and uses a separate root.
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mode=${1:-full}
[[ $mode == smoke || $mode == full ]] || {
    echo 'Usage: bash scripts/submit_count_rank.sh [smoke|full]' >&2
    exit 2
}
mkdir -p outputs/logs
smoke=$(sbatch --parsable scripts/count_rank.sbatch smoke)
smoke=${smoke%%;*}
[[ $smoke =~ ^[0-9]+$ ]] || { echo "Invalid Slurm smoke job ID: $smoke" >&2; exit 1; }
echo "Count ablation CPU synthetic smoke job: $smoke"
if [[ $mode == smoke ]]; then exit 0; fi
full=$(sbatch --parsable --dependency="afterok:$smoke" scripts/count_rank.sbatch full)
full=${full%%;*}
[[ $full =~ ^[0-9]+$ ]] || { echo "Invalid Slurm development job ID: $full" >&2; exit 1; }
echo "Count ablation CPU development job: $full (after successful smoke $smoke)"
echo 'Results: outputs/count_ablation_development_v1'
echo 'Logs: outputs/logs/count-ablation-<job_id>.log'
echo 'Uses audited frozen 40-CT v3/v4 tables; no image reads, GPU inference or fresh confirmation.'
