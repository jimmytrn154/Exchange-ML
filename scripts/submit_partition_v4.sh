#!/usr/bin/env bash
# User-run full development; smoke is synthetic and uses a separate root.
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mode=${1:-full}
[[ $mode == smoke || $mode == full ]] || {
    echo 'Usage: bash scripts/submit_partition_v4.sh [smoke|full]' >&2
    exit 2
}
mkdir -p outputs/logs
smoke=$(sbatch --parsable scripts/partition_v4.sbatch smoke)
smoke=${smoke%%;*}
[[ $smoke =~ ^[0-9]+$ ]] || { echo "Invalid Slurm smoke job ID: $smoke" >&2; exit 1; }
echo "Partition v4 CPU synthetic smoke job: $smoke"
if [[ $mode == smoke ]]; then exit 0; fi
full=$(sbatch --parsable --dependency="afterok:$smoke" scripts/partition_v4.sbatch full)
full=${full%%;*}
[[ $full =~ ^[0-9]+$ ]] || { echo "Invalid Slurm development job ID: $full" >&2; exit 1; }
echo "Partition v4 CPU development job: $full (after successful smoke $smoke)"
echo 'Results: outputs/partition_v4_development_v1'
echo 'Logs: outputs/logs/partition-v4-<job_id>.log'
echo 'Uses frozen 40-CT v3 tables; no image reads, GPU inference or fresh confirmation.'
