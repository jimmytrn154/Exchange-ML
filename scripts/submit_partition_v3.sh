#!/usr/bin/env bash
# User-run full development: verify smoke, then score the frozen 40 validation CTs.
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mode=${1:-full}
[[ $mode == smoke || $mode == full ]] || {
    echo 'Usage: bash scripts/submit_partition_v3.sh [smoke|full]' >&2
    exit 2
}
mkdir -p outputs/logs
smoke=$(sbatch --parsable scripts/partition_v3.sbatch smoke)
smoke=${smoke%%;*}
[[ $smoke =~ ^[0-9]+$ ]] || { echo "Invalid Slurm smoke job ID: $smoke" >&2; exit 1; }
echo "Partition v3 CPU smoke job: $smoke"
if [[ $mode == smoke ]]; then exit 0; fi
full=$(sbatch --parsable --dependency="afterok:$smoke" scripts/partition_v3.sbatch full)
full=${full%%;*}
[[ $full =~ ^[0-9]+$ ]] || { echo "Invalid Slurm development job ID: $full" >&2; exit 1; }
echo "Partition v3 CPU development job: $full (after successful smoke $smoke)"
echo 'Results: outputs/partition_v3_development_v1'
echo 'Logs: outputs/logs/partition-v3-<job_id>.log'
echo 'Development only: no held-out confirmation or E6 training is submitted.'
