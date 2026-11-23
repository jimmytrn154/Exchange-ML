#!/usr/bin/env bash
# User-run: metadata preflight + synthetic smoke before full sensitivity.
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
mode=${1:-full}
[[ $mode == full || $mode == smoke ]] || { echo 'Usage: bash scripts/submit_evaluator_sensitivity.sh [full|smoke]' >&2; exit 2; }
mkdir -p outputs/logs
smoke=$(sbatch --parsable --time=00:20:00 scripts/evaluator_sensitivity.sbatch smoke)
smoke=${smoke%%;*}; [[ $smoke =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm smoke ID' >&2; exit 1; }
echo "Evaluator sensitivity synthetic smoke: $smoke"
[[ $mode == smoke ]] && exit 0
full=$(sbatch --parsable --dependency="afterok:$smoke" scripts/evaluator_sensitivity.sbatch full)
full=${full%%;*}; [[ $full =~ ^[0-9]+$ ]] || { echo 'Invalid Slurm full ID' >&2; exit 1; }
echo "Evaluator sensitivity full CPU: $full"
echo 'Results: outputs/evaluator_sensitivity_v1'
echo 'Supplementary analysis of already-inspected validation/heldout; no inference or fresh confirmation.'
