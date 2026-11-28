#!/usr/bin/env bash
# Continue the existing campaign without changing its hash-bound launchers.
set -euo pipefail
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
dry_run=false
case ${1:-} in
    '') ;;
    --dry-run) dry_run=true ;;
    *) echo 'Usage: bash scripts/finish_ribfrac_fast.sh [--dry-run]' >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || exit 2

if ! $dry_run; then
    exec 9>"/tmp/exchange-ribfrac-fast-finish-${UID}.lock"
    flock -n 9 || { echo 'Another E6 submission is in progress.' >&2; exit 1; }
    active=$(squeue --user="$(id -un)" --noheader --format='%j')
    if [[ $active == *ribfrac-fast* ]]; then
        echo 'E6 jobs are already running/pending; inspect squeue before resubmitting.' >&2
        exit 1
    fi
    mkdir -p outputs/logs
fi

submitted=''
dry_id=9000
submit() {
    if $dry_run; then
        printf 'sbatch --parsable ' >&2
        printf '%q ' "$@" >&2
        printf '\n' >&2
        dry_id=$((dry_id + 1))
        submitted=$dry_id
    else
        submitted=$(sbatch --parsable "$@")
        submitted=${submitted%%;*}
        [[ $submitted =~ ^[0-9]+$ ]] || { echo 'Invalid sbatch job ID.' >&2; exit 1; }
    fi
}

submit --job-name=ribfrac-fast-preflight --time=00:15:00 scripts/ribfrac_finish.sbatch preflight
preflight=$submitted
echo "E6 preflight: $preflight"
submit --job-name=ribfrac-fast-infer --dependency="afterok:$preflight" --kill-on-invalid-dep=yes \
    --array=0-2%2 --gres=gpu:1 --time=24:00:00 \
    --output=outputs/logs/ribfrac-fast-infer-%A_%a.log scripts/ribfrac_fast.sbatch infer
infer=$submitted
echo "E6 inference: $infer (one GPU/member, at most two members concurrently)"
submit --job-name=ribfrac-fast-evaluate --dependency="afterok:$infer" --kill-on-invalid-dep=yes \
    --time=24:00:00 scripts/ribfrac_fast.sbatch evaluate
evaluate=$submitted
echo "E6 evaluation: $evaluate (CPU)"
submit --job-name=ribfrac-fast-audit --dependency="afterok:$evaluate" --kill-on-invalid-dep=yes \
    --time=04:00:00 scripts/ribfrac_finish.sbatch audit
echo "E6 audit + positive/negative cohort report: $submitted (CPU)"
if $dry_run; then
    echo 'DRY RUN: no jobs submitted; displayed IDs are placeholders.'
else
    echo 'Monitor: squeue -u '"$(id -un)"
fi
