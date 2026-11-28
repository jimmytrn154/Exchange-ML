#!/usr/bin/env bash
# Compile the manuscript and its saved tables/figures without regenerating data.
set -euo pipefail
paper_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$paper_dir"
compiler=${TECTONIC:-$(command -v tectonic || true)}
if [[ -z $compiler && -x $HOME/.local/bin/tectonic ]]; then
    compiler=$HOME/.local/bin/tectonic
fi
[[ -n $compiler ]] || { echo 'Install Tectonic or set TECTONIC=/path/to/tectonic.' >&2; exit 1; }
mkdir -p build
"$compiler" --only-cached --keep-logs --outdir build manuscript.tex
# Keep the last successful PDF if compilation fails.
cp build/manuscript.pdf manuscript.pdf
