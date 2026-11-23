# Manuscript

Read `manuscript.pdf`; edit `manuscript.tex`. The English draft uses the supplied IEEE conference template and anonymous author metadata. The contributions concern component separation, a 12-score comparison on 100 CTs, and sensitivity to the definition of failure. They do not claim a new mathematical distance or universal superiority of the combined score.

`figures/overview.pdf` is the vector overview embedded in the paper; `overview.png` is its preview. It shows a real development CT and the three saved predictions on the same slice/crop, whole-3D-bone disagreement values, and actual risk–coverage curves on the separate 100-CT cohort. Instance colors carry no correspondence between models. Selection uses predictions alone. The receipt in `evidence/overview.json` records source hashes, geometry, slice/crop, cleanup, palettes and reconciled scores.

Build the tables, readable Markdown, source receipt and PDF:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B \
  paper/build_paper.py --compile
```

The default build verifies saved experiment manifests and overview provenance. Add `--refresh-overview` to reread the single CT and its three predictions and redraw the figure. Neither command trains models, performs inference, fits scores, or reruns experimental bootstrap analyses. A changed input or plotting script requires an explicit overview refresh; a changed experiment hash fails validation. The CT source must remain accessible for hash verification.

Compiler intermediates stay in ignored `build/`; only a successful compilation replaces `manuscript.pdf`. Tectonic uses its populated local cache with `--only-cached`; `--tectonic /path/to/tectonic` overrides executable discovery. A fresh machine needs the project Python environment, the saved inputs, and a populated Tectonic package/font cache.

- `manuscript.tex`, `references.bib`: canonical text and 10 cited records.
- `manuscript.md`, `main_table.tex`, `endpoint_table.tex`: generated views/tables; edit their sources instead.
- `build_paper.py`, `make_overview.py`: evidence checks and reproducible reporting.
- `conference-latex-template.zip`, `IEEEtran.cls`: supplied template archive and its unmodified class.
- `evidence/`: machine-readable figure/table provenance.
- `review/`: published-paper source ledgers and current manuscript checks.

Technical content must occupy pages 1–4, with acknowledgment and references on page 5; no margins or font sizes are reduced to obtain this layout. External RibFrac results remain unavailable. The completed exploratory triage-policy follow-up did not establish added benefit and is retained in experiment outputs/documentation, rather than presented as a contribution. The stronger partition-only control and shared-omission limit remain explicit in the paper.

Retired drafts, unused review setup, the redundant extracted template and old figure are preserved under `outputs/archive/paper_maintenance_20261123T131726Z/`. `review/archive` preserves the historical draft path through a relative symlink. A full before-state backup and cleanup inventory live in the sibling `.cleanup_backups/Exchange-ML-paper-20261123T131726Z/` directory. No model, dataset, checkpoint, experiment configuration or active E6 job is changed by this maintenance.

This is an author-review draft. Author affiliations and final author approval remain to be supplied; no submission or publication is performed.
