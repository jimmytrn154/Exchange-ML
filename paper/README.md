# Manuscript

Read `manuscript.pdf`; edit `manuscript.tex`. The English draft uses the supplied IEEE conference template and anonymous author metadata.

**Central message.** Ensemble disagreement between instance segmentations decomposes into a coverage term (foreground Jaccard disagreement) and a grouping term (variation of information on the shared foreground). The two terms detect different failures. Voxel entropy and foreground disagreement rank merge/split failures near chance, whereas grouping disagreement ranks them well. Normalizing VI by log voxel count suppresses grouping inside a combined score; joint-entropy normalization prevents this.

**Contributions and their evidence:**
1. Decomposition with controlled double dissociation and the normalization analysis (E1, E2).
2. A 12-score held-out benchmark on 200 bones from 100 CTs (E3). The recorded primary criterion, D_joint vs coverage/entropy, is met. That grouping-only beats D_joint is reported as a secondary result.
3. Failure-definition dependence (evaluator sensitivity) and the unanimous-omission blind spot.

## Figures

- **Fig. 1** `figures/fig1_concept.*`: concept figure.
  - Panel a: the four real CT/prediction panels, cropped from the provenance-tracked `figures/overview.png` (case 004, left hip; see `evidence/overview.json`).
  - Panels b and c: schematic.
- **Fig. 2** `figures/fig2_evidence.*`: evidence figure, built only from saved CSVs.
  - Panel a: controlled interventions.
  - Panel b: development grouping share.
  - Panel c: held-out risk–coverage curves. These are recomputed from saved scores and must reproduce the saved curves and AURCs within 1e-9.
  - Panel d: Spearman ρ for the same five scores shown in panel c; all 12 controls remain in Table I.

Both figures use vector text, paths and schematics, with the real CT overlays retained as raster data. PNG previews are exported at 450 dpi. Panel headings, score colors and typography are consistent across figures; the original full-width dimensions are preserved for the IEEE layout. The figure refinement and evidence checks are recorded in `review/figure_refinement.json`.

`make_figures.py` draws both figures. It also writes every derived number quoted in the text to `evidence/figures.json`.

## Build

On the compute server (full verification, including the CT/prediction hashes and experiment caches):

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B \
  paper/build_paper.py --compile
```

On a machine without the server-only volumes:
- Add `--offline` to skip hashing those volumes and caches. The rendered overview asset and every figure/table source are still hash-checked.
- Without Tectonic, compile with `pdflatex` + `bibtex` in `build/`.

None of these commands trains models, runs inference, fits scores, or reruns bootstraps.

To audit every quantitative statement in the text against saved evidence:

```bash
python -B paper/review/check_manuscript_numbers.py   # must print N/N checks passed
```

## Files

- `manuscript.tex`, `references.bib`: canonical text and 18 cited records (EMBC limit 20).
- `manuscript.md`, `main_table.tex`, `endpoint_table.tex`: generated views and tables. Edit their sources instead.
- `build_paper.py`, `make_figures.py`, `make_overview.py`: evidence checks and reproducible reporting.
- `conference-latex-template.zip`, `IEEEtran.cls`: the supplied template archive and its unmodified class.
- `evidence/`: machine-readable figure/table provenance.
- `review/`: published-paper source ledgers, integrity reports (`integrity_stage2_5_overhaul.md`) and the number audit.

## Layout and scope

- **Page limit:** technical content must fit on pages 1–4, with acknowledgment and references on page 5. Margins and font sizes are not reduced to achieve this.
- **Excluded results:** external RibFrac results are not available, and the exploratory triage-policy follow-up is not presented as a contribution.

Retired drafts and the earlier figure are preserved under `outputs/archive/paper_maintenance_20261123T131726Z/`.

This is an author-review draft. Author affiliations, confirmation of the PENGWIN 2026 dataset DOI, and final author approval remain outstanding. No submission or publication has been performed.
