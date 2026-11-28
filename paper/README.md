# Manuscript

Read `manuscript.pdf`; edit `manuscript.tex` and `references.bib`.

This directory contains the manuscript, its IEEE class, two LaTeX tables,
the three figure PDFs used in the text, and the build command:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
bash paper/build.sh
```

The build uses Tectonic's local cache and writes temporary files to `paper/build/`.
`pdflatex` + `bibtex` (e.g. TinyTeX) compile the same sources. The build does not
regenerate figures or tables or run experiments. To select a compiler, set
`TECTONIC` to its executable path.

## Figures

Figures are regenerated from saved artifacts with the scripts in
`scripts/paper_figures/` (Times text, IEEE widths, shared palette in `style.py`):

| Figure | Script | Inputs |
|---|---|---|
| `fig1_concept.pdf` | `fig1_concept.py` | `outputs/paper_figures_v1/039_1.npz` (PENGWIN case 039 sacrum, written by `extract_concept_case.py`; reference labels only) |
| `fig2_evidence.pdf` | `fig2_evidence.py` | `outputs/stage3_h1_v2/` (E1–E3 CSVs; curves recomputed with `aurc_check.py`, which reproduces all Table I AURCs) |
| `fig3_examples.pdf` | `fig3_examples.py` | E5 qualitative renderings and `heldout/scores.csv`; overlays recovered by `qualitative_masks.py` |

```bash
cd scripts/paper_figures
python fig1_concept.py ../../outputs/paper_figures_v1/039_1.npz ../../outputs/paper_figures_v1/fig1_concept.pdf
python fig2_evidence.py ../../outputs/stage3_h1_v2 ../../outputs/paper_figures_v1/fig2_evidence.pdf
python fig3_examples.py ../../outputs/stage3_h1_v2 ../../outputs/paper_figures_v1/fig3_examples.pdf
```

Fig. 3 overlays are recovered from the stored E5 PNG renderings because raw member
predictions live on the compute server; regenerating them from the raw prediction
files there would give the same overlays at native resolution.

## Status

The 28 November 2026 revision (title, text, figures, acknowledgment) yields six
pages: technical content ends on page 5; conclusion, acknowledgment and references
fill half of page 6. The External Validation (E6) placeholder section was removed
from the text. Author metadata and final author approval remain outstanding. No
submission or publication has been performed. The pre-revision manuscript, tables
and figures are kept in `outputs/paper_figures_v1/previous_paper_version/`.
