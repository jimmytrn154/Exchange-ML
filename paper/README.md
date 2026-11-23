# Manuscript

`manuscript.pdf` is the complete anonymous English scientific draft in the supplied IEEE conference template. Technical content occupies pages1–4; acknowledgment and references are on page5. Author/affiliation metadata remain anonymous pending author input. No institutional approval, funding or clinical benefit is invented.

The paper reports completed PENGWIN experiments using terms defined for readers without project history. It focuses on component separation, normalization, held-out ranking, strong controls, and endpoint dependence. Exploratory fusion-development history is omitted; the stronger partition-only control, shared-omission limit, and uncertain review utility remain explicit. External RibFrac training and the new triage-policy experiment supply no completed outcomes. The source-versus-final-version gap in one EMBC literature comparator is documented in `review/embc_recent_fulltext.md`.

Build from saved completed experiment tables, then compile LaTeX:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/exchange-paper-mpl \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B paper/build_paper.py --compile
```

This performs source-manifest/output hash checks and regenerates the tables, figure, Markdown view and evidence receipt. It does not train, infer or rerun experimental bootstraps. Tectonic uses the locally populated cache (`--only-cached`); a new environment may first require an online Tectonic run to obtain TeX packages/fonts. The unmodified supplied class is retained in `template/IEEEtran.cls` and copied as `IEEEtran.cls`.

- `manuscript.tex`: canonical text.
- `manuscript.md`: regenerated readable view of the same text, tables and bibliography.
- `references.bib`:11 cited records; primary source verification is retained in `review/`.
- `evidence/manuscript_sources.json`: source/output hashes and figure/table provenance.
- `review/`: literature sampling, integrity and review artifacts; these do not alter the manuscript's experiment claims.

The local draft is prepared for author review, not automatic submission. Review reports and remaining metadata decisions must accompany the final handoff.

The first-reader revision and targeted factual/layout checks are recorded in `review/first_reader_revision.md` and `.json`; the previous draft is preserved under `review/archive/before_first_reader_revision_20261123T124414Z/`. This check is not a completed formal integrity or peer-review gate.
