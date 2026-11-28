# Manuscript

Read `manuscript.pdf`; edit `manuscript.tex` and `references.bib`.

This directory contains the manuscript, its IEEE class, two LaTeX tables,
the two figure PDFs used in the text, and the build command:

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
bash paper/build.sh
```

The build uses Tectonic's local cache and writes temporary files to `paper/build/`.
It compiles the saved manuscript, tables and figures; it does not regenerate
figures or tables or run experiments. To select a compiler, set `TECTONIC` to
its executable path.

The current draft is five pages. Technical content ends on page 4; the
acknowledgment and references begin on page 4 and continue on page 5, without
a forced page break. Author metadata and final author approval remain outstanding. The PENGWIN 2026
dataset DOI was verified against its Zenodo record during the 28 November 2026
refinement. Existing result tables are synchronized with completed non-E6
experiment artifacts; External Validation (E6) remains a clean results placeholder. No submission or publication has been performed.

Evidence/review records, figure-generation scripts, template duplicates, previews
and intermediate builds were removed from this directory on 28 November 2026.
The complete previous directory is backed up, with verified file hashes, at
`/mnt/sdb/shared/dang.cpm/.cleanup_backups/Exchange-ML-paper-20261128T071328Z/paper-before.tar.gz`.
Experiment outputs remain in the project's `outputs/` directory.
