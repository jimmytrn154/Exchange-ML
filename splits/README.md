# Frozen PENGWIN split

- `pengwin_stage0_split.json` is the acquisition-safe static split and initial-set manifest.
- `pengwin_stage0_gt_audit.json` contains ground-truth-derived fragment counts, foreground volumes, label IDs, and geometry metadata.

Acquisition code must never load `pengwin_stage0_gt_audit.json`. Ground truth was used only to construct and audit the static split. Test IDs must not be used for tuning, cleanup selection, threshold selection, or acquisition.

Validate these artifacts with:

```bash
python scripts/validate_stage0_artifacts.py
```
