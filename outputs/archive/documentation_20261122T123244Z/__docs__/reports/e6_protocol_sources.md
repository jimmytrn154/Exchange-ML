# E6: verified target and data contract

**Verified 21 November 2026.** This is an implementation/protocol record, not an E6 experimental result.

## Primary sources

- [Official RibFrac validation release, Zenodo DOI 10.5281/zenodo.3893496](https://zenodo.org/records/3893496): 80 CT/annotation pairs, and the meaning of `public_id`, `label_id`, `label_code`. Codes 1–4 identify displaced, nondisplaced, buckle and segmental fractures; 0 identifies background. Code −1 denotes a fracture with uncertain type and is excluded from **classification**, not class-agnostic lesion segmentation. E6 therefore retains every nonzero instance ID, including IDs whose CSV code is −1.
- [Organizer evaluation repository](https://github.com/HINTLab/RibFrac-Challenge): predictions contain integer fracture-region IDs and optional detection/classification metadata. Its [evaluation implementation](https://github.com/HINTLab/RibFrac-Challenge/blob/master/ribfrac/evaluation.py) implements challenge detection FROC, with default IoU 0.2, and separate segmentation/classification metrics. The E6 structural-risk analysis is an internal score-transfer evaluation; it must not be called the official RibFrac challenge score.

The release concerns **fracture lesions**. It does not supply complete rib/bone-fragment instances corresponding to PENGWIN anatomy IDs. The cross-dataset result, when run, will test disagreement-score transfer to this related instance task. It cannot establish transfer of the PENGWIN checkpoints or equivalent anatomy semantics.

## Local observations and interpretation

The metadata-only strict inventory confirms exact filenames and CSV membership for 420 training pairs (IDs 1–420) and 80 official validation pairs (421–500), without including test images. All 80 validation IDs have matching image/label filenames. This does not by itself prove complete voxel-level integrity across the cohort.

The validation CSV has 515 rows: 80 background records and 435 lesion records, of which 243 have code −1. Twenty validation subjects have only a background record. `RibFrac421` has four nonzero instance IDs and the mask/CSV IDs match; its smallest lesion has 798 voxels, approximately 770 mm³. Applying the PENGWIN 1,000 mm³ threshold would delete a valid lesion. `RibFrac435` and `RibFrac491` are background-only in the representative masks read. `RibFrac491` has reversed X/Y direction; image and label share that grid. Read geometry from each file and preserve physical alignment.

These checks read three representative GT masks and image headers. They are not inference and do not substitute for full preprocessing validation. No fracture type is inferred from the image or reassigned. Nonzero GT labels stay intact for evaluation; binary training foreground is their union. No ignored voxel region is introduced for code −1.

## Data API and checks

`src/exchange_ml/e6_data.py` provides strict inventory, CSV interpretation, native-grid volume reading, exact GT/CSV foreground-ID validation and a binary lesion target. IDs are uint32-compatible generic integers rather than anatomy ranges. Probability volumes must be scalar, finite and within [0,1]. Shape, spacing, origin and direction must agree; mismatches are errors, not implicit resampling.

`tests/test_e6_data.py` covers missing pairs, split leakage, duplicate/unknown metadata, large generic IDs, untyped positives, negative subjects, scalar grids, probability range and geometry/orientation round trips. Six tests pass. The first round caught a SimpleITK reader component-count method mismatch; the method dispatch was corrected and all tests rerun successfully.

Agent-typed checks (CPU only):

```bash
cd /mnt/sdb/shared/dang.cpm/Exchange-ML
PYTHONPATH=src:.:tests:scripts OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dang.cpm/miniconda3/envs/exchange-stage2/bin/python -B -m unittest tests/test_e6_data.py -v
```

The representative inventory/read check called `inventory(Path('dataset/RibFrac'))` and `read_case(record, load_image=False)` for `RibFrac421`, `RibFrac435`, `RibFrac491`. All writes from tests are temporary fixtures. User-typed full-run commands are in the separate E6 implementation report; no full training, preprocessing or inference was run to prepare this record.
