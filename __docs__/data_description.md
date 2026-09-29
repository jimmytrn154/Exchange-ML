# Dataset Description

## 1. Scope and role in the study

This workspace contains two 3D CT fracture datasets:

1. **PENGWIN 2026 Task 1/2**, the primary dataset for pelvic and femur fracture-fragment instance segmentation and click-guided interactive segmentation.
2. **RibFrac**, the secondary dataset for cross-anatomy validation on rib-fracture instances.

This description records the data that are physically present under `dataset/`. It does not treat hidden challenge test cases as labeled training data, and it distinguishes metadata that is present from image/label pairs that are not present in this checkout.

## 2. PENGWIN 2026

### 2.1 Dataset contents

The local PENGWIN directory is `dataset/PENGWIN26_task1_2/`. It contains **340 labeled training subjects**:

| Directory | Subject range | Anatomy | Subjects | Files per subject |
|---|---:|---|---:|---|
| `PENGWIN26_task1_2_train_part1` | 001-085 | Pelvic | 85 | `image.mha`, `label.mha` |
| `PENGWIN26_task1_2_train_part2` | 086-120 and 151-200 | Pelvic | 85 | `image.mha`, `label.mha` |
| `PENGWIN26_task1_2_train_part3` | 251-335 | Femur | 85 | `image.mha`, `label.mha` |
| `PENGWIN26_task1_2_train_part4` | 336-420 | Femur | 85 | `image.mha`, `label.mha` |

The four partitions therefore contain 170 pelvic and 170 femur cases. There are 340 image volumes and 340 matching label volumes. The public README describes an additional 160-subject challenge test set, but those hidden test volumes are not present in this local directory and must not be used as a labeled pool.

### 2.2 Image volumes

Each subject has one CT volume named `image.mha`. The files use the MetaImage (`.mha`) container with a local binary payload. The inspected headers describe three-dimensional images with:

- anatomical orientation `RAI`;
- `ElementDataFile = LOCAL`;
- little-endian binary storage;
- image scalar type `MET_INT` / 32-bit integer in the inspected cases;
- variable dimensions, rather than one fixed matrix size;
- in the inspected cases, in-plane spacing near 0.66-0.96 mm and slice spacing of 0.6, 0.8, or 1.0 mm.

One representative case is `512 x 512 x 401` voxels at `0.78125 x 0.78125 x 0.8 mm`. Dimensions and spacing must be read from each header; preprocessing must not hard-code this example geometry. The headers also include physical offsets and direction information, so voxel coordinates and physical coordinates must not be silently conflated.

### 2.3 Instance labels

Each subject has one instance mask named `label.mha`, spatially paired with its image. The label header uses compressed local binary data and `MET_SHORT` / 16-bit signed integer storage in the inspected cases. Label value `0` is background. Each non-zero integer identifies one fracture fragment in that subject.

The label IDs are **instance IDs, not globally meaningful semantic classes**. The numeric ranges encode the associated anatomical group:

| Label values | Anatomical group | Cases |
|---:|---|---|
| 0 | Background | All |
| 1-50 | Sacrum | Pelvic |
| 51-100 | Left hip | Pelvic |
| 101-150 | Right hip | Pelvic |
| 151-200 | Femur | Femur |

The non-zero values are sparse within a case. For example, values `{1, 2, 51, 52, 101}` represent five fragments distributed across the sacrum, left hip, and right hip. The number of distinct non-zero label values is the case's encoded fragment count. Representative inspected cases contained between 2 and 15 fragments; the challenge documentation reports a broader training range of 2-30, so the full distribution should be computed from all labels before final statistical reporting.

The label values should be preserved when loading masks. A connected-component operation may be useful for auditing, but it must not be used to replace the dataset's instance IDs without an explicit connectivity rule. Fragment identity is the central structural property for the planned Instance-Partition Disagreement experiments.

### 2.4 Task 2 click annotations

The directory `PENGWIN26_task2_train_clicks/` contains four click sets for every one of the 340 training subjects, for **1,360 JSON files total**:

- `boundary_internal_margin`
- `center_of_mass`
- `euclidean_distance_transform`
- `uniformly_sampled`

Each strategy directory contains one subject directory and a file named `peripelvic-fragment-clicks.json`. A click file has:

```json
{
	"name": "Center of Mass Points of Interest",
	"type": "Multiple Points",
	"points": [
		{
			"name": "Sacrum Center of Mass Point 1",
			"point": [268, 320, 262]
		}
	],
	"version": {"major": 1, "minor": 0}
}
```

The `point` array is an integer voxel index in `[x, y, z]` order, not a physical-space coordinate. The number of points is intended to equal the number of fracture fragments. Point names encode the anatomical group, click strategy, and a one-based subject-local fragment index. Click strategies differ in how the point is selected inside each fragment: near an internal boundary margin, at the center of mass, at a location maximally distant from boundaries, or by uniform sampling.

Before using clicks for Task 2, code must verify coordinate bounds against that subject's image size and verify the point-count/fragment-count relationship. A multi-volume SimpleITK audit was attempted during documentation inspection but terminated with a segmentation fault; this consistency audit remains pending and must be rerun with the project's eventual data-loading pipeline before experimental use.

## 3. RibFrac

### 3.1 Dataset contents

The local RibFrac directory is `dataset/RibFrac/`. It contains NIfTI files (`.nii.gz`) and CSV label metadata:

| Local path | Intended split | Subjects/files present |
|---|---|---:|
| `Part1/` | Training part 1 | 300 image-label pairs, IDs `RibFrac1`-`RibFrac300` |
| `Part2/` | Training part 2 | 120 image-label pairs, IDs `RibFrac301`-`RibFrac420` |
| `ribfrac-val-labels/` | Validation labels | 80 label files, IDs `RibFrac421`-`RibFrac500` |
| `ribfrac-test-images/` | Test images | 160 image files |

The local checkout does **not** contain a `ribfrac-val-images/` directory, so the 80 validation labels are not currently paired with local validation images. It also contains no test labels. The CSV files are:

- `ribfrac-train-info-1.csv`: 300 training subjects;
- `ribfrac-train-info-2.csv`: 120 training subjects;
- `ribfrac-val-info.csv`: 80 validation subjects;
- `urls.txt`: source download URLs and provenance references.

### 3.2 NIfTI image and label files

Training cases use paired names such as `RibFrac1-image.nii.gz` and `RibFrac1-label.nii.gz`. An inspected pair has dimensions `512 x 512 x 333` and spacing approximately `0.82617 x 0.82617 x 1.25 mm`. Geometry may vary by case and must be read from each NIfTI header.

The inspected image uses NIfTI datatype code 8 (32-bit signed integer) and the label uses datatype code 2 (8-bit unsigned integer). These observations are representative, not a substitute for checking every file. RibFrac label volumes and CSV metadata must be interpreted using the official RibFrac label convention; do not assume that its label codes are interchangeable with PENGWIN's bone-specific ID ranges.

### 3.3 CSV metadata and instances

The CSV schema is:

```text
public_id,label_id,label_code
```

Each row associates a subject with an encoded label ID and a label code. `label_id=0` is the background row. Non-zero rows describe fracture instances or instance-associated classes according to the RibFrac convention. The observed `label_code` values are `-1`, `0`, `1`, `2`, `3`, and `4`; their meaning must be taken from the RibFrac task documentation rather than inferred from PENGWIN.

The training CSVs contain 4,200 rows in total and describe 420 subjects. The validation CSV contains 515 rows for 80 subjects; 60 of those subjects have at least one non-background row in the supplied metadata, so the presence of a validation subject does not by itself imply a non-empty fracture mask.

## 4. Use in the planned study

- **PENGWIN Task 1** is the primary whole-volume instance-segmentation and active-learning dataset.
- **PENGWIN Task 2 clicks** are auxiliary interactive prompts and should remain separate from Task 1 acquisition experiments unless the experiment explicitly studies interactive segmentation.
- **RibFrac** is the secondary cross-task dataset for testing whether structural uncertainty transfers from pelvic/femur fragments to rib-fracture instances.
- Hidden or unlabeled challenge test data are evaluation-only assets, not training or simulated active-learning data.
- Any split, preprocessing, fragment-count distribution, or metric implementation used in a paper must be recorded as an auditable stage output.

## 5. Confirmed inventory

The documentation inspection confirmed:

- 340 PENGWIN image files and 340 PENGWIN label files;
- 1,360 PENGWIN click JSON files across four strategies;
- 300 + 120 RibFrac training image-label pairs;
- 80 RibFrac validation label files without local validation images;
- 160 RibFrac test images without local test labels;
- approximately 61 GB for PENGWIN and 138 GB for RibFrac in this checkout.

These storage measurements are environment-specific and are included only as a local inventory aid.
