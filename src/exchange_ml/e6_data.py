"""RibFrac image/instance contracts for external E6 evaluation.

The target is *fracture lesions*, not complete ribs or PENGWIN bone fragments.
The official release defines label_code=-1 as an untyped fracture, ignored
only in classification. It remains foreground in binary segmentation and in
class-agnostic instance evaluation. See https://zenodo.org/records/3893496.
No PENGWIN anatomy ranges, component pruning or label remapping are applied.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk


LABEL_CODES = {-1: "untyped_fracture", 0: "background", 1: "displaced",
               2: "nondisplaced", 3: "buckle", 4: "segmental"}
TRAIN_IDS = tuple(f"RibFrac{i}" for i in range(1, 421))
VALIDATION_IDS = tuple(f"RibFrac{i}" for i in range(421, 501))
MEMBERS = ("member_01", "member_02", "member_03")


def case_number(case_id: str) -> int:
    """Validate public IDs; zero padding and held-out test IDs are not aliases."""
    if not isinstance(case_id, str) or not case_id.startswith("RibFrac"):
        raise ValueError(f"invalid RibFrac public_id: {case_id!r}")
    suffix = case_id[7:]
    if not suffix.isdecimal() or suffix.startswith("0"):
        raise ValueError(f"invalid RibFrac public_id: {case_id!r}")
    number = int(suffix)
    if number < 1 or number > 660:
        raise ValueError(f"RibFrac public_id outside released range: {case_id}")
    return number


def read_label_codes(path: Path) -> dict[str, dict[int, int]]:
    """Read the official per-instance CSV; reject duplicate/unknown semantics."""
    result: dict[str, dict[int, int]] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["public_id", "label_id", "label_code"]:
            raise ValueError(f"unexpected RibFrac CSV schema: {path}: {reader.fieldnames}")
        for line, row in enumerate(reader, 2):
            cid = row["public_id"]
            case_number(cid)
            try:
                label_id, code = int(row["label_id"]), int(row["label_code"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"noninteger label metadata: {path}:{line}") from error
            if label_id < 0 or code not in LABEL_CODES or (label_id == 0) != (code == 0):
                raise ValueError(f"invalid label_id/label_code pair: {path}:{line}")
            labels = result.setdefault(cid, {})
            if label_id in labels:
                raise ValueError(f"duplicate label_id: {path}:{line}: {cid}/{label_id}")
            labels[label_id] = code
    if not result or any(labels.get(0) != 0 for labels in result.values()):
        raise ValueError(f"each RibFrac case needs one background CSV row: {path}")
    return result


def inventory(root: Path, *, strict: bool = True) -> dict[str, dict[str, Any]]:
    """Inventory only official train/validation pairs, without decoding volumes.

    strict=True enforces 420 training and 80 validation cases, their exact IDs,
    the official folder assignment and CSV membership. The 160 sealed test
    images are intentionally never included. strict=False is for tiny fixtures.
    """
    root = Path(root).resolve()
    specifications = (
        ("train", "Part1", "Part1", "ribfrac-train-info-1.csv", range(1, 301)),
        ("train", "Part2", "Part2", "ribfrac-train-info-2.csv", range(301, 421)),
        ("validation", "ribfrac-val-images", "ribfrac-val-labels", "ribfrac-val-info.csv", range(421, 501)),
    )
    result: dict[str, dict[str, Any]] = {}
    for split, image_dir, label_dir, metadata_name, numbers in specifications:
        metadata_path = root / metadata_name
        codes = read_label_codes(metadata_path)
        image_paths = {p.name.removesuffix("-image.nii.gz"): p
                       for p in (root / image_dir).glob("*-image.nii.gz")}
        label_paths = {p.name.removesuffix("-label.nii.gz"): p
                       for p in (root / label_dir).glob("*-label.nii.gz")}
        expected = {f"RibFrac{i}" for i in numbers}
        if image_paths.keys() != label_paths.keys() or image_paths.keys() != codes.keys():
            raise ValueError(f"RibFrac image/label/CSV membership differs in {image_dir}/{label_dir}")
        if strict and set(image_paths) != expected:
            raise ValueError(f"RibFrac official {image_dir} IDs differ: "
                             f"missing={sorted(expected - image_paths.keys())}, "
                             f"extra={sorted(image_paths.keys() - expected)}")
        if not set(image_paths).issubset(expected):
            raise ValueError(f"RibFrac split leakage or wrong folder membership: {image_dir}")
        for cid in sorted(image_paths, key=case_number):
            if cid in result:
                raise ValueError(f"duplicate RibFrac public_id: {cid}")
            if not image_paths[cid].is_file() or not label_paths[cid].is_file():
                raise FileNotFoundError(f"missing RibFrac pair: {cid}")
            result[cid] = {"case_id": cid, "split": split,
                           "image": image_paths[cid], "label": label_paths[cid],
                           "label_codes": codes[cid], "metadata": metadata_path}
    return result


def geometry(image: sitk.Image | sitk.ImageFileReader) -> dict[str, tuple]:
    """Record physical geometry; arrays use [Z,Y,X], spacing here uses [X,Y,Z]."""
    components = (image.GetNumberOfComponentsPerPixel() if isinstance(image, sitk.Image)
                  else image.GetNumberOfComponents())
    if image.GetDimension() != 3 or components != 1:
        raise ValueError("RibFrac requires scalar 3D volumes")
    result = {"size": tuple(image.GetSize()), "spacing": tuple(image.GetSpacing()),
              "origin": tuple(image.GetOrigin()), "direction": tuple(image.GetDirection())}
    if (any(n <= 0 for n in result["size"])
            or not all(np.isfinite(result[k]).all() for k in ("spacing", "origin", "direction"))
            or any(s <= 0 for s in result["spacing"])):
        raise ValueError("invalid physical image geometry")
    return result


def require_geometry(actual: dict, expected: dict) -> None:
    """Fail on grid mismatch; never silently resample labels or probabilities."""
    if set(actual) != {"size", "spacing", "origin", "direction"} or set(expected) != set(actual):
        raise ValueError("incomplete geometry record")
    if tuple(actual["size"]) != tuple(expected["size"]) or any(
            not np.allclose(actual[k], expected[k], atol=1e-6, rtol=1e-6)
            for k in ("spacing", "origin", "direction")):
        raise ValueError("RibFrac geometry mismatch")


def read_header(path: Path) -> dict[str, tuple]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return geometry(reader)


def read_volume(path: Path, expected: dict | None = None, *, probability: bool = False
                ) -> tuple[np.ndarray, dict[str, tuple]]:
    """Read a scalar native-grid volume; IDs are generic, never anatomy codes."""
    image = sitk.ReadImage(str(path))
    observed = geometry(image)
    if expected is not None:
        require_geometry(observed, expected)
    array = sitk.GetArrayFromImage(image)
    if probability:
        if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
            raise ValueError(f"probabilities must be finite in [0,1]: {path}")
        array = array.astype(np.float32, copy=False)
    else:
        if (not np.issubdtype(array.dtype, np.integer) or (array < 0).any()
                or array.max(initial=0) > np.iinfo(np.uint32).max):
            raise ValueError(f"instance IDs must be nonnegative uint32-compatible integers: {path}")
        array = array.astype(np.uint32, copy=False)
    return array, observed


def validate_labels(labels: np.ndarray, codes: dict[int, int]) -> np.ndarray:
    """Verify exact mask/CSV ID membership, retaining untyped fractures (-1)."""
    labels = np.asarray(labels)
    if (labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer)
            or labels.size == 0 or (labels < 0).any()
            or labels.max(initial=0) > np.iinfo(np.uint32).max):
        raise ValueError("RibFrac GT must be a nonempty 3D nonnegative integer map")
    if not codes or codes.get(0) != 0 or any(
            label_id < 0 or code not in LABEL_CODES or (label_id == 0) != (code == 0)
            for label_id, code in codes.items()):
        raise ValueError("invalid RibFrac CSV label semantics")
    # Background is a required metadata row even if a synthetic foreground fills
    # the whole array; foreground IDs must correspond exactly in both sources.
    present = set(int(i) for i in np.unique(labels) if i > 0)
    expected = set(codes) - {0}
    if present != expected:
        raise ValueError(f"RibFrac GT/CSV label IDs differ: unknown={sorted(present-expected)}, "
                         f"missing={sorted(expected-present)}")
    return labels


def binary_target(labels: np.ndarray, codes: dict[int, int]) -> np.ndarray:
    """Class-agnostic lesion target [Z,Y,X], uint8; -1 codes are foreground."""
    validate_labels(labels, codes)
    return (labels > 0).astype(np.uint8)


def read_case(record: dict, *, load_image: bool = True
              ) -> tuple[np.ndarray | None, np.ndarray, dict[str, tuple]]:
    """Read one aligned case; no intensity transform or label pruning is applied."""
    reference = read_header(Path(record["image"]))
    gt, observed = read_volume(Path(record["label"]), reference)
    validate_labels(gt, record["label_codes"])
    image_array = None
    if load_image:
        image = sitk.ReadImage(str(record["image"]))
        require_geometry(geometry(image), reference)
        image_array = sitk.GetArrayFromImage(image)
        if not np.isfinite(image_array).all():
            raise ValueError(f"CT image has nonfinite values: {record['image']}")
    return image_array, gt, observed
