#!/usr/bin/env python3
"""Render one predeclared E5 example to /tmp; not a full E5 run.

Selection uses the frozen rules on existing score CSVs. Only the selected
case's volumes are read. No cutoffs are relaxed if the category has no case.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os

from run_e4_e5 import (ROOT, joined_rows, load_inputs, read_json, require_hash, save_json, sha256, state)
from exchange_ml.e4_e5_analysis import CATEGORIES, qualitative_candidates
from exchange_ml.e5_visualization import plot_qualitative, prepare_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=CATEGORIES, default=CATEGORIES[0])
    parser.add_argument("--output", type=Path, default=Path("/tmp/exchange-e4-e5-real-visual-smoke"))
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/tmp")) or output == Path("/tmp"):
        raise ValueError("this representative smoke writes only to a directory under /tmp")
    os.chdir(ROOT)
    config_path = ROOT / "configs/e4_e5.json"
    config = read_json(config_path)
    cases, data, specs, inputs = load_inputs(config_path, config)
    candidates, selections = qualitative_candidates(
        joined_rows(data["heldout"], {r["case_id"] for r in cases["heldout"]}), data["heldout"]["model_risks"], config)
    selected = next(r for r in selections if r["category"] == args.category)
    examples, source_hashes = [], {}
    if selected["status"] == "selected":
        spec = specs["heldout", selected["case_id"]]
        before = {k: state(p) for k, p in spec["paths"].items()}
        for key, digest in spec["hashes"].items():
            require_hash(spec["paths"][key], digest)
        source_hashes = {**spec["hashes"], "image": sha256(spec["paths"]["image"])}
        examples.append(prepare_example(selected, spec["paths"], {**config, **{k: inputs["protocol"][k]
                                        for k in ("prune_mm3", "connectivity", "iou_threshold")}}))
        if before != {k: state(p) for k, p in spec["paths"].items()}:
            raise ValueError("source changed during representative visual smoke")
    output.mkdir(parents=True, exist_ok=True)
    plot_qualitative(output / "qualitative", examples, [selected], candidates, data["heldout"]["model_risks"])
    save_json(output / "evidence.json", {"role": "one_real_example_smoke_not_full_E5", "selected": selected,
        "coordinates": examples[0]["coordinates"] if examples else [],
        "mapping": examples[0]["mapping"] if examples else [], "source_sha256": source_hashes,
        "config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__)),
        "png_sha256": sha256(output / "qualitative.png"), "pdf_sha256": sha256(output / "qualitative.pdf")})
    print(f"E5 REPRESENTATIVE VISUAL SMOKE COMPLETE: {output}; selection={selected}", flush=True)


if __name__ == "__main__":
    main()
