from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from exchange_ml.e4_e5_analysis import review_budget
from exchange_ml.e5_supplement_statistics import budget_analysis
from test_e5_supplement_statistics import synthetic_rows
from run_e4_e5 import read_json, write_csv
from run_e5_supplement import check_config, check_legacy_points, check_output_path, validate_result_grid


class SupplementRunnerTests(unittest.TestCase):
    def test_resumed_result_grid_rejects_duplicates_even_when_counts_match(self):
        rows = synthetic_rows()
        points, paired, curves = budget_analysis(rows, resamples=4, seed=1)
        result = {"fixed_budget": points, "paired_differences": paired, "risk_coverage": curves}
        validate_result_grid(result, rows)
        for key in result:
            altered = deepcopy(result)
            altered[key][0] = deepcopy(altered[key][1])
            with self.subTest(table=key), self.assertRaisesRegex(ValueError, "missing/duplicate"):
                validate_result_grid(altered, rows)

    def test_legacy_budget_guard_accepts_original_estimand_and_rejects_drift(self):
        rows = synthetic_rows()
        original = [{"split": "validation", **r} for r in review_budget(rows, {"retained_fractions": [.8, .9]})]
        points, _, _ = budget_analysis(rows, resamples=4, seed=1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root / "tables/review_budget.csv", original)
            check_legacy_points(points, "validation", root)
            altered = deepcopy(points)
            next(r for r in altered if r["score"] != "random" and r["risk"] == "merge_split")["retained_mean_risk"] += .1
            with self.assertRaisesRegex(ValueError, "changed"):
                check_legacy_points(altered, "validation", root)

    def test_output_guard_refuses_protected_root_ancestors_and_descendants(self):
        protected = [ROOT / "outputs/stage3_h1_v2/e4_e5", ROOT / "outputs/stage3_h1_v2/heldout"]
        check_output_path(ROOT / "outputs/stage3_h1_v2/e5_supplement_v1", protected)
        for path in (ROOT / "outputs", ROOT / "outputs/stage3_h1_v2", protected[0], protected[0] / "new", Path("/tmp")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_output_path(path, protected)

    def test_approved_protocol_budgets_and_endpoint_roles_are_explicit(self):
        config = read_json(ROOT / "configs/e5_supplement.json")
        check_config(config)
        for key, value in (("retained_fractions", [.7, .9]), ("analysis_role", "confirmatory"),
                           ("qualitative_selection", "best_new_examples")):
            altered = {**config, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_config(altered)


if __name__ == "__main__":
    unittest.main()
