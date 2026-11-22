from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from exchange_ml.e1_scoring import SCORE_NAMES
from exchange_ml.e1_statistics import aurc_batch
from exchange_ml.structural_rank_reporting import (
    ALL_SCORES, CI_ROLE, FINAL_HEAD, PRIMARY_ESTIMAND, fold_cluster_draws, report, validate_rows,
)


def rows_fixture() -> list[dict]:
    rows = []
    for fold in range(2):
        for local in range(4):
            scan = "pelvic" if local < 2 else "femur"
            bones = ("sacrum", "left_hip", "right_hip") if scan == "pelvic" else ("femur",)
            index = fold * 4 + local
            for j, bone in enumerate(bones):
                p1 = (index + 1 + j * .1) / 10
                risk = (index + 1 + j * .2) / 4
                row = {"case_id": f"case_{index:03}", "scan_type": scan, "bone": bone,
                       "outer_fold": fold, **{s: p1 for s in SCORE_NAMES},
                       "presence": p1 / 2, "graph_degree": p1 * 2, FINAL_HEAD: p1}
                for source, scale in (("member_mean", 1.), ("stage2", .7)):
                    row.update({source + "_merge_split": risk * scale,
                                source + "_one_minus_f1": risk * scale / 3,
                                source + "_missing_volume": risk * scale / 4})
                rows.append(row)
    return rows


CONFIG = {"cv": {"outer_folds": 2}, "reporting": {"bootstrap_resamples": 8, "bootstrap_seed": 29}}


def primary(result: dict) -> dict:
    return next(r for r in result["paired_differences"] if r["contrast_role"] == "v4_primary")


class StructuralRankReportingTests(unittest.TestCase):
    def test_fold_affine_shifts_preserve_primary_but_change_pooled_diagnostic(self):
        rows = rows_fixture()
        initial = report(rows, CONFIG)
        shifted = copy.deepcopy(rows)
        for row in shifted:
            row[FINAL_HEAD] = row[FINAL_HEAD] * (1 + row["outer_fold"]) + (10 if row["outer_fold"] == 0 else 0)
        changed = report(shifted, CONFIG)
        self.assertEqual(primary(initial)["delta_aurc"], 0.)
        self.assertEqual(primary(changed)["delta_aurc"], 0.)
        self.assertEqual(primary(initial)["ci_low"], primary(changed)["ci_low"])
        self.assertEqual(primary(initial)["ci_high"], primary(changed)["ci_high"])
        pooled = lambda r: next(v for v in r["pooled_paired_differences"] if v["stratum"] == "pooled" and v["baseline"] == "partition_v1" and v["risk_source"] == "member_mean" and v["risk"] == "merge_split")
        self.assertEqual(pooled(initial)["delta_aurc"], 0.)
        self.assertGreater(pooled(changed)["delta_aurc"], .1)
        self.assertEqual(pooled(changed)["contrast_role"], "diagnostic_only")
        self.assertEqual(changed["summary"]["primary_estimand"], PRIMARY_ESTIMAND)
        self.assertEqual(changed["summary"]["ci_role"], CI_ROLE)
        self.assertFalse(changed["summary"]["fold_standard_errors_used"])
        json.dumps(changed, allow_nan=False)

    def test_exact_ties_monotone_control_and_deterministic_row_order(self):
        rows = rows_fixture()
        for row in rows:
            row[FINAL_HEAD] = 1 + 2 * row["partition_v1"]
        result = report(rows, CONFIG)
        self.assertAlmostEqual(primary(result)["delta_aurc"], 0.)
        self.assertEqual({r["score"] for r in result["comparison"]}, set(ALL_SCORES))
        self.assertEqual(result, report(rows[::-1], CONFIG))
        # Repeated feature values retain the frozen evaluator's exact tie convention.
        for row in rows:
            row["partition_v1"] = round(row["partition_v1"], 1)
            row[FINAL_HEAD] = row["partition_v1"]
        self.assertEqual(primary(report(rows, CONFIG))["delta_aurc"], 0.)

    def test_cluster_draws_preserve_fold_scan_and_whole_duplicate_cts(self):
        rows = validate_rows(rows_fixture(), 2)
        draws = fold_cluster_draws(rows, 32, 29)
        duplicate_seen = False
        for fold, take in draws.items():
            self.assertEqual(take.shape, (32, 8))
            for draw in take:
                self.assertTrue(all(rows[i]["outer_fold"] == fold for i in draw))
                for scan in ("pelvic", "femur"):
                    expected_units = 6 if scan == "pelvic" else 2
                    self.assertEqual(sum(rows[i]["scan_type"] == scan for i in draw), expected_units)
                for cid in {rows[i]["case_id"] for i in draw}:
                    indices = [i for i, row in enumerate(rows) if row["case_id"] == cid]
                    multiplicities = [int(np.count_nonzero(draw == i)) for i in indices]
                    self.assertEqual(len(set(multiplicities)), 1)
                    duplicate_seen |= multiplicities[0] > 1
        self.assertTrue(duplicate_seen)

    def test_paired_ci_matches_mean_of_hand_reused_fold_draws(self):
        rows = validate_rows(rows_fixture(), 2)
        for row in rows:
            row[FINAL_HEAD] = row["partition_v1"] if row["outer_fold"] else -row["partition_v1"]
        result = report(rows, CONFIG)
        x = np.array([r[FINAL_HEAD] for r in rows])
        p1 = np.array([r["partition_v1"] for r in rows])
        y = np.array([r["member_mean_merge_split"] for r in rows])
        differences = []
        for take in fold_cluster_draws(rows, 8, 29).values():
            differences.append(aurc_batch(x[take], y[take][..., None])[:, 0]
                               - aurc_batch(p1[take], y[take][..., None])[:, 0])
        expected = np.mean(differences, axis=0)
        contrast = primary(result)
        np.testing.assert_allclose([contrast["ci_low"], contrast["ci_high"]], np.quantile(expected, [.025, .975]))

    def test_input_membership_folds_and_ranges_are_checked(self):
        rows = rows_fixture()
        for broken in ([*rows, rows[0]], rows[1:], [r for r in rows if r["outer_fold"] == 0]):
            with self.assertRaises(ValueError):
                report(broken, CONFIG)
        for name, value in ((FINAL_HEAD, float("nan")), ("stage2_missing_volume", 2.), ("graph_degree", -.1), ("outer_fold", True)):
            broken = copy.deepcopy(rows); broken[0][name] = value
            with self.assertRaises(ValueError):
                report(broken, CONFIG)
        broken = copy.deepcopy(rows); broken[0]["outer_fold"] = 1
        with self.assertRaisesRegex(ValueError, "all bones"):
            report(broken, CONFIG)

    def test_constant_risks_and_insufficient_fold_cases_are_explicit(self):
        rows = rows_fixture()
        for row in rows:
            for source in ("member_mean", "stage2"):
                for risk in ("merge_split", "one_minus_f1", "missing_volume"):
                    row[source + "_" + risk] = 0.
        result = report(rows, CONFIG)
        self.assertTrue(all(r["status"] == "constant_risk_uninformative" for r in result["comparison"]))
        self.assertEqual(primary(result)["delta_aurc"], 0.)
        # One pelvic CT still has three bones, but supplies only one independent case.
        tiny = [r for r in rows if r["case_id"] in ("case_000", "case_004")]
        result = report(tiny, CONFIG)
        self.assertTrue(all(r["ci_low"] is None and r["ci_status"] == "insufficient_cases_in_outer_fold" for r in result["comparison"]))
        femurs = [r for r in rows if r["case_id"] in ("case_002", "case_006")]
        result = report(femurs, CONFIG)
        self.assertTrue(all(r["aurc"] is None and r["status"] == "insufficient_units_in_outer_fold" for r in result["comparison"]))
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
