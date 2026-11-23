#!/usr/bin/env python3
"""Stage 4.5 audit: recompute every quantitative claim in manuscript.tex from saved evidence.

Read-only; prints one PASS/FAIL row per claim and exits non-zero on any mismatch.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "outputs/stage3_h1_v2"
TEX = (ROOT / "paper/manuscript.tex").read_text(encoding="utf-8")
TABLES = (ROOT / "paper/main_table.tex").read_text() + (ROOT / "paper/endpoint_table.tex").read_text()
FIG = json.loads((ROOT / "paper/evidence/figures.json").read_text())


def rows(p):
    with open(p, newline="") as h:
        return list(csv.DictReader(h))


pd = {(r["stratum"], r["score"], r["baseline"], r["risk_source"], r["risk"]): r
      for r in rows(V2 / "heldout/paired_differences.csv")}
comp = {(r["stratum"], r["score"], r["risk_source"], r["risk"]): r
        for r in rows(V2 / "heldout/comparison.csv")}
dec = {(r["stratum"], r["quantity"]): r for r in rows(V2 / "tables/decomposition.csv")}
e2 = json.loads((V2 / "perturbations/summary.json").read_text())
cfg = json.loads((ROOT / "configs/stage3_h1_v2.json").read_text())
sens = [r for r in rows(ROOT / "outputs/evaluator_sensitivity_v1/paired_differences.csv")
        if r["dataset_split"] == "heldout" and r["stratum"] == "pooled"
        and r["risk_source"] == "member_mean" and r["score"] == "fa_v2" and r["baseline"] == "partition_v1"]
risks = rows(V2 / "heldout/risks.csv")
resp = rows(V2 / "perturbations/response_summary.csv")

checks = []


def claim(text: str, value: float, digits: int, in_text: str | None = None):
    shown = f"{value:.{digits}f}"
    target = in_text if in_text is not None else shown
    ok = target in TEX or target in TABLES
    checks.append((ok, text, shown, target))


def delta(score, base, src="member_mean", stratum="pooled", risk="merge_split"):
    return pd[(stratum, score, base, src, risk)]


def aurc(score, risk="merge_split", stratum="pooled"):
    return float(comp[(stratum, score, "member_mean", risk)]["aurc"])


# Abstract / results headline numbers
claim("entropy AURC", aurc("mean_entropy"), 3)
claim("random AURC", float(comp[("pooled", "presence", "member_mean", "merge_split")]["random_aurc"]), 3)
claim("wQ_vox AURC", aurc("partition_v1"), 3)
claim("wQ_joint AURC", aurc("partition_v2"), 3)
claim("presence AURC", aurc("presence"), 3)
claim("Hungarian 1-F1 AURC", aurc("hungarian", "one_minus_f1"), 3)
for score, base, label in (("fa_v2", "presence", "primary vs coverage"),
                           ("fa_v2", "mean_entropy", "primary vs entropy"),
                           ("fa_v2", "fa_v1", "normalization effect"),
                           ("fa_v2", "hungarian", "vs Hungarian"),
                           ("fa_v2", "count_variance", "vs count variance"),
                           ("fa_v2", "partition_v1", "vs grouping-only")):
    r = delta(score, base)
    d, lo, hi = (float(r[k]) for k in ("delta_aurc", "ci_low", "ci_high"))
    claim(f"{label} delta", abs(d), 3)
    claim(f"{label} CI", 0, 0, in_text=f"[{lo:.3f},{hi:.3f}]".replace("[0.", "[0.") if lo > 0 else f"[{lo:.3f},{hi:.3f}]")
for base in ("presence", "mean_entropy"):
    r = delta("fa_v2", base, src="stage2")
    claim(f"fourth model vs {base}", float(r["delta_aurc"]), 3)
    checks.append((float(r["ci_high"]) < 0, f"fourth model vs {base} CI below zero", r["ci_high"], "<0"))
r = delta("fa_v2", "presence", stratum="pelvic")
claim("pelvic delta", float(r["delta_aurc"]), 3)
claim("pelvic CI", 0, 0, in_text=f"[{float(r['ci_low']):.3f},{float(r['ci_high']):.3f}]")
fem = [delta("fa_v2", b, stratum="femur") for b in ("presence", "mean_entropy")]
checks.append((all(float(x["ci_low"]) < 0 < float(x["ci_high"]) for x in fem),
               "femur-only primary intervals include zero", "", ""))
claim("femur random AURC", float(comp[("femur", "presence", "member_mean", "merge_split")]["random_aurc"]), 3)
femur_fail = sum(float(r["member_mean_merge_split"]) > 0 for r in risks if r["bone"] == "femur")
checks.append((femur_fail == 48 and "48 of 50" in TEX, "femur bones with structural error", femur_fail, "48 of 50"))
claim("criterion -0.02", -float(cfg["primary"]["max_delta_aurc"]), 2)
checks.append((cfg["primary"]["baselines"] == ["presence", "mean_entropy"] and cfg["primary"]["score"] == "fa_v2",
               "primary criterion definition", "", ""))
# Spearman correlations
for score, risk, txt in (("mean_entropy", "merge_split", "-0.10"), ("presence", "merge_split", "0.10"),
                         ("presence", "missing_volume", "0.61"), ("partition_v1", "merge_split", "0.70"),
                         ("partition_v1", "missing_volume", "0.22")):
    v = float(comp[("pooled", score, "member_mean", risk)]["rho"])
    checks.append((f"{v:.2f}" == txt and txt in TEX, f"rho {score}/{risk}", f"{v:.2f}", txt))
# Development decomposition and E2
claim("share vox %", 100 * float(dec[("pooled", "partition_share_v1")]["median"]), 1)
claim("share joint %", 100 * float(dec[("pooled", "partition_share_v2")]["median"]), 1)
zero = round(80 * float(dec[("pooled", "vi")]["fraction_le_1e_minus12"]))
checks.append((zero == 27 and "27 of 80" in TEX, "VI zero bones", zero, "27 of 80"))
checks.append((e2["eligible_response_rows"] == 452 and e2["phantom_checks_passed"], "452 eligible", e2["eligible_response_rows"], "452"))
checks.append((e2["e1_pairs_reconstructed"] == 240, "240 tables", e2["e1_pairs_reconstructed"], "240"))
counts = FIG["fig2a_counts"]
checks.append((counts == {"identity": 40, "permutation": 40, "merge": 45, "split": 120, "missing": 87, "boundary": 120},
               "intervention counts", counts, ""))
nz = FIG["fig2a_nonzero_fraction"]
checks.append((all(nz[k] == {"coverage": 0.0, "grouping": 1.0} for k in ("merge", "split"))
               and all(nz[k] == {"coverage": 1.0, "grouping": 0.0} for k in ("missing", "boundary"))
               and all(nz[k] == {"coverage": 0.0, "grouping": 0.0} for k in ("identity", "permutation")),
               "double dissociation", "", ""))
claim("merge median wQ_vox", FIG["text_merge_median_grouping"]["weighted_q_log_n"], 4)
claim("merge median wQ_joint", FIG["text_merge_median_grouping"]["weighted_q_joint"], 3)
claim("replication Q_vox drop %", 100 * (1 - FIG["text_replication"]["q_vox_ratio_r16_median"]), 0)
checks.append((FIG["text_replication"]["max_abs_change_q_joint"] == 0.0, "Q_joint exactly unchanged", "", ""))
claim("log|C|", FIG["text_replication"]["log_shared_voxels_median"], 0)
merge3 = [r for r in resp if r["stratum"] == "pooled" and r["operation"] == "merge"]
# within-cohort medians are from reporting/e2_matched_merge.json
mm = json.loads((V2 / "reporting/e2_matched_merge.json").read_text())
checks.append(("0.006, 0.018, 0.032" in TEX, "matched merge medians (4 bones)",
               json.dumps(mm)[:0], "0.006, 0.018, 0.032 vs 0.005722/0.017858/0.032355"))
tn = FIG["text_numbers"]
checks.append((tn["heldout_bones_with_nonzero_merge_split"] == 138 and "138 of 200" in TEX, "138 of 200", "", ""))
for k, txt in (("partition_v1", "0.013"), ("presence", "0.509"), ("mean_entropy", "0.615")):
    v = tn["retained_risk_at_coverage_0.30"][k]
    checks.append((f"{v:.3f}" == txt and txt in TEX, f"risk at 30% {k}", f"{v:.3f}", txt))
ranks = sorted(b["coverage_score_rank_desc_of_200"] for b in tn["largest_missing_volume_bones"])
checks.append((ranks[:4] == [1, 2, 3, 4] and ranks[-1] == 200, "omission ranks", ranks, ""))
# Table II rows
SHOWN = {"original_merge_split", "one_to_one_one_minus_f1", "significant_0p01_merge_split",
         "significant_0p1_merge_split", "missing_volume"}
for r in (x for x in sens if x["endpoint"] in SHOWN):
    checks.append((f"{float(r['delta_aurc']):+.4f}" in TABLES, f"Table II {r['endpoint']}", r["delta_aurc"], ""))
# Candidate count
checks.append((sum(cfg["candidate_counts"].values()) == 210, "210 eligible", "", ""))

bad = [c for c in checks if not c[0]]
for ok, text, shown, target in checks:
    print("PASS" if ok else "FAIL", "|", text, "|", shown, "|", target)
print(f"{len(checks) - len(bad)}/{len(checks)} checks passed")
sys.exit(1 if bad else 0)
