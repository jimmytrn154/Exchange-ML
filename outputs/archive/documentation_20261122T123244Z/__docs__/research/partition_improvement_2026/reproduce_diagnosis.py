"""Read-only validation diagnosis and tiny algebra checks; never fit a model."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from exchange_ml.e1_scoring import pair_components
from exchange_ml.instances import instance_metrics


def main() -> None:
    base = ROOT / "outputs/stage3_h1_v2"
    manifest = json.loads((base / "e1_manifest.json").read_text())
    tables = {}
    for name in ("scores", "pair_components", "risks", "comparison"):
        relative = f"validation/{name}.csv"
        path = base / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["output_sha256"][relative]
        with path.open(newline="") as handle:
            tables[name] = list(csv.DictReader(handle))
    key = lambda row: (row["case_id"], row["bone"])
    risks = {key(row): row for row in tables["risks"]}
    scores = tables["scores"]
    assert len(scores) == len(risks) == 80 and {key(r) for r in scores} == risks.keys()
    names = ("presence", "partition_v1", "partition_v2", "fa_v2")
    for group in ("pooled", "sacrum", "left_hip", "right_hip", "femur"):
        selected = [r for r in scores if group == "pooled" or r["bone"] == group]
        y = np.array([float(risks[key(r)]["member_mean_merge_split"]) for r in selected])
        omission = np.array([float(risks[key(r)]["member_mean_missing_volume"]) for r in selected])
        values = {n: np.array([float(r[n]) for r in selected]) for n in names}
        print(group, "n_bones=", len(selected), "rho_M+S=",
              {n: round(float(spearmanr(x, y).statistic), 6) for n, x in values.items()})
        if group != "pooled":
            continue
        print("rho_missing_volume=", {n: round(float(spearmanr(x, omission).statistic), 6)
                                     for n, x in values.items()})
        p, f = values["partition_v2"], values["fa_v2"]
        assert np.max(np.abs(f - values["presence"] - p)) < 1e-12
        delta = lambda x: x[:, None] - x[None, :]
        dy, dp, df = delta(y), delta(p), delta(f)
        eligible = np.triu((dy != 0) & (dp != 0) & (df != 0), 1)
        reversal = eligible & (dp * df < 0)
        bad = int(np.count_nonzero(reversal & (dp * dy > 0)))
        good = int(np.count_nonzero(reversal & (df * dy > 0)))
        observed = (int(eligible.sum()), int(reversal.sum()), bad, good)
        assert observed == (2688, 206, 166, 40), observed
        print("dependent_pair_counts=(eligible,reversed,P2_correct_to_wrong,reverse)", observed)
    for k in (2, 3, 4):
        info = pair_components(np.ones(12, int), np.repeat(np.arange(1, k + 1), 12 // k))
        assert np.isclose(info["q_joint"], 1) and np.isclose(info["vi"], np.log(k))
        print("single_fragment_split", k, {n: info[n] for n in ("vi", "q_joint", "q_log_n")})
    reference = np.r_[np.ones(990, int), np.full(10, 2)]
    for name, prediction in (("small", np.r_[np.ones(990, int), np.full(5, 2), np.full(5, 3)]),
                             ("large", np.r_[np.ones(495, int), np.full(495, 3), np.full(10, 2)])):
        info = pair_components(reference, prediction)
        metric = instance_metrics(prediction.reshape(1, 1, -1), reference.reshape(1, 1, -1),
                                  (1., 1., 1.), connectivity=26,
                                  prediction_component_prune_min_volume_mm3=0)
        assert metric["normalized_merge_split_error"] == .5
        print("unpruned_toy_split_" + name, "risk=.5", {n: info[n] for n in ("vi", "q_joint")})
    print("PASS: four source hashes, validation-only descriptive diagnosis and tiny algebra checks.")
    print("No fitting, full-volume reads, bootstrap, held-out candidate scoring or file writes.")


if __name__ == "__main__":
    main()
