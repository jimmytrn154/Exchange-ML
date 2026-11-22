"""Small structural-risk heads with nested, scan-stratified case-grouped CV.

The target is log1p(mean per-member normalized merge/split risk). Predictions
are ranking scores, not failure probabilities. All preprocessing is fitted
inside the corresponding training fold; no GT-derived predictor is allowed.
The three feature families and the final family are fixed before evaluation.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

HEAD_FEATURES = {
    "ridge_p1": ("partition_v1",),
    "ridge_p1_a": ("partition_v1", "presence"),
    "ridge_p1_g_a": ("partition_v1", "graph_degree", "presence"),
}
FINAL_HEAD = "ridge_p1_g_a"
TARGET = "member_mean_merge_split"


def _settings(config: dict) -> dict:
    cv = config.get("cv", config)
    result = {"seed": cv.get("seed", cv.get("cv_seed", 20261122)),
              "outer_folds": cv.get("outer_folds", 5),
              "inner_folds": cv.get("inner_folds", 3),
              "alphas": cv.get("ridge_alphas", cv.get("alphas", [.1, 1., 10., 100.]))}
    for key in ("outer_folds", "inner_folds"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 2:
            raise ValueError(f"{key} must be an integer >=2")
    alphas = result["alphas"]
    if (not alphas or len(set(alphas)) != len(alphas)
            or any(not np.isfinite(a) or a <= 0 for a in alphas)):
        raise ValueError("ridge_alphas must be distinct finite positive values")
    result["alphas"] = sorted(float(a) for a in alphas)
    if not isinstance(result["seed"], int) or isinstance(result["seed"], bool) or result["seed"] < 0:
        raise ValueError("CV seed must be a nonnegative integer")
    return result


def validate_rows(rows: Sequence[dict], *, require_target: bool = True) -> None:
    if not rows:
        raise ValueError("nonempty case/bone rows are required")
    seen, scans = set(), {}
    for row in rows:
        cid, bone, scan = row["case_id"], row["bone"], row["scan_type"]
        if not isinstance(cid, str) or not cid or not isinstance(bone, str) or not bone:
            raise ValueError("case_id and bone must be nonempty strings")
        if (cid, bone) in seen:
            raise ValueError("duplicate case/bone unit")
        seen.add((cid, bone))
        if scan not in ("pelvic", "femur") or scans.get(cid, scan) != scan:
            raise ValueError("each case needs one consistent pelvic/femur scan type")
        scans[cid] = scan
        for name in ("partition_v1", "presence", "graph_degree"):
            value = row[name]
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid prediction-derived feature: {name}")
        if row["partition_v1"] > 1 + 1e-10 or row["presence"] > 1 + 1e-10:
            raise ValueError("partition_v1 and presence must be in [0,1]")
        if require_target and (not np.isfinite(row[TARGET]) or row[TARGET] < 0):
            raise ValueError("merge/split target must be finite and nonnegative; it may exceed 1")


def grouped_folds(rows: Sequence[dict], n_folds: int, seed: int) -> dict[str, int]:
    """Distribute complete CTs, balancing each scan type and total fold sizes.

    With very small smoke fixtures, some folds can lack one scan type. Such
    fixtures test execution only and cannot establish stratum performance.
    """
    scans = {}
    for row in rows:
        cid, scan = row["case_id"], row["scan_type"]
        if scan not in ("pelvic", "femur") or scans.get(cid, scan) != scan:
            raise ValueError("case scan type is inconsistent")
        scans[cid] = scan
    if not isinstance(n_folds, int) or isinstance(n_folds, bool) or not 2 <= n_folds <= len(scans):
        raise ValueError("fold count must be between 2 and the number of independent cases")
    rng = np.random.default_rng(seed)
    assigned, offset = {}, 0
    for scan in sorted(set(scans.values())):
        cases = np.array(sorted(c for c, s in scans.items() if s == scan))
        for index, cid in enumerate(rng.permutation(cases)):
            assigned[str(cid)] = int((offset + index) % n_folds)
        offset += len(cases)
    if set(assigned.values()) != set(range(n_folds)):
        raise ValueError("empty grouped fold")
    return assigned


def fit_model(rows: Sequence[dict], head: str, alpha: float) -> dict:
    """Ridge minimizes sum squared log-risk residuals plus alpha*||coef||²."""
    validate_rows(rows)
    if head not in HEAD_FEATURES or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("fixed head name and positive finite ridge alpha required")
    features = HEAD_FEATURES[head]
    x = np.array([[r[name] for name in features] for r in rows], dtype=np.float64)  # [N,F]
    y = np.log1p(np.array([r[TARGET] for r in rows], dtype=np.float64))  # [N]
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    standardized = (x - mean) / scale
    intercept = float(y.mean())
    coefficient = np.linalg.solve(standardized.T @ standardized + alpha * np.eye(len(features)),
                                  standardized.T @ (y - intercept))
    return {"schema_version": 1, "head": head, "features": list(features),
            "target": "log1p_member_mean_merge_split", "prediction_role": "ranking_not_probability",
            "alpha": float(alpha), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficient": coefficient.tolist(), "intercept": intercept,
            "train_case_ids": sorted({r["case_id"] for r in rows}), "n_train_units": len(rows)}


def predict_model(model: dict, rows: Sequence[dict]) -> np.ndarray:
    """Read only a frozen predictor whitelist; no target is accessed here."""
    head = model.get("head")
    if (model.get("schema_version") != 1 or head not in HEAD_FEATURES
            or model.get("features") != list(HEAD_FEATURES[head])
            or model.get("target") != "log1p_member_mean_merge_split"):
        raise ValueError("unsupported model identity or predictor features")
    features = model["features"]
    x = np.asarray([[r[name] for name in features] for r in rows], dtype=np.float64)
    if not rows:
        return np.empty(0, dtype=np.float64)
    mean, scale, coefficient = [np.asarray(model[name], float)
                                for name in ("mean", "scale", "coefficient")]
    if (any(v.shape != (len(features),) for v in (mean, scale, coefficient))
            or not np.isfinite(np.r_[mean, scale, coefficient, model["intercept"]]).all()
            or np.any(scale <= 0) or not np.isfinite(x).all()):
        raise ValueError("nonfinite/malformed model or prediction features")
    return (x - mean) / scale @ coefficient + model["intercept"]


def _choose_alpha(rows: list[dict], head: str, settings: dict, seed: int, scope: str
                  ) -> tuple[float, list[dict]]:
    n_cases = len({r["case_id"] for r in rows})
    if n_cases < 2:
        raise ValueError("insufficient independent cases for inner grouped validation")
    assignment = grouped_folds(rows, min(settings["inner_folds"], n_cases), seed)
    records = []
    for alpha in settings["alphas"]:
        errors = []
        for fold in sorted(set(assignment.values())):
            train = [r for r in rows if assignment[r["case_id"]] != fold]
            test = [r for r in rows if assignment[r["case_id"]] == fold]
            model = fit_model(train, head, alpha)
            y = np.log1p([r[TARGET] for r in test])
            errors.extend(((predict_model(model, test) - y) ** 2).tolist())
        records.append({"scope": scope, "head": head, "alpha": alpha,
                        "inner_mse_log_risk": float(np.mean(errors)),
                        "n_inner_folds": len(set(assignment.values())),
                        "inner_case_folds": dict(sorted(assignment.items())),
                        "selection_target": "bone_macro_MSE_log1p_member_mean_merge_split"})
    # Alphas are sorted: exact ties choose the smallest declared alpha.
    chosen = min(records, key=lambda r: (r["inner_mse_log_risk"], r["alpha"]))["alpha"]
    return chosen, records


def nested_cv(rows: Sequence[dict], config: dict) -> dict:
    """Evaluate all fixed heads out of fold, then fit deployment models.

    Final family is always ridge_p1_g_a. The outer outcomes cannot choose the
    family, alpha, feature scaling, or any other part of their own predictor.
    """
    validate_rows(rows)
    settings = _settings(config)
    ordered = sorted((dict(r) for r in rows), key=lambda r: (r["case_id"], r["bone"]))
    n_cases = len({r["case_id"] for r in ordered})
    if n_cases < 4:
        raise ValueError("insufficient_cases: nested grouped CV requires at least four CTs")
    assignment = grouped_folds(ordered, min(settings["outer_folds"], n_cases), settings["seed"])
    oof = [dict(r, outer_fold=assignment[r["case_id"]]) for r in ordered]
    tuning, fold_models = [], []
    for fold in sorted(set(assignment.values())):
        train = [r for r in ordered if assignment[r["case_id"]] != fold]
        test_indices = [i for i, r in enumerate(ordered) if assignment[r["case_id"]] == fold]
        test = [ordered[i] for i in test_indices]
        for head in HEAD_FEATURES:
            alpha, records = _choose_alpha(train, head, settings, settings["seed"] + 100 + fold,
                                           f"outer_{fold}")
            tuning.extend(records)
            model = fit_model(train, head, alpha)
            fold_models.append({"outer_fold": fold, "head": head, "model": model,
                                "test_case_ids": sorted({r["case_id"] for r in test})})
            for index, value in zip(test_indices, predict_model(model, test)):
                oof[index][head] = float(value)
    models = {}
    for head in HEAD_FEATURES:
        alpha, records = _choose_alpha(ordered, head, settings, settings["seed"] + 1000, "final")
        tuning.extend(records)
        models[head] = fit_model(ordered, head, alpha)
    folds = [{"case_id": cid, "scan_type": next(r["scan_type"] for r in ordered if r["case_id"] == cid),
              "outer_fold": fold} for cid, fold in sorted(assignment.items())]
    return {"oof_scores": oof, "folds": folds, "tuning": tuning, "models": models,
            "fold_models": fold_models, "final_model": models[FINAL_HEAD],
            "summary": {"n_cases": n_cases, "n_units": len(ordered), "final_head": FINAL_HEAD,
                        "outer_folds": len(set(assignment.values())), "inner_folds_requested": settings["inner_folds"],
                        "cv_seed": settings["seed"], "ridge_alphas": settings["alphas"],
                        "target": "log1p_member_mean_merge_split",
                        "selection": "inner_grouped_MSE_per_family_final_family_fixed",
                        "evaluation_role": "development_nested_grouped_oof_not_fresh_confirmation"}}


model_predict = predict_model
