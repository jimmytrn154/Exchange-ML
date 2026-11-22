"""Two-feature linear pairwise structural ranker, separate from frozen v3.

Adaptation of the pairwise logistic idea in Burges et al., ICML 2005,
``Learning to Rank using Gradient Descent`` (RankNet). This is a convex linear
model, not the paper's neural model. Every unordered non-tied training-bone
pair has equal weight; it does not represent an additional independent CT.
Scores have no probability or calibrated risk interpretation.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .e1_statistics import aurc_batch
from .partition_cv import grouped_folds

FINAL_HEAD = "pairwise_p1_g"
FEATURES = ("partition_v1", "graph_degree")
TARGET = "member_mean_merge_split"
OBJECTIVE = "mean_pairwise_logistic_plus_half_alpha_l2"


def _settings(config: dict) -> dict:
    cv = config.get("cv", config)
    settings = {"seed": cv.get("seed", 20261122), "outer_folds": cv.get("outer_folds", 5),
                "inner_folds": cv.get("inner_folds", 3), "alphas": cv.get("alphas", [.001, .01, .1, 1.]),
                "max_iter": cv.get("max_iter", 1000), "tolerance": cv.get("tolerance", 1e-9)}
    for name in ("seed", "outer_folds", "inner_folds", "max_iter"):
        value = settings[name]
        minimum = 0 if name == "seed" else 2 if name.endswith("folds") else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"invalid integer CV setting: {name}")
    alphas = settings["alphas"]
    if (not alphas or len(set(alphas)) != len(alphas)
            or any(isinstance(a, bool) or not np.isfinite(a) or a <= 0 for a in alphas)):
        raise ValueError("alphas must be distinct finite positive numbers")
    settings["alphas"] = sorted(float(a) for a in alphas)
    if (isinstance(settings["tolerance"], bool) or not np.isfinite(settings["tolerance"])
            or not 0 < settings["tolerance"] < 1):
        raise ValueError("optimizer tolerance must be finite and strictly between zero and one")
    return settings


def _feature_array(rows: Sequence[dict]) -> np.ndarray:
    x = np.asarray([[r[name] for name in FEATURES] for r in rows], dtype=np.float64)
    if (not np.isfinite(x).all() or np.any(x < 0)
            or (len(rows) and np.any(x[:, 0] > 1 + 1e-10))):
        raise ValueError("P1 must be in [0,1] and G finite nonnegative")
    return x.reshape(len(rows), len(FEATURES))


def validate_rows(rows: Sequence[dict], *, require_target: bool = True) -> None:
    """Validate complete case groups without reading presence or GT covariates."""
    if not rows:
        raise ValueError("nonempty case/bone rows required")
    seen, groups = set(), {}
    for row in rows:
        cid, bone, scan = row["case_id"], row["bone"], row["scan_type"]
        if not isinstance(cid, str) or not cid or not isinstance(bone, str) or not bone:
            raise ValueError("nonempty string case_id and bone required")
        if (cid, bone) in seen:
            raise ValueError("duplicate case/bone unit")
        seen.add((cid, bone))
        if scan not in ("pelvic", "femur") or (cid in groups and groups[cid][0] != scan):
            raise ValueError("one consistent scan type required per CT")
        if cid not in groups:
            groups[cid] = (scan, set())
        groups[cid][1].add(bone)
        if require_target and (not np.isfinite(row[TARGET]) or row[TARGET] < 0):
            raise ValueError("finite nonnegative M+S required; values may exceed one")
    for scan, bones in groups.values():
        expected = {"sacrum", "left_hip", "right_hip"} if scan == "pelvic" else {"femur"}
        if bones != expected:
            raise ValueError("complete expected anatomy per CT required")
    _feature_array(rows)


def pairwise_differences(features: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """Return oriented differences [M,F], each larger-risk bone minus smaller.

    Exact risk ties are excluded. A monotone transformation of risk would
    leave these pairs unchanged; raw normalized M+S defines their order.
    """
    x, y = np.asarray(features, float), np.asarray(risks, float)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("finite features [N,F] and risks [N] required")
    first, second = np.triu_indices(len(x), 1)
    sign = np.sign(y[first] - y[second])
    keep = sign != 0
    return (x[first[keep]] - x[second[keep]]) * sign[keep, None]


def pairwise_loss_gradient(coefficient: np.ndarray, differences: np.ndarray,
                           alpha: float) -> tuple[float, np.ndarray]:
    """Uniform mean softplus(-D@beta) + alpha/2 ||beta||², analytic gradient."""
    beta, d = np.asarray(coefficient, float), np.asarray(differences, float)
    if (d.ndim != 2 or beta.shape != (d.shape[1],) or not np.isfinite(d).all()
            or not np.isfinite(beta).all() or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError("finite compatible coefficients/differences and positive alpha required")
    loss, gradient = float(alpha * (beta @ beta) / 2), alpha * beta
    if len(d):
        margin = d @ beta  # [M], no pair-by-feature tensor copy after this step.
        loss += float(np.logaddexp(0., -margin).mean())
        gradient = gradient - d.T @ expit(-margin) / len(d)
    return loss, gradient


def fit_model(rows: Sequence[dict], alpha: float, config: dict | None = None) -> dict:
    validate_rows(rows)
    settings = _settings(config or {})
    if isinstance(alpha, bool) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("finite positive alpha required")
    ordered = sorted(rows, key=lambda r: (r["case_id"], r["bone"]))
    x = _feature_array(ordered)  # [N,2], CPU float64.
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    if not np.isfinite(np.r_[mean, scale]).all():
        raise ValueError("training feature scaling overflow")
    z = (x - mean) / scale
    differences = pairwise_differences(z, np.array([r[TARGET] for r in ordered]))
    if not len(differences):
        coefficient, iterations, message, status = np.zeros(len(FEATURES)), 0, "all training risks tied", "no_risk_pairs"
    else:
        result = minimize(pairwise_loss_gradient, np.zeros(len(FEATURES)), args=(differences, alpha),
                          jac=True, method="L-BFGS-B", options={"maxiter": settings["max_iter"],
                          "gtol": settings["tolerance"], "ftol": np.finfo(float).eps * 10})
        if (not result.success or not np.isfinite(result.x).all()
                or np.max(np.abs(result.jac)) > max(1e-7, 10 * settings["tolerance"])):
            raise RuntimeError(f"pairwise optimizer failed convergence: {result.message}; gradient={result.jac}")
        coefficient, iterations, message, status = result.x, int(result.nit), str(result.message), "converged"
    loss, gradient = pairwise_loss_gradient(coefficient, differences, alpha)
    return {"schema_version": 1, "head": FINAL_HEAD, "features": list(FEATURES), "target": TARGET,
            "objective": OBJECTIVE, "pair_weighting": "uniform_non_tied_unordered_bone_pairs",
            "coefficient_constraints": "unconstrained", "prediction_role": "ranking_not_probability",
            "alpha": float(alpha), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficient": coefficient.tolist(), "intercept": 0.,
            "train_case_ids": sorted({r["case_id"] for r in ordered}), "n_train_units": len(ordered),
            "n_training_pairs": len(differences), "optimizer": {"status": status, "iterations": iterations,
            "message": message, "loss": loss, "gradient_infinity_norm": float(np.max(np.abs(gradient)))}}


def predict_model(model: dict, rows: Sequence[dict]) -> np.ndarray:
    """Inference uses exactly two prediction-derived features, never labels."""
    if (model.get("schema_version") != 1 or model.get("head") != FINAL_HEAD
            or model.get("features") != list(FEATURES) or model.get("target") != TARGET
            or model.get("objective") != OBJECTIVE or model.get("intercept") != 0.):
        raise ValueError("unsupported pairwise model identity or predictor whitelist")
    mean, scale, coefficient = (np.asarray(model[n], float) for n in ("mean", "scale", "coefficient"))
    if (any(v.shape != (len(FEATURES),) for v in (mean, scale, coefficient))
            or not np.isfinite(np.r_[mean, scale, coefficient]).all() or np.any(scale <= 0)):
        raise ValueError("nonfinite/malformed model parameters")
    x = _feature_array(rows)
    predictions = (x - mean) / scale @ coefficient
    if not np.isfinite(predictions).all():
        raise ValueError("prediction overflow")
    return predictions


def _choose_alpha(rows: list[dict], settings: dict, seed: int, scope: str) -> tuple[float, list[dict]]:
    n_cases = len({r["case_id"] for r in rows})
    if n_cases < 2:
        raise ValueError("insufficient independent CTs for inner CV")
    assignment = grouped_folds(rows, min(settings["inner_folds"], n_cases), seed)
    records = []
    for alpha in settings["alphas"]:
        fold_values = []
        for fold in sorted(set(assignment.values())):
            train = [r for r in rows if assignment[r["case_id"]] != fold]
            test = [r for r in rows if assignment[r["case_id"]] == fold]
            if len(test) < 2:
                raise ValueError("insufficient inner-fold bone units for AURC; do not drop undefined folds")
            model = fit_model(train, alpha, settings)
            risk = np.array([r[TARGET] for r in test])
            value = float(aurc_batch(predict_model(model, test)[None], risk[None, :, None])[0, 0])
            fold_values.append({"inner_fold": fold, "aurc": value, "n_cases": len({r["case_id"] for r in test}),
                                "n_units": len(test), "fit_status": model["optimizer"]["status"]})
        records.append({"scope": scope, "head": FINAL_HEAD, "alpha": alpha,
                        "mean_inner_fold_aurc": float(np.mean([v["aurc"] for v in fold_values])),
                        "inner_fold_results": fold_values, "inner_case_folds": dict(sorted(assignment.items())),
                        "selection_target": "unweighted_mean_inner_fold_member_mean_merge_split_AURC",
                        "alpha_tie_break": "largest_alpha"})
    chosen = min(records, key=lambda r: (r["mean_inner_fold_aurc"], -r["alpha"]))["alpha"]
    return chosen, records


def nested_cv(rows: Sequence[dict], config: dict) -> dict:
    """Evaluate the single fixed P1+G family in nested CT-grouped folds."""
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
        indices = [i for i, r in enumerate(ordered) if assignment[r["case_id"]] == fold]
        test = [ordered[i] for i in indices]
        alpha, records = _choose_alpha(train, settings, settings["seed"] + 100 + fold, f"outer_{fold}")
        tuning.extend(records)
        model = fit_model(train, alpha, settings)
        fold_models.append({"outer_fold": fold, "head": FINAL_HEAD, "model": model,
                            "test_case_ids": sorted({r["case_id"] for r in test})})
        for index, value in zip(indices, predict_model(model, test)):
            oof[index][FINAL_HEAD] = float(value)
    alpha, records = _choose_alpha(ordered, settings, settings["seed"] + 1000, "final")
    tuning.extend(records)
    final_model = fit_model(ordered, alpha, settings)
    folds = [{"case_id": cid, "scan_type": next(r["scan_type"] for r in ordered if r["case_id"] == cid),
              "outer_fold": fold} for cid, fold in sorted(assignment.items())]
    return {"oof_scores": oof, "folds": folds, "tuning": tuning, "fold_models": fold_models,
            "final_model": final_model, "summary": {"n_cases": n_cases, "n_units": len(ordered),
            "final_head": FINAL_HEAD, "features": list(FEATURES), "outer_folds": len(set(assignment.values())),
            "inner_folds_requested": settings["inner_folds"], "cv_seed": settings["seed"],
            "alphas": settings["alphas"], "target": TARGET, "objective": OBJECTIVE,
            "selection": "mean_inner_fold_AURC_largest_alpha_exact_tie_single_family",
            "primary_estimand": "mean_outer_fold_aurc",
            "evaluation_role": "development_after_v3_inspection_not_fresh_confirmation"}}
