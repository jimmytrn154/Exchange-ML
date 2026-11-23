"""Prediction-only dual-channel CT triage with fold-local rank calibration.

Structural P1 and foreground presence are calibrated independently against
validation predictions by anatomy, then combined by a fixed maximum. No GT
outcome selects a rule, weight or threshold. The existing PENGWIN expected-
anatomy guard is kept separate; its assumptions do not extend to RibFrac.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np

from .partition_cv import grouped_folds
from .e1_statistics import cluster_draws
from .evaluator_sensitivity import ENDPOINTS, SOURCES, guarded_retention

CHANNELS = ('partition_v1', 'presence')
POLICIES = ('p1', 'guard_p1', 'presence', 'guard_presence', 'fa_v2',
            'calibrated_p1', 'guard_calibrated_p1', 'calibrated_presence',
            'dual_channel', 'guard_dual_channel')
CANDIDATE = 'guard_dual_channel'
BASELINE = 'guard_p1'


def validate_predictors(rows: Sequence[dict]) -> None:
    seen = set()
    for r in rows:
        key = r['case_id'], r['bone']
        if key in seen or not isinstance(r['all_members_empty'], (bool, np.bool_)):
            raise ValueError('unique case/bone rows and explicit empty flag required')
        seen.add(key)
        if r['scan_type'] not in ('pelvic', 'femur'):
            raise ValueError('declared PENGWIN scan type required')
        x = np.asarray([r[s] for s in (*CHANNELS, 'fa_v2')], float)
        if not np.isfinite(x).all() or (x < 0).any() or (x > 1+1e-10).any():
            raise ValueError('finite predictor scores in [0,1] required')
    if not seen:
        raise ValueError('nonempty predictors required')
    for cid in {r['case_id'] for r in rows}:
        sub = [r for r in rows if r['case_id'] == cid]
        expected = {'sacrum', 'left_hip', 'right_hip'} if sub[0]['scan_type'] == 'pelvic' else {'femur'}
        if len({r['scan_type'] for r in sub}) != 1 or {r['bone'] for r in sub} != expected:
            raise ValueError('complete expected anatomy per CT required')


def fit_reference(rows: Sequence[dict]) -> dict:
    validate_predictors(rows)
    return {'train_case_ids': sorted({r['case_id'] for r in rows}),
            'anatomy': {bone: {channel: sorted(float(r[channel]) for r in rows if r['bone'] == bone)
                              for channel in CHANNELS} for bone in sorted({r['bone'] for r in rows})},
            'role': 'prediction_score_ECDF_not_risk_calibration', 'GT_used': False}


def apply_reference(model: dict, rows: Sequence[dict]) -> list[dict]:
    validate_predictors(rows)
    result = []
    for r in rows:
        q = {}
        for channel in CHANNELS:
            reference = np.asarray(model['anatomy'].get(r['bone'], {}).get(channel, []), float)
            if not len(reference) or not np.isfinite(reference).all() or (np.diff(reference) < 0).any():
                raise ValueError('finite sorted per-anatomy reference required')
            value = float(r[channel])
            q[channel] = float((np.searchsorted(reference, value, side='left') +
                                np.searchsorted(reference, value, side='right'))/(2*len(reference)))
        # Copy inputs for evaluation only; predictors explicitly read the whitelist above.
        result.append({**r, 'calibrated_p1': q['partition_v1'],
                       'calibrated_presence': q['presence'],
                       'dual_channel': max(q.values())})
    return result


def calibrated_datasets(validation: list[dict], heldout: list[dict], *, folds: int = 5, seed: int = 20261122) -> tuple[dict, dict]:
    validate_predictors(validation); validate_predictors(heldout)
    if {r['case_id'] for r in validation} & {r['case_id'] for r in heldout}:
        raise ValueError('validation and heldout cases overlap')
    assignment = grouped_folds(validation, folds, seed)
    records, oof = [], []
    for fold in sorted(set(assignment.values())):
        train = [r for r in validation if assignment[r['case_id']] != fold]
        test = [r for r in validation if assignment[r['case_id']] == fold]
        model = fit_reference(train)
        records.append({'outer_fold': fold, 'test_case_ids': sorted({r['case_id'] for r in test}), 'model': model})
        oof.extend({**r, 'outer_fold': fold} for r in apply_reference(model, test))
    final = fit_reference(validation)
    return {'validation': sorted(oof, key=lambda r: (r['case_id'], r['bone'])),
            'heldout': apply_reference(final, heldout)}, {'folds': assignment, 'fold_models': records, 'reference': final}


def case_arrays(rows: list[dict], policy: str) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    if policy not in POLICIES:
        raise ValueError('fixed policy required')
    channel = policy.removeprefix('guard_')
    if channel == 'p1': channel = 'partition_v1'
    metadata, scores, risks, flags = [], [], [], []
    for cid in sorted({r['case_id'] for r in rows}):
        sub = [r for r in rows if r['case_id'] == cid]
        metadata.append({'case_id': cid, 'scan_type': sub[0]['scan_type']})
        scores.append(max(float(r[channel]) for r in sub))
        flags.append(any(r['all_members_empty'] for r in sub))
        risks.append([float(np.mean([r[f'{s}_{e}'] for r in sub])) for s in SOURCES for e in ENDPOINTS] +
                     [float(any(r[f'{s}_missing_volume'] == 1 for r in sub)) for s in SOURCES])
    y = np.asarray(risks, float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('finite nonnegative evaluation risks required')
    return metadata, np.asarray(scores), y, np.asarray(flags, bool)


def interval(samples: np.ndarray, enough: bool) -> tuple[float | None, float | None, str]:
    if not enough: return None, None, 'insufficient_cases'
    if not np.isfinite(samples).all(): return None, None, 'undefined_bootstrap_draws'
    low, high = np.quantile(samples, [.025, .975])
    return float(low), float(high), 'estimated'


def report(rows: list[dict], *, resamples: int = 10000, seed: int = 20261123) -> dict:
    validate_predictors(rows)
    if resamples < 1: raise ValueError('positive bootstrap count required')
    names = [(s, e) for s in SOURCES for e in ENDPOINTS]
    tables = {k: [] for k in ('ct_review', 'paired_differences', 'ct_omission', 'case_scores')}
    for cid in sorted({r['case_id'] for r in rows}):
        sub = [r for r in rows if r['case_id'] == cid]
        tables['case_scores'].append({'case_id': cid, 'scan_type': sub[0]['scan_type'],
            'flagged': any(r['all_members_empty'] for r in sub),
            **{policy: float(case_arrays(sub, policy)[1][0]) for policy in POLICIES}})
    for stratum in ('pooled', 'pelvic', 'femur'):
        sub = [r for r in rows if stratum == 'pooled' or r['scan_type'] == stratum]
        if not sub: continue
        meta, _, y, _ = case_arrays(sub, BASELINE)
        n = len(meta);draws = cluster_draws(meta, resamples, seed)
        for fraction in (.1, .2):
            reviewed = int(np.ceil(fraction*n));retained = n-reviewed
            if retained < 1: raise ValueError('at least one retained CT required')
            points, boots, feasible = {}, {}, {}
            for policy in POLICIES:
                _, x, y, flags = case_arrays(sub, policy)
                point, ok = guarded_retention(x[None], y[None], flags[None], retained, guarded=policy.startswith('guard_'))
                boot = np.empty((resamples, y.shape[1]), float); valid = np.empty(resamples, bool)
                for start in range(0, resamples, 128):
                    take = draws[start:start+128]
                    boot[start:start+len(take)], valid[start:start+len(take)] = guarded_retention(
                        x[take], y[take], flags[take], retained, guarded=policy.startswith('guard_'))
                points[policy], boots[policy], feasible[policy] = point[0], boot, bool(ok[0])
                common = {'stratum': stratum, 'policy': policy, 'n_cases': n, 'review_fraction': fraction,
                    'reviewed_CT': reviewed, 'flagged_CT': int(flags.sum()), 'guard_feasible': bool(ok[0]),
                    'bootstrap_feasible_fraction': float(valid.mean()), 'CI_role': 'conditional_fixed_calibration_scores_policy',
                    'analysis_role': 'posthoc_policy_development_not_fresh_confirmation'}
                for j, (source, endpoint) in enumerate(names):
                    lo, hi, status = interval(boot[:, j], n > 1)
                    tables['ct_review'].append({**common, 'risk_source': source, 'endpoint': endpoint,
                        'retained_risk': float(point[0,j]) if ok[0] else None, 'ci_low': lo, 'ci_high': hi,
                        'endpoint_status': 'constant_risk_uninformative' if np.ptp(y[:,j]) == 0 else 'variable_risk',
                        'ci_status': status, 'status': 'estimated' if ok[0] else 'guard_budget_infeasible'})
                for j, source in enumerate(SOURCES):
                    positive = float(y[:, -2+j].sum()); denominator = y[draws, -2+j].sum(1)
                    values = np.full(resamples, np.nan)
                    np.divide(retained*boot[:, -2+j], denominator, out=values, where=denominator > 0)
                    values = 1-values
                    lo, hi, status = interval(values, n > 1)
                    tables['ct_omission'].append({**common, 'risk_source': source, 'positive_CT': int(positive),
                        'sensitivity': float(1-retained*point[0,-2+j]/positive) if positive and ok[0] else None,
                        'ci_low': lo, 'ci_high': hi, 'ci_status': 'no_positive_CT' if not positive else status,
                        'bootstrap_defined_fraction': float(np.isfinite(values).mean())})
            for baseline in POLICIES:
                if baseline == CANDIDATE: continue
                for j, (source, endpoint) in enumerate(names):
                    lo, hi, status = interval(boots[CANDIDATE][:,j]-boots[baseline][:,j], n > 1)
                    target = baseline == BASELINE and stratum == 'pooled' and source == 'member_mean' and endpoint == 'missing_volume' and fraction == .2
                    tables['paired_differences'].append({'stratum': stratum, 'n_cases': n,
                        'policy': CANDIDATE, 'baseline': baseline, 'review_fraction': fraction,
                        'risk_source': source, 'endpoint': endpoint,
                        'delta_retained_risk': float(points[CANDIDATE][j]-points[baseline][j]) if feasible[CANDIDATE] and feasible[baseline] else None,
                        'ci_low': lo, 'ci_high': hi, 'ci_status': status,
                        'endpoint_status': 'constant_risk_uninformative' if np.ptp(y[:,j]) == 0 else 'variable_risk',
                        'contrast_role': 'declared_followup_target_not_original_E3_primary' if target else 'secondary',
                        'interpretation': 'lower_is_better_marginal_CI_no_new_success_gate'})
    return tables
