"""Supplementary PENGWIN matching/overlap and fixed-CT-budget sensitivity.

Original endpoints stay immutable. Significant edges require intersection
covering tau of BOTH objects (intersection/max volume), so small satellites
are suppressed; this can also suppress legitimate unequal fragments. Both
fixed thresholds are reported, never selected using held-out outcomes.
"""
from __future__ import annotations

from math import ceil
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .e1_scoring import SCORE_NAMES
from .e1_statistics import aurc_batch, cluster_draws
from .e5_supplement_statistics import retained_risk_batch

THRESHOLDS = (.01, .1)
SOURCES = ('member_mean', 'stage2')
ENDPOINTS = ('original_merge_split', 'original_one_minus_f1', 'one_to_one_one_minus_f1',
             'significant_0p01_merge_split', 'significant_0p1_merge_split', 'missing_volume')
SCORES = tuple(SCORE_NAMES)


def overlap_metrics(prediction: np.ndarray, reference: np.ndarray, *,
                    iou_threshold: float = .1, thresholds: Sequence[float] = THRESHOLDS) -> dict:
    """One-to-one detection and symmetric significant-edge counts, [Z,Y,X].

    Labels are arbitrary nonnegative integers. No pruning/relabeling is done.
    Maximum-cardinality admissible matching is selected before summed IoU.
    Structural values with zero GT are undefined, never silently zeroed.
    """
    pred, gt = np.asarray(prediction), np.asarray(reference)
    if (pred.shape != gt.shape or pred.ndim != 3 or pred.size == 0
            or any(not np.issubdtype(x.dtype, np.integer) or (x < 0).any() for x in (pred, gt))):
        raise ValueError('equal nonempty 3D nonnegative integer labels required')
    if not 0 < iou_threshold <= 1 or not thresholds or any(not 0 < t <= 1 for t in thresholds):
        raise ValueError('thresholds must be in (0,1]')
    # Restrict the contingency to foreground union, keeping exact background counts.
    active = (pred > 0) | (gt > 0)
    g, gi = np.unique(np.r_[0, gt[active]], return_inverse=True)
    p, pi = np.unique(np.r_[0, pred[active]], return_inverse=True)
    table = np.bincount(gi[1:] * len(p) + pi[1:], minlength=len(g)*len(p)).reshape(len(g), len(p))
    intersections = table[1:, 1:].astype(float)
    gv, pv = table[1:, :].sum(1), table[:, 1:].sum(0)
    ng, npred = len(g)-1, len(p)-1
    unions = gv[:, None] + pv[None, :] - intersections
    iou = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    if ng and npred:
        admissible = iou >= iou_threshold
        utility = admissible + iou / (min(ng, npred)+1)
        rr, cc = linear_sum_assignment(-utility)
        tp = int(admissible[rr, cc].sum())
    else:
        tp = 0
    f1 = 2*tp/(ng+npred) if ng+npred else 1.
    result = {'n_GT': ng, 'n_pred': npred, 'one_to_one_true_positives': tp,
              'one_to_one_false_positives': npred-tp, 'one_to_one_missed_instances': ng-tp,
              'one_to_one_one_minus_f1': 1-f1,
              'missing_volume': 1-float(intersections.sum())/float(gv.sum()) if ng else None}
    denominator = np.maximum(gv[:, None], pv[None, :])
    fraction = np.divide(intersections, denominator, out=np.zeros_like(intersections), where=denominator > 0)
    for threshold in thresholds:
        suffix = str(float(threshold)).replace('.', 'p')
        edges = (intersections > 0) & (fraction >= threshold)
        merges = int(np.maximum(edges.sum(0)-1, 0).sum())
        splits = int(np.maximum(edges.sum(1)-1, 0).sum())
        result.update({f'significant_{suffix}_merges': merges if ng else None,
                       f'significant_{suffix}_splits': splits if ng else None,
                       f'significant_{suffix}_merge_split': (merges+splits)/ng if ng else None})
    return result


def ct_arrays(rows: list[dict], score: str) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """GT-free max-bone score/empty flags, evaluation-only case-macro risks."""
    metadata, values, risks, flags = [], [], [], []
    for cid in sorted({r['case_id'] for r in rows}):
        sub = [r for r in rows if r['case_id'] == cid]
        scan = sub[0]['scan_type']
        expected = {'sacrum','left_hip','right_hip'} if scan == 'pelvic' else {'femur'}
        if {r['bone'] for r in sub} != expected or len(sub) != len(expected):
            raise ValueError('complete expected anatomy per CT required')
        metadata.append({'case_id': cid, 'scan_type': scan})
        values.append(max(float(r[score]) for r in sub))
        flags.append(any(r['all_members_empty'] for r in sub))
        risks.append([np.mean([r[f'{source}_{endpoint}'] for r in sub])
                      for source in SOURCES for endpoint in ENDPOINTS] +
                     [float(any(r[f'{source}_missing_volume'] == 1 for r in sub)) for source in SOURCES])
    return metadata, np.asarray(values), np.asarray(risks), np.asarray(flags, bool)


def _interval(values: np.ndarray) -> tuple[float | None, float | None]:
    return tuple(float(v) for v in np.quantile(values, [.025,.975])) if np.isfinite(values).all() else (None,None)


def guarded_retention(scores: np.ndarray, risks: np.ndarray, flags: np.ndarray,
                      retained: int, *, guarded: bool) -> tuple[np.ndarray, np.ndarray]:
    """Batched CT expectation, exact ties; guard promises all flags or undefined.

    scores/flags [B,C], risks [B,C,E]. Every CT costs one review. If flags
    exceed the fixed review budget, guard feasibility is false, not clipped.
    """
    x, y, f = np.asarray(scores,float), np.asarray(risks,float), np.asarray(flags,bool)
    if x.shape != f.shape or y.shape[:2] != x.shape:
        raise ValueError('incompatible batched CT arrays')
    feasible = f.sum(1) <= x.shape[1]-retained if guarded else np.ones(len(x),bool)
    priority = x + f*(np.ptp(x,axis=1)[:,None]+1) if guarded else x
    value = retained_risk_batch(priority,y,[retained])[:,0]
    value[~feasible] = np.nan
    return value,feasible


def report(rows: list[dict], *, resamples: int, seed: int,
           review_fractions: Sequence[float] = (.1,.2)) -> dict[str,list[dict]]:
    """Paired case-cluster CIs; supplementary, no model fitting or winner choice."""
    if resamples < 1 or not rows:
        raise ValueError('positive bootstrap draws and rows required')
    endpoints = [(source,e) for source in SOURCES for e in ENDPOINTS]
    tables = {k:[] for k in ('comparison','paired_differences','ct_review','ct_review_differences','ct_omission')}
    seen=set()
    for row in rows:
        key=row['case_id'],row['bone']
        if key in seen: raise ValueError('duplicate case/bone')
        seen.add(key)
        if not isinstance(row['all_members_empty'],(bool,np.bool_)):raise ValueError('explicit empty flag required')
        if not np.isfinite([row[s] for s in SCORES]+[row[f'{s}_{e}'] for s,e in endpoints]).all():
            raise ValueError('finite complete scores and endpoints required')
    for stratum in ('pooled','pelvic','femur'):
        sub=[r for r in rows if stratum=='pooled' or r['scan_type']==stratum]
        if not sub:continue
        nc=len({r['case_id'] for r in sub});common={'stratum':stratum,'n_cases':nc,'n_units':len(sub),
                'analysis_role':'post_E3_supplementary_not_fresh_confirmation','bootstrap_unit':'CT_within_scan_type'}
        y=np.asarray([[r[f'{s}_{e}'] for s,e in endpoints] for r in sub])
        x=np.asarray([[r[name] for name in SCORES] for r in sub])
        draws=cluster_draws(sub,resamples,seed)
        point=np.stack([aurc_batch(x[:,i][None],y[None])[0] for i in range(len(SCORES))])
        boot=np.empty((resamples,len(SCORES),len(endpoints)))
        for start in range(0,resamples,128):
            take=draws[start:start+128]
            for i in range(len(SCORES)):boot[start:start+len(take),i]=aurc_batch(x[take,i],y[take])
        for i,name in enumerate(SCORES):
            for j,(source,e) in enumerate(endpoints):
                lo,hi=_interval(boot[:,i,j]) if nc>1 else (None,None)
                tables['comparison'].append({**common,'score':name,'risk_source':source,'endpoint':e,
                     'aurc':float(point[i,j]),'ci_low':lo,'ci_high':hi,
                     'status':'uninformative_constant_risk' if np.ptp(y[:,j])==0 else 'estimated'})
                if name!='partition_v1':
                    k=SCORES.index('partition_v1');lo,hi=_interval(boot[:,i,j]-boot[:,k,j]) if nc>1 else (None,None)
                    tables['paired_differences'].append({**common,'score':name,'baseline':'partition_v1',
                         'risk_source':source,'endpoint':e,'delta_aurc':float(point[i,j]-point[k,j]),'ci_low':lo,'ci_high':hi})
        # Explicit equal-CT policies: raw P1, guard+P1, FA-v2 and presence.
        policies=('partition_v1','guard_partition_v1','fa_v2','presence')
        for fraction in review_fractions:
            meta,_,_,_=ct_arrays(sub,'partition_v1');n=len(meta);reviewed=ceil(fraction*n);retained=n-reviewed
            cdraw=cluster_draws(meta,resamples,seed)
            results=[];boots=[];statuses=[]
            positive_counts=None
            for policy in policies:
                _,cx,cy,flags=ct_arrays(sub,'partition_v1' if policy.startswith('guard_') else policy)
                guarded=policy.startswith('guard_')
                val,feasible=guarded_retention(cx[None],cy[None],flags[None],retained,guarded=guarded)
                bp=np.empty((resamples,cy.shape[1]));bf=np.empty(resamples,bool)
                for start in range(0,resamples,128):
                    take=cdraw[start:start+128]
                    bp[start:start+len(take)],bf[start:start+len(take)]=guarded_retention(cx[take],cy[take],flags[take],retained,guarded=guarded)
                results.append(val[0]);boots.append(bp);statuses.append(bool(feasible[0]))
                ccommon={**common,'policy':policy,'review_fraction':fraction,'reviewed_cts':reviewed,
                         'total_cts':n,'flagged_cts':int(flags.sum()),'guarded':guarded,
                         'bootstrap_feasible_fraction':float(bf.mean()),'estimand':'case_macro_retained_risk',
                         'perfect_human_correction_assumed':False,'measured_reading_time':False}
                for j,(source,e) in enumerate(endpoints):
                    lo,hi=_interval(bp[:,j]) if nc>1 else (None,None)
                    tables['ct_review'].append({**ccommon,'risk_source':source,'endpoint':e,
                         'retained_risk':float(val[0,j]) if feasible[0] else None,'ci_low':lo,'ci_high':hi,
                         'ci_status':'insufficient_cases' if nc<2 else 'guard_budget_infeasible' if not feasible[0]
                                     else 'guard_infeasible_in_bootstrap' if not bf.all() else 'estimated',
                         'status':'estimated' if feasible[0] else 'guard_budget_infeasible'})
                positive_counts=cy[:,-2:].sum(0)
                bootstrap_positive=cy[cdraw,-2:].sum(1)
                for j,source in enumerate(SOURCES):
                    positive=positive_counts[j]
                    sensitivity=1-retained*val[0,-2+j]/positive if positive and feasible[0] else None
                    samples=np.full(resamples,np.nan)
                    np.divide(retained*bp[:,-2+j],bootstrap_positive[:,j],out=samples,where=bootstrap_positive[:,j]>0)
                    samples=1-samples
                    lo,hi=_interval(samples) if nc>1 else (None,None)
                    tables['ct_omission'].append({**ccommon,'risk_source':source,'positive_cts':int(positive),
                         'sensitivity':float(np.clip(sensitivity,0,1)) if sensitivity is not None else None,
                         'ci_low':lo,'ci_high':hi,'status':'no_positive_cts' if not positive else 'estimated' if feasible[0] else 'guard_budget_infeasible',
                         'bootstrap_defined_fraction':float(np.isfinite(samples).mean()),
                         'bootstrap_positive_fraction':float((bootstrap_positive[:,j]>0).mean()),
                         'ci_status':'no_positive_cts' if not positive else 'insufficient_cases' if nc<2
                                     else 'guard_budget_infeasible' if not feasible[0]
                                     else 'no_positive_or_infeasible_bootstrap_draws' if not np.isfinite(samples).all() else 'estimated',
                         'positive_definition':'any_annotated_bone_missing_volume_equals_one'})
            for i,policy in enumerate(policies[1:],1):
                for j,(source,e) in enumerate(endpoints):
                    lo,hi=_interval(boots[i][:,j]-boots[0][:,j]) if nc>1 else (None,None)
                    tables['ct_review_differences'].append({**common,'policy':policy,'baseline':'partition_v1',
                         'review_fraction':fraction,'risk_source':source,'endpoint':e,
                         'delta_retained_risk':float(results[i][j]-results[0][j]) if statuses[i] and statuses[0] else None,
                         'ci_low':lo,'ci_high':hi,
                         'ci_status':'insufficient_cases' if nc<2 else 'guard_budget_infeasible' if not (statuses[i] and statuses[0])
                                     else 'guard_infeasible_in_bootstrap' if not np.isfinite(boots[i][:,j]-boots[0][:,j]).all() else 'estimated'})
    return tables
