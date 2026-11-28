"""Recompute risk-coverage curves/AURC from saved per-bone scores and risks.

Convention (Methods): retain the lowest-scoring fraction c of bones, c = 1/n..1,
expected order within exact ties, trapezoids over c without adding c = 0.
"""
import csv, sys
from collections import defaultdict
import numpy as np

SCORES = ['mean_entropy', 'p95_entropy', 'pairwise_dice', 'presence', 'count_variance',
          'union_nvi', 'hungarian', 'fa_v1', 'fa_v2', 'q_joint_unweighted',
          'partition_v2', 'partition_v1']


def load(split, root):
    s = {(r['case_id'], r['bone']): r for r in csv.DictReader(open(f'{root}/{split}/scores.csv'))}
    k = {(r['case_id'], r['bone']): r for r in csv.DictReader(open(f'{root}/{split}/risks.csv'))}
    keys = sorted(s)
    assert set(keys) == set(k)
    return keys, s, k


def curve(score, risk):
    """Expected prefix-mean risk at coverage k/n under random order within ties."""
    score = np.asarray(score, float); risk = np.asarray(risk, float); n = len(score)
    groups = defaultdict(list)
    for sc, r in zip(score, risk):
        groups[sc].append(r)
    cum = []  # expected retained risk sum after k bones
    total = 0.0
    for sc in sorted(groups):
        g = groups[sc]; m = np.mean(g)
        for _ in g:
            total += m; cum.append(total)
    cum = np.array(cum); k = np.arange(1, n + 1)
    return k / n, cum / k


def aurc(c, r):
    return float(np.trapezoid(r, c))


if __name__ == '__main__':
    root = sys.argv[1]
    for split in ('validation', 'heldout'):
        keys, s, k = load(split, root)
        for risk in ('member_mean_merge_split', 'member_mean_one_minus_f1'):
            rv = [float(k[x][risk]) for x in keys]
            out = {sc: round(aurc(*curve([float(s[x][sc]) for x in keys], rv)), 4) for sc in SCORES}
            print(split, risk, out)
