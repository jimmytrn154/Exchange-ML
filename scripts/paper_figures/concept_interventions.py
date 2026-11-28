"""Controlled copies of one reference fragment map and their coverage/grouping terms.

Uses the definitions of Methods II-A on the whole 3D bone (reference vs. copy).
"""
import numpy as np


def terms(R, P):
    Fi, Fj = R > 0, P > 0
    U = (Fi | Fj).sum(); C = Fi & Fj; nC = C.sum()
    A = (Fi ^ Fj).sum() / U; w = nC / U
    a, b = R[C].astype(np.int64), P[C].astype(np.int64)
    _, n_ab = np.unique(a * 100000 + b, return_counts=True)
    _, n_a = np.unique(a, return_counts=True); _, n_b = np.unique(b, return_counts=True)
    H = lambda n: float(-(n / nC * np.log(n / nC)).sum())
    Hj, Ha, Hb = H(n_ab), H(n_a), H(n_b)
    VI = 2 * Hj - Ha - Hb
    Qj = VI / Hj if Hj > 0 else 0.0
    Qv = VI / np.log(nC) if nC > 1 else 0.0
    return dict(A=A, w=w, VI=VI, H=Hj, Qjoint=Qj, Qvox=Qv, wQjoint=w * Qj, wQvox=w * Qv)


def neighbours(m):
    out = np.zeros_like(m)
    for ax in range(3):
        for s in (1, -1):
            out |= np.roll(m, s, axis=ax)
    return out


def contact(lab, i, j):
    return int((neighbours(lab == i) & (lab == j)).sum())


def interventions(lab, spacing):
    ids, cnt = np.unique(lab[lab > 0], return_counts=True)
    ids = [int(i) for i in ids]
    out = {}
    # Merge: the pair of fragments with the largest contact surface.
    pairs = [(contact(lab, i, j), i, j) for k, i in enumerate(ids) for j in ids[k + 1:]]
    _, i, j = max(pairs)
    m = lab.copy(); m[m == j] = i; out['merge'] = (m, dict(kept=i, merged=j))
    # Split: largest fragment cut by the plane through its centroid normal to its principal axis.
    big = ids[int(np.argmax(cnt))]
    z, y, x = np.where(lab == big)
    pts = np.stack([z * spacing[2], y * spacing[1], x * spacing[0]], 1)
    c = pts - pts.mean(0); ax = np.linalg.svd(c, full_matrices=False)[2][0]
    side = c @ ax > 0
    s = lab.copy(); s[z[side], y[side], x[side]] = max(ids) + 1; out['split'] = (s, dict(split=big))
    # Removal: delete the smallest fragment.
    small = ids[int(np.argmin(cnt))]
    r = lab.copy(); r[r == small] = 0; out['removal'] = (r, dict(removed=small))
    # Boundary shift: erode the union foreground by one voxel, keeping labels.
    fg = lab > 0
    rim = fg & neighbours(~fg)
    b = lab.copy(); b[rim] = 0; out['boundary'] = (b, dict(rim=int(rim.sum())))
    return out
