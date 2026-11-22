"""E1 tie-aware trapezoid AURC and case-cluster paired bootstrap.

All v2 comparisons use the same convention; no change to historical v1 AURC.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def expected_curve(scores: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """Expected prefix risk over all orders within exact score ties."""
    x, y = np.asarray(scores, float), np.asarray(risks, float)
    if x.ndim != 1 or y.shape != x.shape or not np.isfinite([x, y]).all():
        raise ValueError("finite equal score/risk vectors required")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order].copy()
    starts = np.r_[0, np.flatnonzero(np.diff(x)) + 1]
    for start, end in zip(starts, np.r_[starts[1:], len(x)]):
        y[start:end] = y[start:end].mean()
    return np.cumsum(y) / np.arange(1, len(y) + 1)


def aurc_batch(scores: np.ndarray, risks: np.ndarray) -> np.ndarray:
    """Scores [B,N], risks [B,N,K] -> AURCs [B,K], preserving exact ties."""
    x, y = np.asarray(scores, float), np.asarray(risks, float)
    if x.ndim != 2 or y.ndim != 3 or x.shape != y.shape[:2]:
        raise ValueError("expected scores [B,N] and risks [B,N,K]")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("scores and risks must be finite")
    b, n = x.shape
    if n < 2:
        return np.full((b, y.shape[-1]), np.nan)
    order = np.argsort(x, axis=1, kind="stable")
    x = np.take_along_axis(x, order, 1)
    y = np.take_along_axis(y, order[..., None], 1)
    positions = np.broadcast_to(np.arange(n), (b, n))
    starts = np.maximum.accumulate(np.where(np.c_[np.ones(b, bool), np.diff(x) != 0], positions, 0), axis=1)
    ends = np.minimum.accumulate(np.where(np.c_[np.diff(x) != 0, np.ones(b, bool)], positions, n - 1)[:, ::-1], axis=1)[:, ::-1]
    sums = np.concatenate((np.zeros((b, 1, y.shape[-1])), np.cumsum(y, axis=1)), axis=1)
    means = (np.take_along_axis(sums, (ends + 1)[..., None], 1)
             - np.take_along_axis(sums, starts[..., None], 1)) / (ends - starts + 1)[..., None]
    curve = np.cumsum(means, axis=1) / np.arange(1, n + 1)[None, :, None]
    return np.sum((curve[:, :-1] + curve[:, 1:]) / (2 * n), axis=1)


def correlation(scores: np.ndarray, risks: np.ndarray, confounders: np.ndarray | None = None) -> float | None:
    """Average-rank Spearman, optionally rank residualization with intercept."""
    x, y = rankdata(scores), rankdata(risks)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    if confounders is not None:
        c = np.asarray(confounders, float)
        if not np.isfinite(c).all():
            return None
        design = np.column_stack((np.ones(len(x)), rankdata(c, axis=0)))
        if len(x) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
            return None
        x -= design @ np.linalg.lstsq(design, x, rcond=None)[0]
        y -= design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.linalg.norm(x - x.mean()) < 1e-10 or np.linalg.norm(y - y.mean()) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def cluster_draws(rows: list[dict], count: int, seed: int) -> np.ndarray:
    """Resample cases within scan type; carry every bone and duplicate draw."""
    rng = np.random.default_rng(seed)
    blocks = []
    for scan_type in sorted({r["scan_type"] for r in rows}):
        cases = sorted({r["case_id"] for r in rows if r["scan_type"] == scan_type})
        groups = [[i for i, r in enumerate(rows) if r["case_id"] == case] for case in cases]
        if len({len(g) for g in groups}) != 1:
            raise ValueError("all cases within a scan type must have the same annotated bones")
        chosen = rng.integers(len(groups), size=(count, len(groups)))
        blocks.append(np.asarray(groups, dtype=int)[chosen].reshape(count, -1))
    return np.concatenate(blocks, axis=1)
