"""The content-free controls.

The question this project asks is how much of a sentence's measured importance
is explained by *where it is* rather than *what it says*.  The instrument for
that is a predictor that is allowed to see position and nothing else.

Two things make this an honest test rather than a rigged one:

* The predictor is fit **leave-one-trace-out**.  Its curve is estimated from
  other problems entirely, so it cannot memorise the trace it scores.  A
  positional rule that only works in-sample would prove nothing.
* We report **top-k agreement**, not just R^2.  In practice nobody uses the
  importance scalar; they use it to pick the handful of sentences to look at.
  If a rule that has never read the text picks the same sentences, the expensive
  measurement did not add much.

This is the same control that a prefix-overlap evaluation in my earlier work
turned out to need, where a content-free positional rule outscored every
detector we had built.  The failure mode generalises, so it is worth checking
whether it applies here too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PositionBaselineResult:
    n_points: int
    n_traces: int
    spearman: float
    spearman_p: float
    r2_oos: float
    topk_agreement: dict[int, float]
    topk_chance: dict[int, float]
    curve_x: np.ndarray
    curve_y: np.ndarray


def _binned_curve(pos: np.ndarray, val: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean importance per position bin, with empty bins filled by interpolation."""
    idx = np.clip(np.digitize(pos, edges) - 1, 0, len(edges) - 2)
    curve = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            curve[b] = val[m].mean()
    ok = ~np.isnan(curve)
    if ok.sum() >= 2:
        centres = 0.5 * (edges[:-1] + edges[1:])
        curve = np.interp(centres, centres[ok], curve[ok])
    else:
        curve = np.nan_to_num(curve, nan=float(np.nanmean(val)))
    return curve


def position_only_baseline(
    trace_ids: np.ndarray,
    positions: np.ndarray,
    importance: np.ndarray,
    n_bins: int = 12,
    ks: tuple[int, ...] = (1, 3, 5),
    rng_seed: int = 0,
) -> PositionBaselineResult:
    """Predict importance from normalised position alone, leave-one-trace-out."""
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    preds = np.empty_like(importance, dtype=float)

    for tid in np.unique(trace_ids):
        held = trace_ids == tid
        curve = _binned_curve(positions[~held], importance[~held], edges)
        preds[held] = np.interp(positions[held], centres, curve)

    rho, p = stats.spearmanr(preds, importance)
    ss_res = float(((importance - preds) ** 2).sum())
    ss_tot = float(((importance - importance.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    rng = np.random.default_rng(rng_seed)
    agree: dict[int, list[float]] = {k: [] for k in ks}
    chance: dict[int, list[float]] = {k: [] for k in ks}
    for tid in np.unique(trace_ids):
        m = trace_ids == tid
        if m.sum() < max(ks) + 1:
            continue
        true_rank = np.argsort(-importance[m])
        pred_rank = np.argsort(-preds[m])
        n = int(m.sum())
        for k in ks:
            hit = len(set(true_rank[:k]) & set(pred_rank[:k])) / k
            agree[k].append(hit)
            # Chance for k draws without replacement from n is k/n.
            chance[k].append(k / n)

    return PositionBaselineResult(
        n_points=len(importance),
        n_traces=int(len(np.unique(trace_ids))),
        spearman=float(rho),
        spearman_p=float(p),
        r2_oos=r2,
        topk_agreement={k: float(np.mean(v)) for k, v in agree.items() if v},
        topk_chance={k: float(np.mean(v)) for k, v in chance.items() if v},
        curve_x=centres,
        curve_y=_binned_curve(positions, importance, edges),
    )


def bootstrap_ci_by_trace(
    trace_ids: np.ndarray,
    values: np.ndarray,
    statistic,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Cluster bootstrap over traces: sentences within a trace are not independent."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(trace_ids)
    point = statistic(values)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        sample = np.concatenate([values[trace_ids == t] for t in pick])
        draws.append(statistic(sample))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)
