"""A simulator where the ground truth about anchors is known by construction.

Purpose.  Before trusting any importance number measured on a real model, we
need to know two things about the *instrument*:

1. Does it stay quiet when nothing is happening?  If resampling importance
   reports large values on traces where no sentence has any causal effect, then
   every number it produces on a real model is suspect.
2. Does it fire in the right place when something *is* happening?  A measure
   that never detects a planted anchor is useless regardless of how quiet it is.

The simulator gives both.  A trace is a latent belief state over ``K`` answers
that drifts and concentrates with depth; a rollout from prefix ``i`` resamples
every step after ``i`` and reads off an answer.  Because rollouts are generated
by the same forward dynamics as the trace, ``A_i`` here means exactly what it
means in the real pipeline.

Two worlds:

``null``    every step is pure drift.  No sentence causes anything.
``anchor``  one designated step applies a real jump -- and, crucially, a
            *resampled* step at that position reproduces the jump only with
            probability ``anchor_reproduce_p``.  This is what makes the anchor
            visible to a counterfactual measure at all: if resampling always
            regenerated the same content, ``A_{i-1}`` and ``A_i`` would agree
            and no black-box method could see the anchor.  An early version of
            this file had resampled steps always reproduce the anchor, and the
            planted effect was correctly invisible; that was a bug in the
            simulator, not a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimConfig:
    n_answers: int = 6
    n_sentences: int = 30
    n_rollouts: int = 48
    # Inverse temperature grows with depth: the distribution over answers
    # concentrates as the trace proceeds, for reasons that have nothing to do
    # with what any individual sentence says.  This is the structure the
    # position-only control is meant to expose.
    beta_start: float = 0.6
    beta_end: float = 3.0
    drift_scale: float = 0.5
    anchor_step: int | None = None
    anchor_strength: float = 5.0
    anchor_reproduce_p: float = 0.25


def _beta(cfg: SimConfig, t: int) -> float:
    return cfg.beta_start + (cfg.beta_end - cfg.beta_start) * t / cfg.n_sentences


def _drift(logits: np.ndarray, cfg: SimConfig, rng: np.random.Generator) -> np.ndarray:
    return logits + rng.normal(0.0, cfg.drift_scale, size=logits.shape)


def _maybe_anchor(logits: np.ndarray, t: int, cfg: SimConfig,
                  rng: np.random.Generator, realised: bool) -> np.ndarray:
    """Apply the planted effect if this step is the anchor.

    On the realised trace it always applies -- that is what the sentence said.
    On a resampled step it applies only sometimes, because the model would have
    to write equivalent content again.
    """
    if cfg.anchor_step is None or t != cfg.anchor_step:
        return logits
    if realised or rng.random() < cfg.anchor_reproduce_p:
        out = logits.copy()
        out[0] += cfg.anchor_strength
        return out
    return logits


def _answer(logits: np.ndarray, cfg: SimConfig, rng: np.random.Generator) -> str:
    p = np.exp(_beta(cfg, cfg.n_sentences) * (logits - logits.max()))
    p /= p.sum()
    return str(rng.choice(cfg.n_answers, p=p))


def simulate_trace(cfg: SimConfig, seed: int) -> dict:
    """One trace plus the rollout answer sets at every prefix boundary.

    The returned dict mirrors the shape the real pipeline produces, so the same
    analysis code consumes simulated and real data without branching.
    """
    rng = np.random.default_rng(seed)

    states = [np.zeros(cfg.n_answers)]
    for t in range(cfg.n_sentences):
        s = _drift(states[-1], cfg, rng)
        states.append(_maybe_anchor(s, t, cfg, rng, realised=True))

    def rollout(state: np.ndarray, t0: int, n: int) -> list[str]:
        """Resample steps ``t0 ... n_sentences-1`` starting from ``state``."""
        answers = []
        for _ in range(n):
            s = state
            for t in range(t0, cfg.n_sentences):
                s = _maybe_anchor(_drift(s, cfg, rng), t, cfg, rng, realised=False)
            answers.append(_answer(s, cfg, rng))
        return answers

    # A_i: continue from the realised state after sentence i.
    main = {i: rollout(states[i + 1], i + 1, cfg.n_rollouts)
            for i in range(-1, cfg.n_sentences)}

    # Filler: step i happens but carries no content -- pure drift, never the
    # anchor -- and then the trace continues as usual.
    filler = {}
    for i in range(cfg.n_sentences):
        s = _drift(states[i], cfg, rng)
        filler[i] = rollout(s, i + 1, cfg.n_rollouts)

    gold = _answer(states[-1], cfg, np.random.default_rng(seed + 7919))
    return {"main": main, "filler": filler, "gold": gold,
            "n_sentences": cfg.n_sentences}
