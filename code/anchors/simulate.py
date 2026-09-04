from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimConfig:
    n_answers: int = 6
    n_sentences: int = 30
    n_rollouts: int = 48

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

    rng = np.random.default_rng(seed)

    states = [np.zeros(cfg.n_answers)]
    for t in range(cfg.n_sentences):
        s = _drift(states[-1], cfg, rng)
        states.append(_maybe_anchor(s, t, cfg, rng, realised=True))

    def rollout(state: np.ndarray, t0: int, n: int) -> list[str]:
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

   
    filler = {}
    for i in range(cfg.n_sentences):
        s = _drift(states[i], cfg, rng)
        filler[i] = rollout(s, i + 1, cfg.n_rollouts)

    gold = _answer(states[-1], cfg, np.random.default_rng(seed + 7919))
    return {"main": main, "filler": filler, "gold": gold,
            "n_sentences": cfg.n_sentences}
