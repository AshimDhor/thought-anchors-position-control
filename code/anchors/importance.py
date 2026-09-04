"""Sentence importance by resampling, following Bogdan et al. (thought anchors).

Their definitions, which we reimplement rather than approximate:

    resampling importance      importance_r := D_KL[ p(A'_i) || p(A_i) ]
    counterfactual importance  importance   := D_KL[ p(A'_i | T_i !~ S_i) || p(A_i) ]

where ``A_i`` is the distribution over final answers when the model continues
from the prefix that *includes* sentence ``S_i``, and ``A'_i`` the distribution
when it continues from the prefix that *stops before* ``S_i`` -- so the model
resamples a replacement sentence ``T_i`` and carries on.  The counterfactual
variant keeps only rollouts whose replacement was semantically different from
the original (cosine similarity below SIM_THRESHOLD under all-MiniLM-L6-v2).

Note that ``A'_i`` and ``A_{i-1}`` are the same object: the rollouts launched
from prefix ``S_{<i}``.  So a single sweep over the ``n+1`` prefix boundaries
supplies every importance in the trace.

Two things this file is careful about, because both create positional artefacts
if ignored:

* **A finite-sample floor.**  With R rollouts a side, the empirical KL is
  strictly positive even when the two distributions are identical.  Its size
  depends on how much mass and how many distinct answers there are, both of
  which change systematically along a trace.  We estimate the floor by
  parametric bootstrap under the null and report it alongside every number.
* **Rollouts with no answer.**  A rollout that exhausts its token budget
  mid-thought gets its own outcome label rather than being scored wrong.
  Folding it into "incorrect" would put a positional trend into the data by
  hand, since later prefixes leave less budget to run over.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

NO_ANSWER = "<none>"

# Jeffreys prior.  Some smoothing is unavoidable -- an empirical distribution
# with a zero makes KL infinite -- and 1/2 is the standard non-informative
# choice.  03_analyze.py re-runs the headline numbers at 0.25 and 1.0 to show
# the conclusions do not depend on it.
SMOOTHING = 0.5


def _counts(answers: list[str | None]) -> Counter:
    return Counter(a if a is not None else NO_ANSWER for a in answers)


def _probs(counts: Counter, support: list[str], smoothing: float = 0.0) -> np.ndarray:
    raw = np.array([counts.get(s, 0) for s in support], dtype=float) + smoothing
    total = raw.sum()
    return raw / total if total > 0 else np.full(len(support), 1.0 / max(len(support), 1))


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL[p || q]; both must be smoothed so q has no zeros."""
    mask = p > 0
    return float((p[mask] * np.log(p[mask] / q[mask])).sum())


@dataclass
class PrefixStats:
    """Everything measured at one prefix boundary."""

    i: int                       # -1 = empty prefix: the model writes its own trace
    n_rollouts: int
    p_correct: float
    p_no_answer: float
    counts: Counter
    counts_dissimilar: Counter   # subset whose first new sentence differed from S_{i+1}
    counts_similar: Counter      # ... and the subset that came out semantically alike
    n_dissimilar: int
    n_similar: int

    @property
    def entropy(self) -> float:
        c = np.array(list(self.counts.values()), dtype=float)
        if c.sum() == 0:
            return 0.0
        p = c / c.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())


def prefix_stats(
    i: int,
    answers: list[str | None],
    gold: str,
    dissimilar_mask: list[bool] | None = None,
) -> PrefixStats:
    counts = _counts(answers)
    n = len(answers)
    if dissimilar_mask is None:
        diss, n_diss = Counter(), 0
        sim, n_sim = Counter(), 0
    else:
        keep = [a for a, m in zip(answers, dissimilar_mask) if m]
        drop = [a for a, m in zip(answers, dissimilar_mask) if not m]
        diss, n_diss = _counts(keep), len(keep)
        sim, n_sim = _counts(drop), len(drop)
    return PrefixStats(
        i=i,
        n_rollouts=n,
        p_correct=counts[gold] / n if n else float("nan"),
        p_no_answer=counts[NO_ANSWER] / n if n else float("nan"),
        counts=counts,
        counts_dissimilar=diss,
        counts_similar=sim,
        n_dissimilar=n_diss,
        n_similar=n_sim,
    )


def _null_divergence(
    counts_a: Counter, counts_b: Counter, rng: np.random.Generator,
    n_boot: int = 300,
) -> tuple[float, float]:
    """(E[KL], E[TV]) between two independent samples of the pooled distribution.

    This is what the statistic reads when the sentence did nothing at all.
    """
    pooled = counts_a + counts_b
    support = sorted(pooled)
    na, nb = sum(counts_a.values()), sum(counts_b.values())
    if na == 0 or nb == 0 or len(support) < 2:
        return 0.0, 0.0
    p = _probs(pooled, support)
    A = rng.multinomial(na, p, size=n_boot).astype(float)
    B = rng.multinomial(nb, p, size=n_boot).astype(float)
    Ps = (A + SMOOTHING) / (A + SMOOTHING).sum(axis=1, keepdims=True)
    Qs = (B + SMOOTHING) / (B + SMOOTHING).sum(axis=1, keepdims=True)
    kls = (Ps * np.log(Ps / Qs)).sum(axis=1)
    tvs = 0.5 * np.abs(A / na - B / nb).sum(axis=1)
    return float(kls.mean()), float(tvs.mean())


@dataclass
class SentenceImportance:
    index: int
    position: float              # (i + 1) / n, in (0, 1]
    kl_resampling: float         # the paper's importance_r
    kl_counterfactual: float     # the paper's importance (semantic filter applied)
    kl_similar: float            # same comparison, but for the rollouts whose
                                 # replacement WAS semantically like the original
    kl_null: float               # finite-sample floor for kl_resampling
    kl_corrected: float
    tv: float
    tv_null: float
    tv_corrected: float
    delta_acc: float
    abs_delta_acc: float
    entropy_before: float
    entropy_after: float
    p_correct_before: float
    p_correct_after: float
    n_dissimilar: int
    n_similar: int


def sentence_importances(
    stats_by_prefix: dict[int, PrefixStats],
    n_sentences: int,
    gold: str,
    seed: int = 0,
    smoothing: float = SMOOTHING,
) -> list[SentenceImportance]:
    rng = np.random.default_rng(seed)
    out: list[SentenceImportance] = []
    for i in range(n_sentences):
        before, after = stats_by_prefix.get(i - 1), stats_by_prefix.get(i)
        if before is None or after is None:
            continue

        support = sorted(set(before.counts) | set(after.counts))
        p_before = _probs(before.counts, support, smoothing)
        p_after = _probs(after.counts, support, smoothing)

        # Counterfactual variant: same A_i, but A' restricted to the rollouts
        # whose replacement sentence was semantically unlike the original.
        if before.n_dissimilar > 0:
            sup_c = sorted(set(before.counts_dissimilar) | set(after.counts))
            kl_cf = kl(_probs(before.counts_dissimilar, sup_c, smoothing),
                       _probs(after.counts, sup_c, smoothing))
        else:
            kl_cf = float("nan")

        # The free control.  The *complement* of that filter -- rollouts whose
        # replacement sentence came out semantically LIKE the original -- is an
        # on-policy paraphrase arm that costs nothing extra, because those
        # rollouts were generated anyway and then discarded.
        #
        # If importance is about content, replacing a sentence with a paraphrase
        # should leave the answer distribution alone, so kl_similar should sit
        # well below kl_counterfactual.  If the two are the same size, then what
        # the measure responds to is that a resampling happened at position i,
        # not what was said there.
        if before.n_similar > 0:
            sup_s = sorted(set(before.counts_similar) | set(after.counts))
            kl_sim = kl(_probs(before.counts_similar, sup_s, smoothing),
                        _probs(after.counts, sup_s, smoothing))
        else:
            kl_sim = float("nan")

        kl_null, tv_null = _null_divergence(before.counts, after.counts, rng)
        kl_raw, tv_raw = kl(p_before, p_after), tv(p_before, p_after)

        out.append(
            SentenceImportance(
                index=i,
                position=(i + 1) / n_sentences,
                kl_resampling=kl_raw,
                kl_counterfactual=kl_cf,
                kl_similar=kl_sim,
                kl_null=kl_null,
                kl_corrected=max(kl_raw - kl_null, 0.0),
                tv=tv_raw,
                tv_null=tv_null,
                tv_corrected=max(tv_raw - tv_null, 0.0),
                delta_acc=after.p_correct - before.p_correct,
                abs_delta_acc=abs(after.p_correct - before.p_correct),
                entropy_before=before.entropy,
                entropy_after=after.entropy,
                p_correct_before=before.p_correct,
                p_correct_after=after.p_correct,
                n_dissimilar=before.n_dissimilar,
                n_similar=before.n_similar,
            )
        )
    return out
