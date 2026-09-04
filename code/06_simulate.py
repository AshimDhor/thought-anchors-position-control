"""Instrument validation: run the real analysis on data whose truth we planted.

Three worlds, and the third exists because building the first two exposed a
limitation of the position-only control that matters for how the real results
should be read.

``null``          no sentence causes anything.  A well-behaved measure should
                  sit at its noise floor, and the filler arm should match the
                  real arm.  Whether importance *also* shows a positional
                  pattern here is the diagnostic we care about: if it does, the
                  measure is positional by construction and nothing measured on
                  a real model can be trusted without correction.

``anchor_random`` one step per trace carries a real effect, at a position drawn
                  uniformly.  The measure must find it -- raw, and after the
                  position-only prediction is subtracted.

``anchor_fixed``  the same effect, but always at the same normalised position.
                  Here the position-only predictor learns the anchor itself, so
                  residualising *deletes a real finding*.  This is the honest
                  limit of the correlational control, and the reason the filler
                  arm -- which holds position fixed and varies only content --
                  is the load-bearing comparison rather than a nice-to-have.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from anchors import config as C
from anchors.baselines import position_only_baseline
from anchors.importance import _probs, kl, prefix_stats, sentence_importances
from anchors.simulate import SimConfig, simulate_trace


def run_world(name: str, cfg: SimConfig, n_traces: int, seed0: int,
              random_anchor: bool = False, bin_sweep: tuple[int, ...] = ()) -> dict:
    rng = np.random.default_rng(seed0)
    tids, poss, kls, fillers, idxs, planted = [], [], [], [], [], []

    for k in range(n_traces):
        this = cfg
        if random_anchor and cfg.anchor_step is not None:
            # Keep it away from the very ends, where a trace has little room
            # either side and the comparison degenerates.
            step = int(rng.integers(4, cfg.n_sentences - 4))
            this = SimConfig(**{**cfg.__dict__, "anchor_step": step})
        sim = simulate_trace(this, seed=seed0 + k)
        n = this.n_sentences

        stats = {i: prefix_stats(i, a, sim["gold"]) for i, a in sim["main"].items()}
        fstats = {i: prefix_stats(i, a, sim["gold"]) for i, a in sim["filler"].items()}

        for imp in sentence_importances(stats, n, sim["gold"], seed=seed0 + k):
            tids.append(f"{name}-{k}")
            poss.append(imp.position)
            kls.append(imp.kl_resampling)
            idxs.append(imp.index)
            planted.append(this.anchor_step if this.anchor_step is not None else -1)
            before, f = stats.get(imp.index - 1), fstats.get(imp.index)
            if before is None or f is None:
                fillers.append(np.nan)
                continue
            sup = sorted(set(before.counts) | set(f.counts))
            fillers.append(kl(_probs(before.counts, sup, 0.5), _probs(f.counts, sup, 0.5)))

    tids, poss = np.array(tids), np.array(poss)
    kls, fillers = np.array(kls), np.array(fillers, dtype=float)
    idxs, planted = np.array(idxs), np.array(planted)

    res = position_only_baseline(tids, poss, kls)
    resid = kls - np.interp(poss, res.curve_x, res.curve_y)

    bins = np.linspace(0, 1 + 1e-9, 7)
    which = np.clip(np.digitize(poss, bins) - 1, 0, len(bins) - 2)
    by_pos = [round(float(np.nanmean(kls[which == b])), 4) if (which == b).any() else None
              for b in range(len(bins) - 1)]

    out = {
        "world": name,
        "n_traces": n_traces,
        "mean_kl_by_position_sextile": by_pos,
        "position_only_spearman": res.spearman,
        "position_only_r2": res.r2_oos,
        "top3_agreement_position_only": res.topk_agreement.get(3),
        "top3_chance": res.topk_chance.get(3),
        "mean_real_kl": float(np.nanmean(kls)),
        "mean_filler_kl": float(np.nanmean(fillers)),
        "mean_real_minus_filler_all_positions": float(np.nanmean(kls - fillers)),
    }

    if (planted >= 0).any():
        at = planted == idxs                       # rows that ARE the planted step
        off = (planted >= 0) & ~at
        out["planted_real_minus_filler"] = float(np.nanmean((kls - fillers)[at]))
        out["offtarget_real_minus_filler"] = float(np.nanmean((kls - fillers)[off]))
        ranks_raw, ranks_res = [], []
        for tid in np.unique(tids):
            m = tids == tid
            step = planted[m][0]
            ranks_raw.append(int(np.where(idxs[m][np.argsort(-kls[m])] == step)[0][0]) + 1)
            ranks_res.append(int(np.where(idxs[m][np.argsort(-resid[m])] == step)[0][0]) + 1)
        out["median_rank_raw"] = float(np.median(ranks_raw))
        out["median_rank_residual"] = float(np.median(ranks_res))
        out["frac_top3_raw"] = float(np.mean([r <= 3 for r in ranks_raw]))
        out["frac_top3_residual"] = float(np.mean([r <= 3 for r in ranks_res]))

        # How much of the planted effect does residualising remove?  The
        # position-only curve is a smoother, so its resolution is a knob: coarse
        # bins smear a positionally-clustered anchor across neighbours and leave
        # most of it standing, fine bins subtract it away.  Anyone using this
        # control needs to know the knob exists.
        if bin_sweep:
            sweep = {}
            for nb in bin_sweep:
                r2 = position_only_baseline(tids, poss, kls, n_bins=nb)
                rr = kls - np.interp(poss, r2.curve_x, r2.curve_y)
                keep = []
                for tid in np.unique(tids):
                    m = tids == tid
                    step = planted[m][0]
                    rank = int(np.where(idxs[m][np.argsort(-rr[m])] == step)[0][0]) + 1
                    keep.append(rank <= 3)
                sweep[str(nb)] = {
                    "frac_top3_residual": float(np.mean(keep)),
                    "mean_planted_residual": float(np.nanmean(rr[at])),
                    "mean_planted_raw": float(np.nanmean(kls[at])),
                }
            out["bin_sweep"] = sweep
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", type=int, default=32)
    args = ap.parse_args()

    base = dict(n_answers=6, n_sentences=30, n_rollouts=C.ROLLOUTS_PER_PREFIX)
    results = [
        run_world("null", SimConfig(**base, anchor_step=None), args.traces, C.SEED),
        run_world("anchor_random", SimConfig(**base, anchor_step=12), args.traces,
                  C.SEED + 1000, random_anchor=True),
        run_world("anchor_fixed", SimConfig(**base, anchor_step=12), args.traces,
                  C.SEED + 2000, random_anchor=False, bin_sweep=(6, 12, 30, 60)),
    ]
    for r in results:
        print(json.dumps(r, indent=2))

    null, rand, fixed = results
    checks = [
        ("null: real arm matches filler arm (|gap| < 0.05 nats)",
         abs(null["mean_real_minus_filler_all_positions"]) < 0.05),
        ("anchor_random: filler gap is positive AT the planted step",
         rand["planted_real_minus_filler"] > 0),
        ("anchor_random: filler gap is ~zero away from the planted step",
         abs(rand["offtarget_real_minus_filler"]) < 0.05),
        ("anchor_random: planted step in top-3 raw (>60% of traces)",
         rand["frac_top3_raw"] > 0.6),
        ("anchor_random: planted step in top-3 after residualising (>60%)",
         rand["frac_top3_residual"] > 0.6),
        # Stated as a directional prediction, not a threshold: the point is
        # that the attenuation grows as the smoother gets finer.
        ("anchor_fixed: residualising attenuates the real anchor monotonically "
         "as bins get finer",
         all(fixed["bin_sweep"][a]["mean_planted_residual"]
             >= fixed["bin_sweep"][b]["mean_planted_residual"]
             for a, b in zip(["6", "12", "30"], ["12", "30", "60"]))),
    ]
    print("\n--- declared pass conditions ---")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print(f"\n  [diagnostic] null-world KL by position sextile: "
          f"{null['mean_kl_by_position_sextile']}")
    print(f"  [diagnostic] null-world position-only rho = "
          f"{null['position_only_spearman']:+.3f}")
    print("  [diagnostic] anchor_fixed, effect surviving residualisation by bin count:")
    for nb, v in fixed["bin_sweep"].items():
        print(f"      {nb:>3s} bins: planted KL {v['mean_planted_raw']:.3f} raw -> "
              f"{v['mean_planted_residual']:.3f} residual  "
              f"(top-3 in {v['frac_top3_residual']:.0%} of traces)")

    (C.DATA / "simulation.json").write_text(json.dumps(
        {"results": results,
         "checks": [{"check": c, "pass": bool(o)} for c, o in checks]}, indent=2))
    print(f"\nwrote {C.DATA / 'simulation.json'}")


if __name__ == "__main__":
    main()
