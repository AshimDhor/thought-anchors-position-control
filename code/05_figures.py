from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from anchors import config as C
from anchors.baselines import position_only_baseline

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.facecolor": "white",
})

REAL, CTRL, NEUTRAL = "#1f5fa8", "#c0392b", "#7f8c8d"
NICE = {"kl_resampling": "resampling importance  $D_{KL}[A_{i-1}\\,\\|\\,A_i]$",
        "kl_counterfactual": "counterfactual importance",
        "kl_corrected": "importance, floor-corrected",
        "tv": "total variation", "abs_delta_acc": "$|\\Delta P(\\mathrm{correct})|$"}
SHORT = {"kl_resampling": "resampling", "kl_counterfactual": "counterfactual",
         "kl_corrected": "floor-corrected", "tv": "TV", "abs_delta_acc": "|ΔP(correct)|"}


def _binned(df: pd.DataFrame, col: str, nb: int = 12):
    bins = np.linspace(0, 1 + 1e-9, nb + 1)
    g = df.groupby(pd.cut(df.position, bins), observed=True)[col]
    return np.linspace(0.5 / nb, 1 - 0.5 / nb, nb)[: len(g.mean())], g.mean().values, g.sem().values


def fig_position(df: pd.DataFrame, tag: str, measure: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))

    ax = axes[0]
    ax.scatter(df.position, df[measure], s=7, alpha=0.22, color=NEUTRAL,
               edgecolors="none", label="one sentence")
    res = position_only_baseline(df.trace_id.values, df.position.values,
                                 df[measure].values.astype(float))
    ax.plot(res.curve_x, res.curve_y, color=REAL, lw=2.3,
            label=f"position-only predictor ($\\rho={res.spearman:+.2f}$ out-of-sample)")
    floor_col = "kl_null" if measure.startswith("kl") else "tv_null"
    if floor_col in df:
        x, y, _ = _binned(df, floor_col)
        ax.plot(x, y, color="k", ls=":", lw=1.5, label="finite-sample floor")
    ax.set_xlabel("normalised position in trace,  $(i{+}1)/n$")
    ax.set_ylabel(NICE.get(measure, measure))
    ax.set_title("Importance vs. position", loc="left", fontsize=10)
    ax.legend(loc="best", fontsize=7.2)

    ax = axes[1]
    ax.scatter(df.position, df.entropy_before, s=7, alpha=0.22, color=NEUTRAL,
               edgecolors="none")
    x, y, se = _binned(df, "entropy_before")
    ax.plot(x, y, color=CTRL, lw=2.3)
    ax.fill_between(x, y - se, y + se, color=CTRL, alpha=0.18)
    rho = stats.spearmanr(df.position, df.entropy_before).statistic
    ax.set_xlabel("normalised position in trace")
    ax.set_ylabel("entropy of $A_{i-1}$  (nats)")
    ax.set_title(f"Headroom left to move  ($\\rho={rho:+.2f}$)", loc="left", fontsize=10)

    fig.tight_layout()
    fig.savefig(C.FIGURES / f"fig1_position_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_topk(summary: dict, tag: str) -> None:
    measures = [m for m in ["kl_resampling", "kl_counterfactual", "tv", "abs_delta_acc"]
                if m in summary and "topk_agreement" in summary[m]]
    if not measures:
        return
    ks = [1, 3, 5]
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    width = 0.8 / len(measures)
    x = np.arange(len(ks))
    for j, m in enumerate(measures):
        vals = [summary[m]["topk_agreement"].get(str(k), np.nan) for k in ks]
        ax.bar(x + (j - (len(measures) - 1) / 2) * width, vals, width,
               label=SHORT.get(m, m))
    chance = [summary[measures[0]]["topk_chance"].get(str(k), np.nan) for k in ks]
    ax.plot(x, chance, "k--", marker="o", ms=4, lw=1.3, label="chance")
    ax.set_xticks(x, [f"top-{k}" for k in ks])
    ax.set_ylabel("fraction of the true top-$k$ recovered")
    ax.set_title("What a predictor that never reads the text recovers",
                 loc="left", fontsize=10)
    ax.legend(fontsize=7.2, ncol=2)
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"fig2_topk_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_filler(df: pd.DataFrame, tag: str, measure: str) -> None:
    fcol = "filler_kl" if measure.startswith("kl") else "filler_tv"
    if fcol not in df or df[fcol].isna().all():
        return
    sub = df.dropna(subset=[fcol]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))

    ax = axes[0]
    for col, colour, lab in [(measure, REAL, "the model's own sentence"),
                             (fcol, CTRL, "content-free filler")]:
        x, y, se = _binned(sub, col)
        ax.plot(x, y, color=colour, lw=2.2, label=lab)
        ax.fill_between(x, y - se, y + se, color=colour, alpha=0.18)
    ax.set_xlabel("normalised position in trace")
    ax.set_ylabel("divergence from the same $A_{i-1}$")
    ax.set_title("Same slot, with and without content", loc="left", fontsize=10)
    ax.legend(fontsize=7.8)

    ax = axes[1]
    lim = float(max(sub[measure].max(), sub[fcol].max())) * 1.05
    ax.scatter(sub[fcol], sub[measure], s=9, alpha=0.35, color=NEUTRAL, edgecolors="none")
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="equal effect")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("filler in slot $i$")
    ax.set_ylabel("real sentence in slot $i$")
    frac = float((sub[measure] > sub[fcol]).mean())
    ax.set_title(f"Real sentence wins in {frac:.0%} of slots", loc="left", fontsize=10)
    ax.legend(fontsize=7.8)

    fig.tight_layout()
    fig.savefig(C.FIGURES / f"fig3_filler_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def _place_labels(ax, xs, ys, labels, fontsize=7.5, colour="#222222"):

    fig = ax.get_figure()
    fig.canvas.draw()                      # needed before any bbox is valid
    placed = []
    # (dx, dy) in points, tried in order: right, left, above, below, diagonals.
    candidates = [(7, 3), (-7, 3), (7, -8), (-7, -8), (0, 9), (0, -13),
                  (13, 8), (-13, 8), (13, -13), (-13, -13), (0, 16), (0, -20)]
    order = np.argsort(-np.asarray(ys))    # place high points first
    for i in order:
        best = None
        for dx, dy in candidates:
            ha = "left" if dx >= 0 else "right"
            ann = ax.annotate(labels[i], (xs[i], ys[i]), fontsize=fontsize,
                              xytext=(dx, dy), textcoords="offset points",
                              ha=ha, color=colour, zorder=4)
            fig.canvas.draw()
            bb = ann.get_window_extent()
            inside = ax.get_window_extent().contains(bb.x0, bb.y0) and \
                     ax.get_window_extent().contains(bb.x1, bb.y1)
            if inside and not any(bb.overlaps(q) for q in placed):
                best = (ann, bb)
                break
            ann.remove()
        if best is None:                   # nothing clear: accept the default
            ann = ax.annotate(labels[i], (xs[i], ys[i]), fontsize=fontsize,
                              xytext=(7, 3), textcoords="offset points",
                              ha="left", color=colour, zorder=4)
            fig.canvas.draw()
            best = (ann, ann.get_window_extent())
        placed.append(best[1])


def fig_categories(df: pd.DataFrame, tag: str, label_col: str, measure: str) -> None:
    sub = df[(df[label_col] != "unparsed") & df[label_col].notna()].copy()
    if sub.empty or sub[label_col].nunique() < 3:
        return
    res = position_only_baseline(sub.trace_id.values, sub.position.values,
                                 sub[measure].values.astype(float))
    sub["residual"] = sub[measure].values - np.interp(sub.position.values,
                                                      res.curve_x, res.curve_y)
    order = sub.groupby(label_col)[measure].mean().sort_values(ascending=False).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9), sharey=True)
    for ax, col, title, colour in [
        (axes[0], measure, "Raw importance by sentence kind", REAL),
        (axes[1], "residual", "After subtracting the position-only prediction", CTRL),
    ]:
        m = sub.groupby(label_col)[col].mean().reindex(order)
        se = sub.groupby(label_col)[col].sem().reindex(order)
        ax.barh(range(len(order)), m.values, xerr=se.values, color=colour,
                alpha=0.85, error_kw={"lw": 1})
        ax.set_yticks(range(len(order)), [o.replace("_", " ") for o in order])
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("mean " + ("importance" if col == measure else "residual"))
        ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"fig4_categories_{tag}.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    mp = sub.groupby(label_col).position.mean().reindex(order)
    mi = sub.groupby(label_col)[measure].mean().reindex(order)
    ax.scatter(mp.values, mi.values, s=44, color=REAL, zorder=3)
    # Give the labels room, otherwise the outermost ones run off the axes.
    xpad = 0.10 * (mp.max() - mp.min())
    ypad = 0.12 * (mi.max() - mi.min())
    ax.set_xlim(mp.min() - xpad, mp.max() + xpad * 1.9)
    ax.set_ylim(mi.min() - ypad, mi.max() + ypad)
    _place_labels(ax, mp.values, mi.values,
                  [c.replace("_", " ") for c in order], fontsize=7.5)
    if len(order) >= 3:
        r = stats.pearsonr(mp.values, mi.values)
        ax.set_title(f"Category importance tracks category position  ($r={r.statistic:+.2f}$)",
                     loc="left", fontsize=10)
    ax.set_xlabel("mean position of sentences in this category")
    ax.set_ylabel("mean measured importance")
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"fig5_category_position_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    ap.add_argument("--measure", default="kl_resampling")
    ap.add_argument("--label-col", default=None)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]

    df = pd.read_csv(C.DATA / f"sentences_{tag}.csv")
    summary = json.loads((C.DATA / f"summary_{tag}.json").read_text())

    fig_position(df, tag, args.measure)
    fig_topk(summary, tag)
    fig_filler(df, tag, args.measure)
    label_col = args.label_col or next((c for c in df.columns if c.startswith("label_")), None)
    if label_col:
        fig_categories(df, tag, label_col, args.measure)
    print(f"figures written to {C.FIGURES}")


if __name__ == "__main__":
    main()
