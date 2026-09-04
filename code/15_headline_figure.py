"""The headline figure: what resampling importance is actually made of.

Neither position (my hypothesis) nor sentence category (the paper's) explains
much of the variance in resampling importance. What does is the dispersion of
the answer distribution the sentence arrives into -- how undecided the model
still was. That is a property of the *state*, not of the sentence, and it is
what a reader most needs to see.
"""

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

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
})
REAL, CTRL, NEUTRAL, ACC = "#1f5fa8", "#c0392b", "#7f8c8d", "#e08214"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]

    vd = json.loads((C.DATA / "variance_decomposition.json").read_text())
    df = pd.read_csv(C.DATA / f"sentences_{tag}.csv")
    d = df.dropna(subset=["kl_resampling", "position", "entropy_before"])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))

    # (a) what explains importance
    ax = axes[0]
    names = ["position\n(cubic)", "sentence\ncategory", "headroom\n$H(A_{i-1})$", "all three"]
    vals = [vd["r2_position"], vd["r2_category"], vd["r2_entropy_before"], vd["r2_all"]]
    cols = [REAL, ACC, CTRL, NEUTRAL]
    bars = ax.bar(range(4), vals, color=cols, alpha=0.9)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", fontsize=8.5)
    ax.set_xticks(range(4), names, fontsize=8)
    ax.set_ylabel("$R^2$ of resampling importance")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("What importance is made of", loc="left", fontsize=10)
    ax.grid(alpha=0.25, axis="y")

    # (b) the relationship itself
    ax = axes[1]
    ax.scatter(d.entropy_before, d.kl_resampling, s=10, alpha=0.35,
               color=NEUTRAL, edgecolors="none")
    x = d.entropy_before.values.astype(float)
    y = d.kl_resampling.values.astype(float)
    order = np.argsort(x)
    k = max(len(x) // 12, 5)
    xs = np.array([x[order][i:i + k].mean() for i in range(0, len(x) - k + 1, k)])
    ys = np.array([y[order][i:i + k].mean() for i in range(0, len(x) - k + 1, k)])
    ax.plot(xs, ys, color=CTRL, lw=2.4)
    rho = stats.spearmanr(x, y).statistic
    ax.set_xlabel("entropy of $A_{i-1}$ (nats) — how undecided the model still was")
    ax.set_ylabel("resampling importance  $D_{KL}[A_{i-1}\\,\\|\\,A_i]$")
    ax.set_title(f"Importance tracks the headroom it had  ($\\rho={rho:+.2f}$)",
                 loc="left", fontsize=10)
    ax.grid(alpha=0.25)

    # (c) and headroom is only weakly positional, so this is not position in disguise
    ax = axes[2]
    ax.scatter(d.position, d.entropy_before, s=10, alpha=0.35,
               color=NEUTRAL, edgecolors="none")
    bins = np.linspace(0, 1 + 1e-9, 11)
    g = d.groupby(pd.cut(d.position, bins), observed=True).entropy_before
    xs = np.linspace(0.05, 0.95, len(g.mean()))
    ax.plot(xs, g.mean().values, color=REAL, lw=2.4)
    ax.fill_between(xs, (g.mean() - g.sem()).values, (g.mean() + g.sem()).values,
                    color=REAL, alpha=0.18)
    r2 = stats.spearmanr(d.position, d.entropy_before).statistic
    ax.set_xlabel("normalised position in trace")
    ax.set_ylabel("entropy of $A_{i-1}$ (nats)")
    ax.set_title(f"Headroom is only partly positional  ($\\rho={r2:+.2f}$)",
                 loc="left", fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(C.FIGURES / f"fig0_headline_{tag}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig0_headline_{tag}.png")


if __name__ == "__main__":
    main()
