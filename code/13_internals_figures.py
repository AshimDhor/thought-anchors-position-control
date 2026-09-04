from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from anchors import config as C
from anchors.baselines import position_only_baseline

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white",
})
REAL, CTRL, NEUTRAL = "#1f5fa8", "#c0392b", "#7f8c8d"


def load_attn(tag: str, trace_id: str) -> np.ndarray:
    return np.load(C.DATA / f"attn_{tag}_{trace_id.replace('/', '_')}.npy").astype(np.float32)


def fig_head_map(summary: dict, tag: str) -> None:
    """Layer x head map of how sharply each head concentrates backward attention."""
    peaked = np.array(summary["peakedness_by_layer_head"])
    L, H = peaked.shape
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8),
                             gridspec_kw={"width_ratios": [2.4, 1]})

    ax = axes[0]
    im = ax.imshow(peaked.T, aspect="auto", cmap="magma", origin="lower")
    ax.set_xlabel("layer"); ax.set_ylabel("head")
    ax.set_title("Receiver-head score: how sharply a head concentrates its "
                 "backward attention\non a few earlier sentences "
                 f"({L} layers x {H} heads)", loc="left", fontsize=9.5)
    for e in summary["top_receiver_heads"][:8]:
        ax.plot(e["layer"], e["head"], "o", mfc="none", mec="#39FF88", mew=1.6, ms=9)
    fig.colorbar(im, ax=ax, label="max / mean of received attention")

    ax = axes[1]
    ax.plot(peaked.mean(axis=1), np.arange(L), color=REAL, lw=1.8, label="mean over heads")
    ax.plot(peaked.max(axis=1), np.arange(L), color=CTRL, lw=1.4, ls="--", label="max head")
    ax.set_ylim(0, L - 1); ax.set_ylabel("layer"); ax.set_xlabel("receiver score")
    ax.set_title("Depth profile", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(C.FIGURES / f"fig6_headmap_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_attention_matrix(tag: str, trace_id: str, summary: dict) -> None:
    """Sentence-to-sentence attention: an average head vs the top receiver heads."""
    M = load_attn(tag, trace_id)
    top = [(e["layer"], e["head"]) for e in summary["top_receiver_heads"][:6]]
    avg = M.mean(axis=(0, 1))
    sel = np.mean([M[l, h] for l, h in top], axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, mat, title in [
        (axes[0], avg, "Averaged over all heads"),
        (axes[1], sel, "Top-6 receiver heads"),
    ]:
        v = np.percentile(mat[mat > 0], 99) if (mat > 0).any() else 1.0
        im = ax.imshow(mat, cmap="magma", origin="upper", vmin=0, vmax=v)
        ax.set_xlabel("attended-to sentence $j$")
        ax.set_ylabel("attending sentence $i$")
        ax.set_title(title, loc="left", fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.046)

    # The vertical stripes in the middle panel are the "broadcasting" sentences.
    # Whether they are content or position is the whole question.
    ax = axes[2]
    S = sel.shape[0]
    recv = np.array([sel[j + 1:, j].mean() if j + 1 < S else np.nan for j in range(S)])
    pos = (np.arange(S) + 1) / S
    ax.plot(pos, recv, color=REAL, lw=1.6)
    ax.fill_between(pos, 0, recv, color=REAL, alpha=0.18)
    ax.set_xlabel("normalised position of sentence $j$")
    ax.set_ylabel("attention received from later sentences")
    ax.set_title("Who gets broadcast to", loc="left", fontsize=9.5)
    ax.grid(alpha=0.25)

    fig.suptitle(f"Sentence-level attention, {trace_id}", fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"fig7_attnmatrix_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def fig_whitebox_vs_position(summary: dict, tag: str) -> None:
    """The headline white-box result, and the black-box measure beside it."""
    rows = summary["per_sentence"]
    tid = np.array([r["trace_id"] for r in rows])
    pos = np.array([r["position"] for r in rows])
    val = np.array([r["attention_received"] for r in rows])
    res = position_only_baseline(tid, pos, val)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    ax = axes[0]
    ax.scatter(pos, val, s=7, alpha=0.2, color=NEUTRAL, edgecolors="none")
    ax.plot(res.curve_x, res.curve_y, color=CTRL, lw=2.4,
            label=f"position-only predictor\n$\\rho={res.spearman:+.2f}$, "
                  f"$R^2_{{oos}}={res.r2_oos:+.2f}$")
    ax.set_xlabel("normalised position in trace")
    ax.set_ylabel("attention received from later sentences")
    # The honest title. The raw number is large, but the diagonal sweep below
    # shows most of it is recency, so the framing must not claim more.
    ax.set_title("Attention importance rises sharply at the end of the trace",
                 loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

    ax = axes[1]
    labels, agree, chance = [], [], []
    wb = summary["whitebox_position_only"]
    for k in ["1", "3", "5"]:
        labels.append(f"top-{k}")
        agree.append(wb["topk_agreement"].get(k, np.nan))
        chance.append(wb["topk_chance"].get(k, np.nan))
    bb = json.loads((C.DATA / f"summary_{tag}.json").read_text())["kl_resampling"]
    bb_agree = [bb["topk_agreement"].get(k, np.nan) for k in ["1", "3", "5"]]
    bb_chance = [bb["topk_chance"].get(k, np.nan) for k in ["1", "3", "5"]]

    x = np.arange(3); w = 0.35
    ax.bar(x - w / 2, agree, w, color=CTRL, label="white-box (attention)")
    ax.bar(x + w / 2, bb_agree, w, color=REAL, label="black-box (resampling KL)")
    ax.plot(x - w / 2, chance, "k_", ms=14, mew=2)
    ax.plot(x + w / 2, bb_chance, "k_", ms=14, mew=2, label="chance")
    ax.set_xticks(x, labels)
    ax.set_ylabel("fraction of true top-$k$ recovered\nby a predictor that never reads the text")
    ax.set_title("The two measures come apart", loc="left", fontsize=9.5)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25, axis="y")

    # Panel 3: the correction. Excluding attention near the diagonal removes
    # most of the effect, which is what identifies it as recency rather than
    # the long-range "broadcasting" the receiver-head story describes.
    ax = axes[2]
    sweep_path = C.DATA / f"mingap_sweep_{tag}.json"
    if sweep_path.exists():
        sw = json.loads(sweep_path.read_text())
        gaps = sorted(int(k) for k in sw)
        rho = [sw[str(g)]["spearman_oos"] for g in gaps]
        diag = [sw[str(g)]["near_diagonal_share"] for g in gaps]
        ax.plot(gaps, rho, "o-", color=CTRL, lw=2, label=r"position-only $\rho$")
        ax.plot(gaps, diag, "s--", color=NEUTRAL, lw=1.6,
                label="share of mass within 1\nsentence of the diagonal")
        ax.set_xscale("log", base=2)
        ax.set_xticks(gaps, [str(g) for g in gaps])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("minimum sentence gap counted as 'later'")
        ax.set_ylabel("value")
        ax.set_title("Excluding the diagonal removes most of it", loc="left",
                     fontsize=9.5)
        ax.legend(fontsize=7); ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(C.FIGURES / f"fig8_whitebox_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    summary = json.loads((C.DATA / f"internals_{tag}.json").read_text())

    fig_head_map(summary, tag)
    fig_whitebox_vs_position(summary, tag)
    longest = max(summary["traces"], key=lambda t: t["n_sentences"])
    fig_attention_matrix(tag, longest["trace_id"], summary)
    print(f"internals figures written to {C.FIGURES}")
    wb = summary["whitebox_position_only"]
    print(f"white-box position-only: rho={wb['spearman_oos']:+.3f} "
          f"R2={wb['r2_oos']:+.3f} top3={wb['topk_agreement'].get('3'):.2f} "
          f"(chance {wb['topk_chance'].get('3'):.3f})")


if __name__ == "__main__":
    main()
