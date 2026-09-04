from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from anchors import config as C
from anchors.baselines import position_only_baseline


def _ols_r2(X: np.ndarray, y: np.ndarray) -> float:
    A = np.column_stack([np.ones(len(y)), X]) if X.size else np.ones((len(y), 1))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    ap.add_argument("--measure", default="kl_resampling")
    ap.add_argument("--label-col", default=None)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]

    df = pd.read_csv(C.DATA / f"sentences_{tag}.csv")
    label_col = args.label_col or next(
        (c for c in df.columns if c.startswith("label_")), None)
    if label_col is None:
        print("no label column found; run 04_labels.py first")
        return

    d = df[(df[label_col].notna()) & (df[label_col] != "unparsed")].copy()
    d = d.dropna(subset=[args.measure, "position"])
    if d.empty or d[label_col].nunique() < 3:
        print("not enough labelled data")
        return
    y = d[args.measure].values.astype(float)
    pos = d.position.values.astype(float)
    cats = sorted(d[label_col].unique())
    D = np.column_stack([(d[label_col] == c).astype(float) for c in cats[:-1]])
    # Position enters as a cubic so the comparison is not rigged by giving
    # category a flexible form and position a straight line.
    P = np.column_stack([pos, pos ** 2, pos ** 3])

    out: dict = {"model": args.model, "measure": args.measure,
                 "label_col": label_col, "n": int(len(d)),
                 "categories": cats}

    # 1. How positional is category membership?
    grp = d.groupby(label_col)
    mean_pos = grp.position.mean()
    kw = stats.kruskal(*[g.position.values for _, g in grp])
    out["category_position"] = {
        "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue),
        "mean_position_by_category": {k: float(v) for k, v in mean_pos.items()},
        "spread_of_category_mean_positions": float(mean_pos.max() - mean_pos.min()),
    }
    print(f"category vs position: Kruskal H={kw.statistic:.1f} p={kw.pvalue:.2e}; "
          f"category mean positions span {mean_pos.min():.2f}-{mean_pos.max():.2f}")

    # 2. Nested variance decomposition.
    r2_pos = _ols_r2(P, y)
    r2_cat = _ols_r2(D, y)
    r2_both = _ols_r2(np.column_stack([P, D]), y)
    out["variance"] = {
        "r2_position_only": r2_pos, "r2_category_only": r2_cat,
        "r2_both": r2_both,
        "category_unique": r2_both - r2_pos,
        "position_unique": r2_both - r2_cat,
        "shared": r2_pos + r2_cat - r2_both,
    }
    v = out["variance"]
    print(f"R2: position {r2_pos:.3f} | category {r2_cat:.3f} | both {r2_both:.3f}")
    print(f"   unique to category {v['category_unique']:.3f}; "
          f"unique to position {v['position_unique']:.3f}; "
          f"shared {v['shared']:.3f}")

    # 3. Does the category ranking survive residualisation?
    res = position_only_baseline(d.trace_id.values, pos, y)
    d["residual"] = y - np.interp(pos, res.curve_x, res.curve_y)
    raw_rank = grp[args.measure].mean().sort_values(ascending=False)
    res_rank = d.groupby(label_col).residual.mean().sort_values(ascending=False)
    common = [c for c in raw_rank.index if c in res_rank.index]
    rho = stats.spearmanr(
        [list(raw_rank.index).index(c) for c in common],
        [list(res_rank.index).index(c) for c in common]).statistic
    out["ranking"] = {
        "raw_order": list(raw_rank.index),
        "residual_order": list(res_rank.index),
        "rank_correlation": float(rho),
        "raw_means": {k: float(v) for k, v in raw_rank.items()},
        "residual_means": {k: float(v) for k, v in res_rank.items()},
        "top_raw": list(raw_rank.index[:3]),
        "top_residual": list(res_rank.index[:3]),
    }
    print(f"category ranking, raw vs position-residualised: Spearman {rho:+.2f}")
    print(f"   top-3 raw      : {out['ranking']['top_raw']}")
    print(f"   top-3 residual : {out['ranking']['top_residual']}")

    # 4. And the direct, position-free comparison for each category: does the
    # real sentence beat its own filler within that category?
    fcol = "filler_kl" if args.measure.startswith("kl") else "filler_tv"
    if fcol in d and d[fcol].notna().any():
        sub = d.dropna(subset=[fcol])
        rows = {}
        for c, g in sub.groupby(label_col):
            rows[c] = {
                "n": int(len(g)),
                "mean_real": float(g[args.measure].mean()),
                "mean_filler": float(g[fcol].mean()),
                "gap": float((g[args.measure] - g[fcol]).mean()),
                "frac_real_exceeds": float((g[args.measure] > g[fcol]).mean()),
            }
        out["filler_by_category"] = rows
        print("\nfiller contrast within each category "
              "(position held fixed, content removed):")
        for c, r in sorted(rows.items(), key=lambda kv: -kv[1]["gap"]):
            print(f"   {c:24s} n={r['n']:4d}  real {r['mean_real']:.3f} "
                  f"vs filler {r['mean_filler']:.3f}  gap {r['gap']:+.3f}  "
                  f"real wins {r['frac_real_exceeds']:.0%}")

    dest = C.DATA / f"categories_{tag}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
