"""Are the 'receiver heads' actually receiver heads, or just recency heads?

The sentence-level attention matrix for the heads my score picked out is almost
entirely diagonal: they attend to the sentence immediately before them. That is
recency, not the phenomenon Bogdan et al. describe, where a head reaches *back*
to a distant sentence that is broadcasting to the rest of the trace.

My receiver score used min_gap = 1, so the immediately-preceding sentence counted
as "received attention from a later sentence". This sweeps min_gap: excluding a
band around the diagonal forces the score to consider only genuinely long-range
attention. If the strong positional result survives at large gaps, it is about
broadcasting. If it collapses, my headline white-box number was measuring
recency and has to be restated.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from anchors import config as C
from anchors.baselines import position_only_baseline


def receiver_profile(M: np.ndarray, min_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """(peakedness[L,H], received[L,H,S]) counting only attention from >= min_gap later."""
    L, H, S, _ = M.shape
    r = np.zeros((L, H, S), dtype=np.float32)
    for j in range(S):
        rows = np.arange(j + min_gap, S)
        if len(rows) == 0:
            continue
        r[:, :, j] = M[:, :, rows, j].mean(axis=2)
    denom = r.mean(axis=2)
    peaked = r.max(axis=2) / np.clip(denom, 1e-9, None)
    return peaked, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    ap.add_argument("--top-heads", type=int, default=20)
    ap.add_argument("--gaps", default="1,2,4,8,16")
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    summary = json.loads((C.DATA / f"internals_{tag}.json").read_text())

    mats = {}
    for rec in summary["traces"]:
        tid = rec["trace_id"]
        mats[tid] = np.load(
            C.DATA / f"attn_{tag}_{tid.replace('/', '_')}.npy").astype(np.float32)

    out = {}
    print(f"{'gap':>4} {'rho':>8} {'R2_oos':>8} {'top3':>6} {'chance':>7}  "
          f"{'diag share':>10}  top heads")
    for gap in [int(g) for g in args.gaps.split(",")]:
        peaked_all, rows = [], []
        for tid, M in mats.items():
            peaked, r = receiver_profile(M, gap)
            peaked_all.append(peaked)
            rows.append((tid, r, M.shape[2]))
        peaked_mean = np.mean(peaked_all, axis=0)
        L, H = peaked_mean.shape
        flat = np.argsort(-peaked_mean.ravel())[: args.top_heads]
        sel = np.array([[int(i // H), int(i % H)] for i in flat])

        tids, poss, vals = [], [], []
        for tid, r, S in rows:
            v = r[sel[:, 0], sel[:, 1], :].mean(axis=0)
            for j in range(S):
                tids.append(tid); poss.append((j + 1) / S); vals.append(float(v[j]))
        res = position_only_baseline(np.array(tids), np.array(poss), np.array(vals))

        # How much of these heads' attention mass sits within one sentence of the
        # diagonal? High means the "receiver" heads are really recency heads.
        diag = []
        for tid, M in mats.items():
            A = M[sel[:, 0], sel[:, 1]].mean(axis=0)
            S = A.shape[0]
            band = sum(A[i, max(i - 1, 0):i + 1].sum() for i in range(S))
            diag.append(band / max(A.sum(), 1e-9))
        diag_share = float(np.mean(diag))

        heads = ", ".join(f"L{l}H{h}" for l, h in sel[:3])
        print(f"{gap:>4} {res.spearman:>+8.3f} {res.r2_oos:>+8.3f} "
              f"{res.topk_agreement.get(3, float('nan')):>6.2f} "
              f"{res.topk_chance.get(3, float('nan')):>7.3f}  {diag_share:>10.2f}  {heads}")
        out[str(gap)] = {
            "spearman_oos": res.spearman, "r2_oos": res.r2_oos,
            "topk_agreement": {str(k): v for k, v in res.topk_agreement.items()},
            "topk_chance": {str(k): v for k, v in res.topk_chance.items()},
            "near_diagonal_share": diag_share,
            "top_heads": [[int(a), int(b)] for a, b in sel[:10]],
        }

    (C.DATA / f"mingap_sweep_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote mingap_sweep_{tag}.json")


if __name__ == "__main__":
    main()
