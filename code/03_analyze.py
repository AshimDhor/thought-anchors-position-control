from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from anchors import config as C
from anchors.baselines import bootstrap_ci_by_trace, position_only_baseline
from anchors.importance import _probs, kl, prefix_stats, sentence_importances, tv as tvdist

MEASURES = ["kl_resampling", "kl_counterfactual", "kl_similar", "kl_corrected",
            "tv", "abs_delta_acc"]


def load_arm(tag: str, arm: str) -> dict[tuple[str, int], dict]:
    path = C.DATA / f"rollouts_{arm}_{tag}.json"
    if not path.exists():
        print(f"  [warn] missing {path}")
        return {}
    return {(r["trace_id"], r["i"]): r for r in json.loads(path.read_text())}


def dissimilarity_masks(traces, main_arm) -> dict[tuple[str, int], list[bool]]:

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(C.EMBED_MODEL, device="cpu")
    pairs, keys = [], []
    for t in traces:
        tid, sents = t["trace_id"], t["sentences"]
        for i in range(-1, t["n_sentences"] - 1):
            rec = main_arm.get((tid, i))
            if rec is None or "first_sentences" not in rec:
                continue
            original = sents[i + 1]["text"]
            keys.append((tid, i, len(rec["first_sentences"])))
            pairs.append((original, rec["first_sentences"]))

    originals = [o for o, _ in pairs]
    flat = [f for _, fs in pairs for f in fs]
    if not flat:
        return {}
    emb_o = model.encode(originals, normalize_embeddings=True, batch_size=256,
                         show_progress_bar=False)
    emb_f = model.encode(flat, normalize_embeddings=True, batch_size=256,
                         show_progress_bar=False)

    masks, cursor = {}, 0
    for (tid, i, n_roll), eo in zip(keys, emb_o):
        block = emb_f[cursor : cursor + n_roll]
        cursor += n_roll
        sims = block @ eo
        masks[(tid, i)] = [bool(s < C.SIM_THRESHOLD) for s in sims]
    return masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]

    traces = json.loads((C.DATA / f"traces_{tag}.json").read_text())
    main_arm = load_arm(tag, "main")
    filler_arm = load_arm(tag, "filler")
    print(f"{len(traces)} traces | main prefixes {len(main_arm)} | filler {len(filler_arm)}")

    print("computing semantic-dissimilarity masks ...")
    masks = dissimilarity_masks(traces, main_arm)
    frac_diss = np.mean([np.mean(m) for m in masks.values()]) if masks else float("nan")
    print(f"  {frac_diss:.1%} of resampled first sentences count as 'different'")

    rows = []
    for t in traces:
        tid, gold, n = t["trace_id"], t["gold"], t["n_sentences"]
        stats = {}
        for i in range(-1, n):
            rec = main_arm.get((tid, i))
            if rec is None:
                continue
            stats[i] = prefix_stats(i, rec["answers"], gold, masks.get((tid, i)))

        for imp in sentence_importances(stats, n, gold, seed=C.SEED):
            i = imp.index
            row = {
                "trace_id": tid, "pid": t["pid"], "subject": t["subject"],
                "level": t["level"], "n_sentences": n,
                "base_correct": t["base_correct"],
                "screen_pass_rate": t["screen_pass_rate"],
                "index": i, "position": imp.position,
                "text": t["sentences"][i]["text"],
                "n_chars": len(t["sentences"][i]["text"]),
                **{k: getattr(imp, k) for k in (
                    "kl_resampling", "kl_counterfactual", "kl_null", "kl_corrected",
                    "tv", "tv_null", "tv_corrected", "delta_acc", "abs_delta_acc",
                    "entropy_before", "entropy_after", "kl_similar",
                    "p_correct_before", "p_correct_after",
                    "n_dissimilar", "n_similar")},
            }
            rec_f, before = filler_arm.get((tid, i)), stats.get(i - 1)
            if rec_f is not None and before is not None:
                fs = prefix_stats(i, rec_f["answers"], gold)
                sup = sorted(set(before.counts) | set(fs.counts))
                row["filler_kl"] = kl(_probs(before.counts, sup, 0.5),
                                      _probs(fs.counts, sup, 0.5))
                row["filler_tv"] = tvdist(_probs(before.counts, sup),
                                          _probs(fs.counts, sup))
                row["filler_p_correct"] = fs.p_correct
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(C.DATA / f"sentences_{tag}.csv", index=False)
    print(f"wrote sentences_{tag}.csv  ({len(df)} sentences)")

    summary: dict = {
        "model": args.model,
        "n_traces": int(df.trace_id.nunique()),
        "n_sentences": int(len(df)),
        "rollouts_per_prefix": int(np.median(
            [len(r["answers"]) for r in main_arm.values()])),
        "rollouts_per_prefix_config": C.ROLLOUTS_PER_PREFIX,
        "frac_resamples_dissimilar": float(frac_diss),
    }

    for m in MEASURES:
        sub = df.dropna(subset=[m])
        if sub.empty:
            continue
        res = position_only_baseline(sub.trace_id.values, sub.position.values,
                                     sub[m].values.astype(float))
        summary[m] = {
            "n": int(len(sub)),
            "spearman_oos": res.spearman, "spearman_p": res.spearman_p,
            "r2_oos": res.r2_oos,
            "topk_agreement": {str(k): v for k, v in res.topk_agreement.items()},
            "topk_chance": {str(k): v for k, v in res.topk_chance.items()},
            "curve_x": res.curve_x.tolist(), "curve_y": res.curve_y.tolist(),
            "mean": float(sub[m].mean()),
        }
        print(f"[{m:18s}] position-only  rho={res.spearman:+.3f} (p={res.spearman_p:.1e})  "
              f"R2_oos={res.r2_oos:+.3f}  top1={res.topk_agreement.get(1, float('nan')):.2f} "
              f"top3={res.topk_agreement.get(3, float('nan')):.2f} "
              f"(chance {res.topk_chance.get(3, float('nan')):.2f})")

    summary["noise_floor"] = {
        "mean_kl": float(df.kl_resampling.mean()),
        "mean_kl_null": float(df.kl_null.mean()),
        "frac_kl_above_null": float((df.kl_resampling > df.kl_null).mean()),
        "mean_tv": float(df.tv.mean()), "mean_tv_null": float(df.tv_null.mean()),
    }
    print("noise floor: mean KL {mean_kl:.3f} vs null {mean_kl_null:.3f}; "
          "{frac_kl_above_null:.1%} of sentences above floor".format(**summary["noise_floor"]))

    if "filler_kl" in df and df.filler_kl.notna().any():
        sub = df.dropna(subset=["filler_kl"])
        pt, lo, hi = bootstrap_ci_by_trace(
            sub.trace_id.values, (sub.kl_resampling - sub.filler_kl).values.astype(float),
            statistic=lambda v: float(np.mean(v)), seed=C.SEED)
        summary["filler"] = {
            "n": int(len(sub)),
            "mean_real_kl": float(sub.kl_resampling.mean()),
            "mean_filler_kl": float(sub.filler_kl.mean()),
            "mean_gap": pt, "gap_ci": [lo, hi],
            "frac_real_exceeds_filler": float((sub.kl_resampling > sub.filler_kl).mean()),
        }
        print(f"filler control: real KL {sub.kl_resampling.mean():.3f} vs filler "
              f"{sub.filler_kl.mean():.3f}; gap {pt:+.3f} [{lo:+.3f},{hi:+.3f}]; "
              f"real > filler in {summary['filler']['frac_real_exceeds_filler']:.0%} of slots")

    # The paraphrase contrast: does the semantic filter that defines
    # counterfactual importance actually change the answer?
    both = df.dropna(subset=["kl_counterfactual", "kl_similar"])
    if not both.empty:
        pt, lo, hi = bootstrap_ci_by_trace(
            both.trace_id.values,
            (both.kl_counterfactual - both.kl_similar).values.astype(float),
            statistic=lambda v: float(np.mean(v)), seed=C.SEED)
        summary["paraphrase_contrast"] = {
            "n": int(len(both)),
            "mean_kl_different": float(both.kl_counterfactual.mean()),
            "mean_kl_similar": float(both.kl_similar.mean()),
            "mean_gap": pt, "gap_ci": [lo, hi],
            "frac_different_exceeds_similar": float(
                (both.kl_counterfactual > both.kl_similar).mean()),
        }
        print(f"paraphrase contrast: semantically-different replacements KL "
              f"{both.kl_counterfactual.mean():.3f} vs semantically-similar "
              f"{both.kl_similar.mean():.3f}; gap {pt:+.3f} [{lo:+.3f},{hi:+.3f}]")


    no_ans_rate, kept_vals, kept_tid, kept_pos = [], [], [], []
    for t in traces:
        tid, n, gold = t["trace_id"], t["n_sentences"], t["gold"]
        st = {}
        for i in range(-1, n):
            rec = main_arm.get((tid, i))
            if rec is None:
                continue
            answers = rec["answers"]
            mask = masks.get((tid, i))
            keep = [(a, m) for a, m in zip(answers, mask or [True] * len(answers))
                    if a is not None]
            no_ans_rate.append(((i + 1) / n, 1.0 - len(keep) / max(len(answers), 1)))
            if not keep:
                continue
            st[i] = prefix_stats(i, [a for a, _ in keep], gold, [m for _, m in keep])
        for imp in sentence_importances(st, n, gold, seed=C.SEED):
            kept_vals.append(imp.kl_resampling)
            kept_tid.append(tid)
            kept_pos.append(imp.position)
    if kept_vals:
        r = position_only_baseline(np.array(kept_tid), np.array(kept_pos),
                                   np.array(kept_vals))
        pos_arr = np.array([p for p, _ in no_ans_rate])
        rate_arr = np.array([v for _, v in no_ans_rate])
        summary["drop_no_answer"] = {
            "n": len(kept_vals),
            "spearman_oos": r.spearman, "r2_oos": r.r2_oos,
            "top3": r.topk_agreement.get(3),
            "mean_kl": float(np.mean(kept_vals)),
            "no_answer_rate_overall": float(rate_arr.mean()),
            "no_answer_rate_vs_position_rho": float(
                pd.Series(pos_arr).corr(pd.Series(rate_arr), method="spearman")),
        }
        d = summary["drop_no_answer"]
        print(f"drop-<none> robustness: rho={d['spearman_oos']:+.3f} "
              f"R2={d['r2_oos']:+.3f} top3={d['top3']:.2f} "
              f"(overall no-answer rate {d['no_answer_rate_overall']:.1%}, "
              f"vs position rho={d['no_answer_rate_vs_position_rho']:+.3f})")

    ok = df.entropy_before.notna()
    summary["mechanism"] = {
        "entropy_vs_position_rho": float(df.position[ok].corr(df.entropy_before[ok], method="spearman")),
        "kl_vs_entropy_before_rho": float(df.kl_resampling[ok].corr(df.entropy_before[ok], method="spearman")),
        "kl_vs_entropy_drop_rho": float(
            df.kl_resampling[ok].corr((df.entropy_before - df.entropy_after)[ok], method="spearman")),
    }
    print("mechanism: entropy~position rho={entropy_vs_position_rho:+.3f}; "
          "KL~entropy_before rho={kl_vs_entropy_before_rho:+.3f}; "
          "KL~entropy_drop rho={kl_vs_entropy_drop_rho:+.3f}".format(**summary["mechanism"]))

    # Smoothing sensitivity: the KL numbers should not hinge on the prior.
    sens = {}
    for lam in (0.25, 0.5, 1.0):
        vals, tids, poss = [], [], []
        for t in traces:
            tid, n = t["trace_id"], t["n_sentences"]
            st = {i: prefix_stats(i, main_arm[(tid, i)]["answers"], t["gold"],
                                  masks.get((tid, i)))
                  for i in range(-1, n) if (tid, i) in main_arm}
            for imp in sentence_importances(st, n, t["gold"], seed=C.SEED, smoothing=lam):
                vals.append(imp.kl_resampling); tids.append(tid); poss.append(imp.position)
        r = position_only_baseline(np.array(tids), np.array(poss), np.array(vals))
        sens[str(lam)] = {"spearman_oos": r.spearman, "r2_oos": r.r2_oos,
                          "top3": r.topk_agreement.get(3)}
        print(f"  smoothing {lam}: rho={r.spearman:+.3f} R2={r.r2_oos:+.3f} "
              f"top3={r.topk_agreement.get(3):.2f}")
    summary["smoothing_sensitivity"] = sens

    (C.DATA / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote summary_{tag}.json")


if __name__ == "__main__":
    main()
