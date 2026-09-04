"""Diagnostic: how long are this model's traces, and when does it finish?

Why this exists.  The difficulty screen was supposed to find problems the model
is genuinely uncertain about.  It found instead that pass rate and truncation
rate correlate at -0.90: at a 3584-token budget, Olmo-3-7B-Think either finishes
and is right, or runs out of budget and scores zero.  The screen was measuring
the token cap, not difficulty.

Sweep cost is quadratic in trace length -- a rollout from prefix i has to write
the rest of the trace -- so "just raise the budget" is not free.  Before
choosing a model, a budget, and a prompt, measure the joint distribution of
(trace length, finished, correct) for each candidate.  That is what this does.
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
from collections import Counter
from datasets import load_dataset

from anchors import config as C
from anchors.answers import final_answer, normalise
from anchors.rollouts import Engine, GenConfig, split_thinking
from anchors.splitting import split_sentences

# A brevity nudge, tested as a separate arm.  If it shortens traces without
# flattening the answer distribution, it buys the whole study; if it makes the
# model terse and confident, it costs the variance the measure needs.
CONCISE = (" Keep your reasoning focused and avoid restating work you have "
           "already done.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    ap.add_argument("--n-problems", type=int, default=60)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--levels", default="4,5")
    ap.add_argument("--concise", action="store_true")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    # Parse any digits found. sbatch --export treats commas as separators
    # between exported variables, so LEVELS="4,5" silently arrived as "4" and
    # a probe I thought covered levels 4-5 actually covered level 3 only.
    levels = {int(x) for x in re.findall(r"\d", args.levels)}
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    probs = [r for r in ds if r["level"] in levels][: args.n_problems]
    print(f"{len(probs)} problems, levels {sorted(levels)}, "
          f"{args.samples} samples, budget {args.max_tokens}")

    instr = C.INSTRUCTION + (CONCISE if args.concise else "")
    eng = Engine(args.model, gpu_memory_utilization=args.gpu_frac,
                 max_model_len=args.max_model_len)
    prompts = [eng.chat_prefix(p["problem"], instr) for p in probs]
    outs = eng.generate(prompts, GenConfig(temperature=C.TEMPERATURE, top_p=C.TOP_P,
                                           max_tokens=args.max_tokens), n=args.samples)

    tok = eng.tokenizer
    rows = []
    for p, comps in zip(probs, outs):
        gold = normalise(p["answer"])
        recs = []
        for c in comps:
            think, tail = split_thinking(c)
            ans = final_answer(c)
            recs.append({
                "closed": bool(tail),
                "n_tok": len(tok(think, add_special_tokens=False)["input_ids"]),
                "n_sent": len(split_sentences(think)),
                "answer": ans,
                "correct": ans == gold,
            })
        # What the estimator actually needs is dispersion in the answer
        # distribution, not a particular accuracy. A problem where the model
        # gives six different wrong answers has plenty of room for a sentence to
        # move things; one where it confidently repeats the same wrong answer has
        # none. Pass rate is only a proxy for this, and a lossy one, so record
        # the distribution itself.
        finished = [r["answer"] for r in recs if r["answer"] is not None]
        cnt = Counter(finished)
        if finished:
            q = np.array(list(cnt.values()), dtype=float) / len(finished)
            entropy = float(-(q * np.log(q)).sum())
        else:
            entropy = 0.0
        rows.append({"pid": p["unique_id"], "level": p["level"], "gold": gold,
                     "n_finished": len(finished), "n_distinct": len(cnt),
                     "answer_entropy": entropy,
                     "answer_counts": dict(cnt),
                     "samples": recs,
                     "closed_rate": float(np.mean([r["closed"] for r in recs])),
                     "pass_rate": float(np.mean([r["correct"] for r in recs])),
                     "median_tok": float(np.median([r["n_tok"] for r in recs])),
                     "median_sent": float(np.median([r["n_sent"] for r in recs]))})

    tag = args.model.split("/")[-1] + ("_concise" if args.concise else "")
    (C.DATA / f"lengthprobe_{tag}.json").write_text(json.dumps(rows, indent=2))

    toks = np.array([r["n_tok"] for row in rows for r in row["samples"]])
    closed = np.array([r["closed"] for row in rows for r in row["samples"]])
    sents = np.array([r["n_sent"] for row in rows for r in row["samples"]])
    pr = np.array([r["pass_rate"] for r in rows])
    cr = np.array([r["closed_rate"] for r in rows])

    print(f"\n== {tag} ==")
    print(f"closed </think> within budget : {closed.mean():.1%}")
    print(f"thinking tokens  : p10 {np.percentile(toks,10):.0f}  median "
          f"{np.median(toks):.0f}  p90 {np.percentile(toks,90):.0f}  max {toks.max():.0f}")
    print(f"sentences        : median {np.median(sents):.0f}  p90 {np.percentile(sents,90):.0f}")
    print(f"pass rate        : mean {pr.mean():.2f}")
    ent = np.array([r["answer_entropy"] for r in rows])
    nd = np.array([r["n_distinct"] for r in rows])
    print(f"answer entropy   : median {np.median(ent):.2f}  "
          f"frac with >=2 distinct answers {np.mean(nd >= 2):.0%}  "
          f">=3 distinct {np.mean(nd >= 3):.0%}")

    print("\n-- selection on DISPERSION (what the estimator needs) --")
    for min_ent, min_close in [(0.4, 0.8), (0.4, 0.9), (0.6, 0.8)]:
        m = (ent >= min_ent) & (cr >= min_close)
        print(f"  entropy>={min_ent}, closing>={min_close:.0%}: {m.sum()} / {len(rows)}")
        if m.any():
            mt = np.array([r["median_tok"] for r, k in zip(rows, m) if k])
            ms = np.array([r["median_sent"] for r, k in zip(rows, m) if k])
            print(f"      traces: median {np.median(mt):.0f} tok, {np.median(ms):.0f} sent; "
                  f"p25 {np.percentile(mt,25):.0f} tok")
            n_use = min(10, m.sum())
            short = np.sort(mt)[:n_use]
            cost = 55 * 32 * (short / 2 + 120)
            print(f"      windowed sweep, {n_use} shortest: "
                  f"~{cost.sum()/1e6:.1f}M tok (~{cost.sum()/3000/3600:.1f} h main arm)")

    print("\n-- selection on PASS RATE (the paper's proxy) --")
    for lo, hi in [(0.15, 0.85), (0.25, 0.75)]:
        m = (pr >= lo) & (pr <= hi) & (cr >= 0.8)
        print(f"  in [{lo},{hi}] and closing >=80%: {m.sum()} / {len(rows)}")
        if m.any():
            mt = np.array([r["median_tok"] for r, k in zip(rows, m) if k])
            ms = np.array([r["median_sent"] for r, k in zip(rows, m) if k])
            print(f"      their traces: median {np.median(mt):.0f} tokens, "
                  f"{np.median(ms):.0f} sentences")
            # The number that decides feasibility.
            cost = np.sum((ms + 1) * C.ROLLOUTS_PER_PREFIX * (mt / 2 + 120))
            print(f"      full sweep over those {m.sum()} traces: "
                  f"~{cost/1e6:.0f}M tokens (~{cost/3000/3600:.1f} h at 3000 tok/s)")


if __name__ == "__main__":
    main()
