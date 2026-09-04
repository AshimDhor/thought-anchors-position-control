"""Stage 1: pick the in-band problems and generate one base trace for each.

The base trace is the object whose sentences we score.  We generate it with a
fixed seed and record it verbatim, because every prefix used later is a literal
slice of this string.
"""

from __future__ import annotations

import argparse
import json

from anchors import config as C
from anchors.answers import final_answer
from anchors.rollouts import Engine, GenConfig, split_thinking
from anchors.splitting import split_sentences


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    ap.add_argument("--n-problems", type=int, default=24)
    ap.add_argument("--traces-per-problem", type=int, default=1)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    tag = args.model.split("/")[-1]

    probe_path = C.DATA / f"lengthprobe_{tag}.json"
    if probe_path.exists():
        # The length probe already measured, per problem: answer-distribution
        # entropy, how reliably the model closes </think>, and trace length.
        # That is strictly more than the difficulty screen produced, so there is
        # no reason to spend another GPU-hour re-measuring it.
        probe = json.loads(probe_path.read_text())
        band = [r for r in probe
                if r["answer_entropy"] >= C.MIN_ANSWER_ENTROPY
                and r["closed_rate"] >= C.MIN_CLOSED_RATE]
        print(f"{len(probe)} problems probed; "
              f"{len(band)} pass entropy>={C.MIN_ANSWER_ENTROPY} "
              f"and closing>={C.MIN_CLOSED_RATE}")
        # Among qualifying problems, prefer the shortest traces. Sweep cost is
        # linear in trace length and the selection is on a nuisance variable,
        # not on anything the hypothesis is about -- but it does bias the trace
        # set toward shorter reasoning, which the write-up states.
        band.sort(key=lambda r: r["median_tok"])
        for r in band[: args.n_problems]:
            print(f"    {r['pid']:26s} L{r['level']} entropy {r['answer_entropy']:.2f} "
                  f"distinct {r['n_distinct']} closed {r['closed_rate']:.2f} "
                  f"{r['median_tok']:.0f} tok / {r['median_sent']:.0f} sent")
        import datasets
        text = {r["unique_id"]: r["problem"]
                for r in datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")}
        for r in band:
            r["problem"] = text[r["pid"]]
            r["subject"] = "MATH"
            r["source"] = "MATH-500"
            r["pass_rate"] = r["pass_rate"]
    else:
        screen = json.loads((C.DATA / f"screen_{tag}.json").read_text())
        lo, hi = C.DIFFICULTY_BAND
        band = [r for r in screen
                if lo <= r["pass_rate"] <= hi and r["no_answer_rate"] < 0.5]
        band.sort(key=lambda r: abs(r["pass_rate"] - 0.5))
        print(f"{len(band)} problems in band; taking up to {args.n_problems}")

    eng = Engine(args.model, gpu_memory_utilization=args.gpu_frac,
                 max_model_len=C.MAX_MODEL_LEN)
    cfg = GenConfig(temperature=C.TEMPERATURE, top_p=C.TOP_P,
                    max_tokens=C.MAX_NEW_TOKENS)

    # Over-generate.  Traces get rejected for running past the token budget
    # without closing </think>, or for falling outside the sentence-count window,
    # and on hard problems that rejection rate can be high.  Generating 6x the
    # target costs a few minutes and avoids coming up short.
    cands = band[: max(args.n_problems * 2, args.n_problems + 6)]
    prompts = [eng.chat_prefix(p["problem"], C.INSTRUCTION) for p in cands]
    outs = eng.generate(prompts, cfg, n=args.traces_per_problem)

    traces, rejected = [], {"unfinished": 0, "too_many_chars": 0,
                            "too_short": 0, "too_long": 0}
    for prob, comps in zip(cands, outs):
        for k, comp in enumerate(comps):
            think, tail = split_thinking(comp)
            if not tail:
                rejected["unfinished"] += 1
                continue
            if len(think) > C.MAX_THINKING_CHARS:
                rejected["too_many_chars"] += 1
                continue
            sents = split_sentences(think)
            if len(sents) < C.MIN_SENTENCES:
                rejected["too_short"] += 1
                continue
            if len(sents) > C.MAX_SENTENCES:
                rejected["too_long"] += 1
                continue
            traces.append(
                {
                    "trace_id": f"{prob['pid']}#{k}",
                    "pid": prob["pid"],
                    "source": prob["source"],
                    "level": prob["level"],
                    "subject": prob["subject"],
                    "problem": prob["problem"],
                    "gold": prob["gold"],
                    "screen_pass_rate": prob["pass_rate"],
                    "thinking": think,
                    "tail": tail,
                    "base_answer": final_answer(comp),
                    "base_correct": final_answer(comp) == prob["gold"],
                    "sentences": [
                        {"index": s.index, "start": s.start, "end": s.end, "text": s.text}
                        for s in sents
                    ],
                    "n_sentences": len(sents),
                }
            )
            if len(traces) >= args.n_problems * args.traces_per_problem:
                break
        if len(traces) >= args.n_problems * args.traces_per_problem:
            break

    out = C.DATA / f"traces_{tag}.json"
    out.write_text(json.dumps(traces, indent=2))
    n_sent = [t["n_sentences"] for t in traces]
    print(f"wrote {len(traces)} traces to {out}")
    print(f"  rejected: {rejected}  (of {len(cands)} candidates generated)")
    print(f"  NOTE: traces are selected on finishing within the token budget and on "
          f"having {C.MIN_SENTENCES}-{C.MAX_SENTENCES} sentences. Both are selection "
          f"effects on trace length and are stated as limitations.")
    print(f"  sentences per trace: min {min(n_sent)}, median {sorted(n_sent)[len(n_sent)//2]}, max {max(n_sent)}")
    print(f"  base trace correct: {sum(t['base_correct'] for t in traces)}/{len(traces)}")
    # Cost is reported for the sweep we actually run. With windowed sampling the
    # prefix count is roughly constant per trace rather than proportional to
    # sentence count, so quoting (n+1) here overstated it by ~4x.
    if C.N_WINDOWS > 0:
        n_prefix = sum(min(n + 1, C.N_WINDOWS * (C.WINDOW_LEN + 1) + 2) for n in n_sent)
    else:
        n_prefix = sum(n + 1 for n in n_sent)
    chars = [len(t["thinking"]) for t in traces]
    # Rough cost model: a rollout from prefix i writes the remainder of the
    # trace, so the mean rollout is about half a trace, plus the answer.
    per_trace_prefixes = (n_prefix / max(len(traces), 1))
    est_tokens = sum(per_trace_prefixes * C.ROLLOUTS_PER_PREFIX *
                     (len(t["thinking"]) / 4 / 2 + 120) for t in traces)
    print(f"  thinking chars: median {int(sorted(chars)[len(chars)//2])}, max {max(chars)}")
    print(f"  total prefix boundaries to sweep: {n_prefix}")
    print(f"  main arm: {n_prefix * C.ROLLOUTS_PER_PREFIX} rollouts, "
          f"~{est_tokens/1e6:.1f}M output tokens "
          f"(~{est_tokens/2900/60:.0f} min at 2900 tok/s)")
    print(f"  filler arm at {C.FILLER_SUBSAMPLE:.0%} of slots: "
          f"~{est_tokens * C.FILLER_SUBSAMPLE/1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
