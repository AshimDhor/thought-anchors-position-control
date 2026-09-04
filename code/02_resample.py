from __future__ import annotations

import argparse
import json
import re

import numpy as np

from anchors import config as C
from anchors.answers import final_answer
from anchors.rollouts import Engine, GenConfig, split_thinking
from anchors.splitting import split_sentences
ENDED_REASONING = "<ended reasoning and gave the answer>"


FILLERS = [
    "Let me think about this a bit more carefully.",
    "Hmm, let me continue.",
    "Okay, let me keep going.",
]


_MARKER = re.compile(r"^([-*+\u2022]\s+|\d{1,2}[.)]\s+)")


def matched_filler(original: str, k: int) -> str:
    base = FILLERS[k % len(FILLERS)]
    m = _MARKER.match(original)
    return (m.group(1) + base) if m else base


def sample_windows(n_sent: int, n_windows: int, window_len: int,
                   rng: np.random.Generator) -> list[int]:
    if n_sent <= n_windows * window_len:
        return list(range(-1, n_sent))

    edges = np.linspace(0, n_sent - window_len, n_windows + 1)
    starts = {0, n_sent - window_len}
    for k in range(n_windows):
        lo, hi = int(edges[k]), max(int(edges[k + 1]), int(edges[k]) + 1)
        starts.add(int(rng.integers(lo, hi)))
    starts = sorted(s for s in starts if 0 <= s <= n_sent - window_len)
    need: set[int] = set()
    for st in starts:
        for i in range(st - 1, min(st + window_len, n_sent)):
            need.add(i)
    return sorted(need)


def build_jobs(traces: list[dict], arm: str, subsample: float = 1.0,
               seed: int = 0, n_windows: int = 0, window_len: int = 4) -> list[dict]:
    rng = np.random.default_rng(seed)
    jobs = []
    for t in traces:
        sents = t["sentences"]
        if arm == "main":
            if n_windows > 0:
                idxs = sample_windows(len(sents), n_windows, window_len, rng)
            else:
                idxs = list(range(-1, len(sents)))
            for i in idxs:
                # i = -1 is the empty prefix: the model writes its own reasoning.
                jobs.append({"trace_id": t["trace_id"], "i": i,
                             "prefix": "" if i < 0 else t["thinking"][: sents[i]["end"]]})
        else:
            if n_windows > 0:
                # Only sentences whose i-1 prefix was also swept are measurable,
                # so the filler arm has to target that same set -- a filler at a
                # slot with no A_{i-1} to compare against is wasted compute.
                swept = set(sample_windows(len(sents), n_windows, window_len,
                                           np.random.default_rng(seed)))
                keep = np.array(sorted(i for i in swept if i >= 0 and (i - 1) in swept))
            else:
                keep = np.arange(len(sents))
            if subsample < 1.0 and len(keep):
                # Uniform over slots, so the filler arm stays spread across
                # positions rather than clustering anywhere.
                n_keep = max(1, int(round(subsample * len(keep))))
                keep = np.sort(rng.choice(keep, size=n_keep, replace=False))
            for k in keep:
                s = sents[int(k)]
                filler = matched_filler(s["text"], int(k))
                # Sentence spans tile the trace exactly, and the whitespace
                # separating a sentence from the one before it sits at the START
                # of the sentence's own span. So thinking[:s.start] ends flush
                # against the previous full stop, and concatenating the filler
                # straight on gives "...left-hand side.Let me think...". Carry
                # the original leading whitespace over so the filler arm differs
                # from the main arm in content and nothing else.
                raw = t["thinking"][s["start"] : s["end"]]
                lead = raw[: len(raw) - len(raw.lstrip())]
                jobs.append({"trace_id": t["trace_id"], "i": s["index"],
                             "prefix": t["thinking"][: s["start"]] + lead + filler,
                             "filler": filler})
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    ap.add_argument("--arm", choices=["main", "filler"], default="main")
    ap.add_argument("--rollouts", type=int, default=C.ROLLOUTS_PER_PREFIX)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=C.MAX_NEW_TOKENS)

    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-windows", type=int, default=C.N_WINDOWS,
                    help="0 = sweep every prefix; else sample this many windows")
    ap.add_argument("--window-len", type=int, default=C.WINDOW_LEN)

    ap.add_argument("--pass-id", type=int, default=1)
    args = ap.parse_args()

    tag = args.model.split("/")[-1]
    traces = json.loads((C.DATA / f"traces_{tag}.json").read_text())
    if args.n_shards > 1:
        traces = [t for k, t in enumerate(traces) if k % args.n_shards == args.shard]
        print(f"shard {args.shard}/{args.n_shards}: {len(traces)} traces")
    by_id = {t["trace_id"]: t for t in traces}
    sub = C.FILLER_SUBSAMPLE if args.arm == "filler" else 1.0
    jobs = build_jobs(traces, args.arm, sub, seed=C.SEED,
                      n_windows=args.n_windows, window_len=args.window_len)
    print(f"{args.arm}: {len(jobs)} prefixes x {args.rollouts} rollouts "
          f"= {len(jobs) * args.rollouts} sequences")

    eng = Engine(args.model, gpu_memory_utilization=args.gpu_frac,
                 max_model_len=C.MAX_MODEL_LEN)
    prompts = [
        eng.rollout_prompt(by_id[j["trace_id"]]["problem"], C.INSTRUCTION, j["prefix"])
        for j in jobs
    ]
    cfg = GenConfig(temperature=C.TEMPERATURE, top_p=C.TOP_P,
                    max_tokens=args.max_tokens)
    outs = eng.generate(prompts, cfg, n=args.rollouts)

    records = []
    for j, comps in zip(jobs, outs):
        answers = [final_answer(c) for c in comps]

        firsts = []
        for c in comps:

            think, _ = split_thinking(c)
            sents = split_sentences(think)
            firsts.append(sents[0].text if sents else ENDED_REASONING)
        records.append(
            {
                "trace_id": j["trace_id"],
                "i": j["i"],
                "filler": j.get("filler"),
                "answers": answers,
                "first_sentences": firsts,
                # Keep two completions per prefix so the write-up can show raw,
                # randomly-selected continuations rather than only summary stats.
                "sample_completions": comps[:2],
            }
        )

    suffix = "" if args.n_shards == 1 else f".shard{args.shard}of{args.n_shards}"
    if args.pass_id != 1:
        suffix += f".pass{args.pass_id}"
    out = C.DATA / f"rollouts_{args.arm}_{tag}{suffix}.json"
    out.write_text(json.dumps(records))
    n_none = sum(a is None for r in records for a in r["answers"])
    total = sum(len(r["answers"]) for r in records)
    print(f"wrote {out}  ({total} rollouts, {n_none/total:.1%} produced no boxed answer)")


if __name__ == "__main__":
    main()
