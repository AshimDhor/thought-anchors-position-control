from __future__ import annotations

import argparse
import json
from collections import Counter

from datasets import load_dataset

from anchors import config as C
from anchors.answers import final_answer, normalise
from anchors.rollouts import Engine, GenConfig, split_thinking


def build_pool(limit_math: int, include_aime: bool) -> list[dict]:
    pool: list[dict] = []
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    hard = [r for r in math500 if r["level"] >= 3]
    for r in hard[:limit_math]:
        pool.append(
            {
                "pid": r["unique_id"],
                "source": "MATH-500",
                "level": r["level"],
                "subject": r["subject"],
                "problem": r["problem"],
                "gold": normalise(r["answer"]),
            }
        )
    if include_aime:
        for name, split in [("MathArena/aime_2025", "train"), ("MathArena/aime_2026", "train")]:
            try:
                ds = load_dataset(name, split=split)
            except Exception as exc:  # dataset layout varies; never fail the run
                print(f"  [skip] {name}: {exc}")
                continue
            for i, r in enumerate(ds):
                q = r.get("problem") or r.get("question")
                a = r.get("answer")
                if q is None or a is None:
                    continue
                pool.append(
                    {
                        "pid": f"{name}/{i}",
                        "source": name,
                        "level": 6,
                        "subject": "AIME",
                        "problem": q,
                        "gold": normalise(str(a)),
                    }
                )
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    ap.add_argument("--limit-math", type=int, default=400)
    ap.add_argument("--no-aime", action="store_true")
    ap.add_argument("--samples", type=int, default=C.SCREEN_SAMPLES)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()

    pool = build_pool(args.limit_math, not args.no_aime)
    print(f"pool: {len(pool)} problems")

    eng = Engine(args.model, gpu_memory_utilization=args.gpu_frac,
                 max_model_len=C.MAX_MODEL_LEN)
    prompts = [eng.chat_prefix(p["problem"], C.INSTRUCTION) for p in pool]
    cfg = GenConfig(temperature=C.TEMPERATURE, top_p=C.TOP_P,
                    max_tokens=C.MAX_NEW_TOKENS)
    completions = eng.generate(prompts, cfg, n=args.samples)

    rows = []
    for p, comps in zip(pool, completions):
        answers, unfinished = [], 0
        for c in comps:
            think, tail = split_thinking(c)
            if not tail:
                unfinished += 1
            answers.append(final_answer(c))
        counts = Counter(a if a is not None else "<none>" for a in answers)
        n = len(answers)
        rows.append(
            {
                **p,
                "n_samples": n,
                "pass_rate": counts[p["gold"]] / n,
                "no_answer_rate": unfinished / n,
                "n_distinct": len([k for k in counts if k != "<none>"]),
                "modal_answer": counts.most_common(1)[0][0],
                "answer_counts": dict(counts),
            }
        )

    out = C.DATA / f"screen_{args.model.split('/')[-1]}.json"
    out.write_text(json.dumps(rows, indent=2))
    lo, hi = C.DIFFICULTY_BAND
    keep = [r for r in rows if lo <= r["pass_rate"] <= hi and r["no_answer_rate"] < 0.5]
    print(f"wrote {out}")
    print(f"in band [{lo},{hi}] and mostly finishing: {len(keep)} / {len(rows)}")
    print(f"mean pass rate over pool: {sum(r['pass_rate'] for r in rows)/len(rows):.3f}")


if __name__ == "__main__":
    main()
