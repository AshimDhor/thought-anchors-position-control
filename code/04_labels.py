from __future__ import annotations

import argparse
import json
import re

import pandas as pd

from anchors import config as C
from anchors.rollouts import Engine, GenConfig


def build_prompt(eng: Engine, context: str, target: str) -> str:
    user = (
        f"{RUBRIC}\n\nPreceding context (may be empty):\n{context}\n\n"
        f"TARGET sentence:\n{target}\n\nCategory:"
    )
    msgs = [{"role": "user", "content": user}]

    try:
        return eng.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return eng.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def parse(out: str) -> str:
    low = out.lower()
    # Models that think first bury the answer; take the last category mentioned.
    hits = [c for c in CATEGORIES if c in low]
    if hits:
        return max(hits, key=lambda c: low.rfind(c))
    compact = re.sub(r"[^a-z_]", "", low)
    for c in CATEGORIES:
        if c.replace("_", "") in compact:
            return c
    return "unparsed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", default=C.PRIMARY_MODEL,
                    help="whose sentences we are labelling")
    ap.add_argument("--labeller", default="Qwen/Qwen3-8B")
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    ap.add_argument("--context-sentences", type=int, default=2)
    args = ap.parse_args()

    tag = args.target_model.split("/")[-1]
    traces = {t["trace_id"]: t for t in json.loads((C.DATA / f"traces_{tag}.json").read_text())}
    df = pd.read_csv(C.DATA / f"sentences_{tag}.csv")

    eng = Engine(args.labeller, gpu_memory_utilization=args.gpu_frac, max_model_len=4096)
    prompts = []
    for _, r in df.iterrows():
        sents = traces[r.trace_id]["sentences"]
        lo = max(0, int(r["index"]) - args.context_sentences)
        context = " ".join(s["text"] for s in sents[lo : int(r["index"])])
        prompts.append(build_prompt(eng, context, str(r["text"])))

    outs = eng.generate(prompts, GenConfig(temperature=0.0, top_p=1.0, max_tokens=768), n=1)
    labels = [parse(o[0]) for o in outs]

    col = f"label_{args.labeller.split('/')[-1]}"
    df[col] = labels
    df.to_csv(C.DATA / f"sentences_{tag}.csv", index=False)
    print(f"wrote labels to column {col}")
    print(df[col].value_counts().to_string())
    print(f"unparsed: {(df[col] == 'unparsed').mean():.2%}")

    # If two labellers have been run, report how often they agree.  A category
    # story is only as good as the labels underneath it.
    label_cols = [c for c in df.columns if c.startswith("label_")]
    if len(label_cols) >= 2:
        a, b = label_cols[-2], label_cols[-1]
        both = df[(df[a] != "unparsed") & (df[b] != "unparsed")]
        if len(both):
            print(f"labeller agreement ({a} vs {b}): "
                  f"{(both[a] == both[b]).mean():.1%} over {len(both)} sentences")


if __name__ == "__main__":
    main()
