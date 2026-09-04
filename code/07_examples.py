from __future__ import annotations

import argparse
import json
import textwrap

import numpy as np
import pandas as pd

from anchors import config as C


def wrap(s: str, width: int = 96, indent: str = "      ") -> str:
    return textwrap.fill(" ".join(str(s).split()), width=width,
                         initial_indent=indent, subsequent_indent=indent)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    rng = np.random.default_rng(C.SEED)

    traces = json.loads((C.DATA / f"traces_{tag}.json").read_text())
    df = pd.read_csv(C.DATA / f"sentences_{tag}.csv")
    out = []

    out.append("# Randomly selected raw examples\n")
    out.append(f"Model: {args.model}. Seed: {C.SEED}. "
               "Every example below was drawn uniformly at random, not chosen.\n")

    out.append("\n## 1. Sentence segmentation\n")
    out.append("One randomly chosen trace, its first 12 sentences as the splitter cut them.\n")
    t = traces[int(rng.integers(len(traces)))]
    out.append(f"**Problem** ({t['subject']}, level {t['level']}, "
               f"screen pass rate {t['screen_pass_rate']:.2f}):\n")
    out.append(wrap(t["problem"], indent="  ") + "\n")
    out.append(f"**Gold answer**: `{t['gold']}`  |  "
               f"**base trace answer**: `{t['base_answer']}` "
               f"({'correct' if t['base_correct'] else 'incorrect'})\n")
    for s in t["sentences"][:12]:
        out.append(f"  [{s['index']:2d}] {' '.join(s['text'].split())[:200]}")
    out.append("")

    out.append("\n## 2. Randomly selected sentences with their measured importance\n")
    pick = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
    label_col = next((c for c in df.columns if c.startswith("label_")), None)
    for k in sorted(pick):
        r = df.iloc[int(k)]
        bits = [f"pos {r.position:.2f}", f"KL {r.kl_resampling:.3f}",
                f"floor {r.kl_null:.3f}"]
        if "filler_kl" in df and pd.notna(r.get("filler_kl")):
            bits.append(f"filler KL {r.filler_kl:.3f}")
        if label_col:
            bits.append(str(r[label_col]))
        out.append(f"- ({', '.join(bits)})")
        out.append(wrap(r.text))
    out.append("")

    out.append("\n## 3. Randomly selected rollout continuations\n")
    out.append("Two continuations from one randomly chosen prefix, to show what a "
               "rollout actually looks like and that the answer extractor is reading "
               "the right thing.\n")
    path = C.DATA / f"rollouts_main_{tag}.json"
    if path.exists():
        recs = json.loads(path.read_text())
        rec = recs[int(rng.integers(len(recs)))]
        tr = next(x for x in traces if x["trace_id"] == rec["trace_id"])
        out.append(f"Trace `{rec['trace_id']}`, prefix ends after sentence {rec['i']} "
                   f"of {tr['n_sentences']}.\n")
        if rec["i"] >= 0:
            out.append("**Prefix ends with**:")
            out.append(wrap(tr["sentences"][rec["i"]]["text"]))
        for j, comp in enumerate(rec["sample_completions"][:2]):
            out.append(f"\n**Continuation {j + 1}** "
                       f"(extracted answer: `{rec['answers'][j]}`, gold `{tr['gold']}`):")
            out.append(wrap(comp[:1200] + ("..." if len(comp) > 1200 else "")))
        out.append("")

    if label_col:
        out.append(f"\n## 4. Randomly selected category labels ({label_col})\n")
        pick = rng.choice(len(df), size=min(20, len(df)), replace=False)
        for k in sorted(pick):
            r = df.iloc[int(k)]
            out.append(f"- **{r[label_col]}** — {' '.join(str(r.text).split())[:180]}")
        out.append("")

    text = "\n".join(out)
    dest = C.ROOT / "writeup" / "random_examples.md"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(text)
    print(text[:3000])
    print(f"\n[wrote {dest}]")


if __name__ == "__main__":
    main()
