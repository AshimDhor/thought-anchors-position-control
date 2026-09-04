from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from anchors import config as C
from anchors.rollouts import Engine

# Reuse the exact filler construction from the sweep rather than a copy of it.
_spec = importlib.util.spec_from_file_location(
    "resample", Path(__file__).with_name("02_resample.py"))
_resample = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_resample)

# A sentence from a different universe entirely: the scale reference for what
# "off-distribution" looks like here.
ALIEN = "The Baltic herring fishery reported record landings last quarter."


def score_continuations(eng: Engine, prompt: str, continuations: list[str]) -> list[float]:
    """Mean token log-probability of each continuation given ``prompt``.

    Uses prompt logprobs, so nothing is sampled: this is a pure prefill.
    """
    from vllm import SamplingParams

    tok = eng.tokenizer
    n_prompt = len(tok(prompt)["input_ids"])
    full = [prompt + c for c in continuations]
    params = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0)
    outs = eng.llm.generate(full, params)

    scores = []
    for out in outs:
        lps = out.prompt_logprobs or []
        tail = lps[n_prompt:]
        vals = []
        for entry in tail:
            if not entry:
                continue
            # prompt_logprobs=0 returns only the realised token.
            vals.append(next(iter(entry.values())).logprob)
        scores.append(float(np.mean(vals)) if vals else float("nan"))
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    ap.add_argument("--n-slots", type=int, default=120)
    ap.add_argument("--gpu-frac", type=float, default=0.85)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]

    traces = json.loads((C.DATA / f"traces_{tag}.json").read_text())
    rng = np.random.default_rng(C.SEED)

    slots = []
    for t in traces:
        for s in t["sentences"]:
            slots.append((t, s))
    pick = rng.choice(len(slots), size=min(args.n_slots, len(slots)), replace=False)

    eng = Engine(args.model, gpu_memory_utilization=args.gpu_frac,
                 max_model_len=C.MAX_MODEL_LEN)

    rows = []
    for k in pick:
        t, s = slots[int(k)]
        head = t["thinking"][: s["start"]]
        raw = t["thinking"][s["start"] : s["end"]]
        lead = raw[: len(raw) - len(raw.lstrip())]
        prompt = eng.rollout_prompt(t["problem"], C.INSTRUCTION, head)
        real = lead + raw.strip()
        filler = lead + _resample.matched_filler(s["text"], s["index"])
        alien = lead + ALIEN
        lp_real, lp_fill, lp_alien = score_continuations(eng, prompt, [real, filler, alien])
        rows.append({"trace_id": t["trace_id"], "i": s["index"],
                     "position": (s["index"] + 1) / t["n_sentences"],
                     "logp_real": lp_real, "logp_filler": lp_fill,
                     "logp_alien": lp_alien})

    real = np.array([r["logp_real"] for r in rows])
    fill = np.array([r["logp_filler"] for r in rows])
    alien = np.array([r["logp_alien"] for r in rows])

    out = {
        "model": args.model, "n_slots": len(rows),
        "mean_logp_real": float(np.nanmean(real)),
        "mean_logp_filler": float(np.nanmean(fill)),
        "mean_logp_alien": float(np.nanmean(alien)),
        "sd_logp_real_across_sentences": float(np.nanstd(real)),
        "gap_filler_minus_real": float(np.nanmean(fill - real)),
        "gap_alien_minus_real": float(np.nanmean(alien - real)),
        # How far the filler sits from the model's own sentences, measured in
        # units of the spread among those sentences.
        "filler_gap_in_sd": float(np.nanmean(fill - real) / np.nanstd(real)),
        "alien_gap_in_sd": float(np.nanmean(alien - real) / np.nanstd(real)),
        "frac_filler_more_likely_than_real": float(np.nanmean(fill > real)),
        "rows": rows,
    }
    (C.DATA / f"filler_indistribution_{tag}.json").write_text(json.dumps(out, indent=2))

    print(f"slots scored: {len(rows)}")
    print(f"mean log p/token  real {out['mean_logp_real']:+.3f}  "
          f"filler {out['mean_logp_filler']:+.3f}  alien {out['mean_logp_alien']:+.3f}")
    print(f"spread across the model's own sentences (sd): "
          f"{out['sd_logp_real_across_sentences']:.3f}")
    print(f"filler is {abs(out['gap_filler_minus_real']):.3f} nats/token worse than real "
          f"({out['filler_gap_in_sd']:+.2f} sd)")
    print(f"alien  is {abs(out['gap_alien_minus_real']):.3f} nats/token worse than real "
          f"({out['alien_gap_in_sd']:+.2f} sd)   <- what an intrusion looks like")
    print(f"filler more likely than the real sentence in "
          f"{out['frac_filler_more_likely_than_real']:.0%} of slots")


if __name__ == "__main__":
    main()
