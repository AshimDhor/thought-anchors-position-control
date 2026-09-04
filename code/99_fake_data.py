from __future__ import annotations

import json

import numpy as np

from anchors import config as C
from anchors.simulate import SimConfig, simulate_trace
from anchors.splitting import split_sentences

TAG = "FAKE"

SENTENCE_POOL = [
    "Let me start by understanding what the problem is asking.",
    "I need to find the value of x that satisfies this equation.",
    "First I will expand the left-hand side.",
    "So we get x^2 + 4x + 4 on the left.",
    "Wait, that does not look right, let me redo that step.",
    "Actually the expansion should be x^2 + 4x + 4, which is what I had.",
    "Now I recall that the discriminant is b^2 - 4ac.",
    "Substituting gives 16 - 16 = 0.",
    "So there is exactly one real root.",
    "Let me double-check by plugging the value back in.",
    "That checks out.",
    "So the answer is 2.",
]


def main() -> None:
    rng = np.random.default_rng(0)
    cfg = SimConfig(n_sentences=24, n_rollouts=24, anchor_step=9)
    traces, main_recs, filler_recs = [], [], []

    for k in range(6):
        sim = simulate_trace(cfg, seed=100 + k)
        # Build a plausible thinking string out of the pool so that the splitter,
        # the example dumper and the label stage all have real text to chew on.
        parts = [SENTENCE_POOL[(k + j) % len(SENTENCE_POOL)] for j in range(cfg.n_sentences)]
        thinking = " ".join(parts)
        sents = split_sentences(thinking)
        # The pool may merge/split differently; trust the splitter and trim the
        # simulated rollouts to match, which is what the real pipeline does too.
        n = min(len(sents), cfg.n_sentences)
        tid = f"fake/{k}"
        traces.append({
            "trace_id": tid, "pid": tid, "source": "FAKE", "level": 5,
            "subject": "Algebra", "problem": f"Fake problem {k}: solve for x.",
            "gold": sim["gold"], "screen_pass_rate": 0.5,
            "thinking": thinking, "tail": " The answer is \\boxed{2}.",
            "base_answer": sim["gold"], "base_correct": True,
            "sentences": [{"index": s.index, "start": s.start, "end": s.end,
                           "text": s.text} for s in sents[:n]],
            "n_sentences": n,
        })
        for i in range(-1, n):
            ans = sim["main"].get(i, sim["main"][0])
            main_recs.append({
                "trace_id": tid, "i": i, "filler": None, "answers": ans,
                "first_sentences": [SENTENCE_POOL[int(rng.integers(len(SENTENCE_POOL)))]
                                    for _ in ans],
                "sample_completions": ["Continuing... the answer is \\boxed{2}."] * 2,
            })
        for i in range(n):
            ans = sim["filler"].get(i, sim["filler"][0])
            filler_recs.append({
                "trace_id": tid, "i": i, "filler": "Hmm, let me continue.",
                "answers": ans,
                "first_sentences": ["Hmm, let me continue."] * len(ans),
                "sample_completions": ["Continuing... \\boxed{2}."] * 2,
            })

    (C.DATA / f"traces_{TAG}.json").write_text(json.dumps(traces, indent=2))
    (C.DATA / f"rollouts_main_{TAG}.json").write_text(json.dumps(main_recs))
    (C.DATA / f"rollouts_filler_{TAG}.json").write_text(json.dumps(filler_recs))
    print(f"wrote fake data for tag {TAG}: {len(traces)} traces, "
          f"{len(main_recs)} main prefixes, {len(filler_recs)} filler prefixes")


if __name__ == "__main__":
    main()
