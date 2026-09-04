from __future__ import annotations

import argparse, json, sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, ".")
from anchors import config as C
from anchors.rollouts import split_thinking
from anchors.answers import final_answer, normalise

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=C.PRIMARY_MODEL)
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--max-tokens", type=int, default=6144)
ap.add_argument("--level", type=int, default=4)
a = ap.parse_args()

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
probs = [r for r in ds if r["level"] == a.level][: a.n]

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                            device_map="cuda:0").eval()
out = []
for r in probs:
    msgs = [{"role": "user", "content": f"{r['problem']}\n\n{C.INSTRUCTION}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        g = model.generate(**enc, do_sample=True, temperature=C.TEMPERATURE,
                           top_p=C.TOP_P, max_new_tokens=a.max_tokens,
                           pad_token_id=tok.eos_token_id)
    comp = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    think, tail = split_thinking(comp)
    n_tok = int(g.shape[1] - enc["input_ids"].shape[1])
    out.append({"problem": r["problem"], "level": r["level"],
                "gold": normalise(r["answer"]), "thinking": think, "tail": tail,
                "answer": final_answer(comp), "closed": bool(tail), "n_tokens": n_tok})
    print(f"  tokens={n_tok:5d} closed={bool(tail)!s:5s} chars={len(think):6d} "
          f"answer={final_answer(comp)!r} gold={normalise(r['answer'])!r}")

tag = a.model.split("/")[-1]
p = C.DATA / f"sample_traces_{tag}.json"
p.write_text(json.dumps(out, indent=2))
print(f"wrote {p}")
