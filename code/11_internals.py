from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from anchors import config as C


def sentence_token_spans(tokenizer, chat_prefix: str, thinking: str,
                         sentences: list[dict]) -> tuple[list[range], dict]:

    full = chat_prefix + thinking
    enc = tokenizer(full, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    base = len(chat_prefix)

    spans = []
    for s in sentences:
        lo, hi = base + s["start"], base + s["end"]
        idx = [k for k, (a, b) in enumerate(offsets)
               if b > a and a >= lo and a < hi]
        spans.append(range(idx[0], idx[-1] + 1) if idx else range(0, 0))
    return spans, {"n_tokens": len(offsets), "input_ids": enc["input_ids"]}


def aggregate_to_sentences(attn: torch.Tensor, spans: list[range],
                           drop_sink: bool) -> torch.Tensor:
    """[H, T, T] token attention -> [H, S, S] sentence attention.

    ``M[h, i, j]`` is the mean over query tokens in sentence i of the total
    attention those tokens send to sentence j.  Averaging over queries (rather
    than summing) stops long sentences from looking important merely for being
    long -- a confound of exactly the kind this project is about.
    """
    H, T, _ = attn.shape
    if drop_sink:
        # The first token soaks up a large, content-free share of attention in
        # most models. Zeroing it and renormalising stops that constant from
        # dominating every head's profile.
        attn = attn.clone()
        attn[:, :, 0] = 0.0
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    S = len(spans)
    keyed = torch.zeros((H, T, S), device=attn.device, dtype=attn.dtype)
    for j, sp in enumerate(spans):
        if len(sp):
            keyed[:, :, j] = attn[:, :, sp.start : sp.stop].sum(dim=-1)
    out = torch.zeros((H, S, S), device=attn.device, dtype=attn.dtype)
    for i, sp in enumerate(spans):
        if len(sp):
            out[:, i, :] = keyed[:, sp.start : sp.stop, :].mean(dim=1)
    return out


def receiver_scores(M: np.ndarray, min_gap: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Per-head receiver score, and per-head per-sentence attention received.

    ``r[l, h, j]`` is the mean attention sentence j receives from sentences that
    come at least ``min_gap`` later.  A head's receiver score is how peaked that
    profile is: the ratio of its maximum to its mean.  A head that spreads its
    backward attention evenly scores 1; one that fixates on a single earlier
    sentence scores near S.
    """
    L, H, S, _ = M.shape
    r = np.zeros((L, H, S), dtype=np.float32)
    for j in range(S):
        rows = np.arange(j + min_gap, S)
        if len(rows) == 0:
            continue
        r[:, :, j] = M[:, :, rows, j].mean(axis=2)
    denom = r.mean(axis=2, keepdims=True)
    peaked = (r.max(axis=2) / np.squeeze(denom, axis=2).clip(1e-9))
    return peaked, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    ap.add_argument("--max-traces", type=int, default=20)
    ap.add_argument("--top-heads", type=int, default=20)
    ap.add_argument("--keep-sink", action="store_true",
                    help="do not zero the first-token attention sink")
    args = ap.parse_args()
    tag = args.model.split("/")[-1]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from anchors.rollouts import Engine  # only for chat_prefix formatting

    traces = json.loads((C.DATA / f"traces_{tag}.json").read_text())[: args.max_traces]
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"loading {args.model} with eager attention (vLLM cannot expose attention)")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager")
    model.eval()

    def chat_prefix(question: str) -> str:
        msgs = [{"role": "user", "content": f"{question}\n\n{C.INSTRUCTION}"}]
        base = tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        if "<think>" not in base[-64:]:
            base = base + "<think>\n"
        return base

    per_trace = []
    for t in traces:
        pre = chat_prefix(t["problem"])
        spans, meta = sentence_token_spans(tokenizer, pre, t["thinking"], t["sentences"])
        ok = [k for k, sp in enumerate(spans) if len(sp)]
        if len(ok) < C.MIN_SENTENCES:
            print(f"  [skip] {t['trace_id']}: only {len(ok)} sentences mapped to tokens")
            continue

        ids = torch.tensor([meta["input_ids"]], device=model.device)
        with torch.no_grad():
            out = model(ids, output_attentions=True, use_cache=False)

        mats = []
        for layer_attn in out.attentions:            # each [1, H, T, T]
            mats.append(aggregate_to_sentences(
                layer_attn[0].float(), spans, drop_sink=not args.keep_sink).cpu().numpy())
        M = np.stack(mats).astype(np.float16)        # [L, H, S, S]
        del out
        torch.cuda.empty_cache()

        np.save(C.DATA / f"attn_{tag}_{t['trace_id'].replace('/', '_')}.npy", M)
        per_trace.append({"trace_id": t["trace_id"], "n_sentences": len(spans),
                          "n_tokens": meta["n_tokens"],
                          "shape": list(M.shape)})
        print(f"  {t['trace_id']}: {meta['n_tokens']} tokens, "
              f"{len(spans)} sentences, attention {list(M.shape)}")

    if not per_trace:
        print("no traces processed")
        return

    # Pool the head profiles across traces to find consistently receiver-like heads.
    peaked_all, rows = [], []
    for rec in per_trace:
        M = np.load(C.DATA / f"attn_{tag}_{rec['trace_id'].replace('/', '_')}.npy").astype(np.float32)
        peaked, r = receiver_scores(M)
        peaked_all.append(peaked)
        rows.append((rec["trace_id"], r, rec["n_sentences"]))
    peaked_mean = np.mean(peaked_all, axis=0)                    # [L, H]

    L, H = peaked_mean.shape
    flat = np.argsort(-peaked_mean.ravel())[: args.top_heads]
    top = [(int(i // H), int(i % H), float(peaked_mean.ravel()[i])) for i in flat]
    print(f"\ntop {args.top_heads} receiver heads (layer, head, peakedness):")
    for l, h, v in top[:10]:
        print(f"   L{l:2d}H{h:2d}  {v:.2f}")

    # White-box per-sentence importance, from those heads only.
    sel = np.array([[l, h] for l, h, _ in top])
    tids, poss, vals = [], [], []
    for tid, r, n in rows:
        v = r[sel[:, 0], sel[:, 1], :].mean(axis=0)             # [S]
        for j in range(n):
            tids.append(tid); poss.append((j + 1) / n); vals.append(float(v[j]))

    from anchors.baselines import position_only_baseline
    res = position_only_baseline(np.array(tids), np.array(poss), np.array(vals))
    print(f"\nposition-only control on the WHITE-BOX measure: "
          f"rho={res.spearman:+.3f} (p={res.spearman_p:.1e}) R2_oos={res.r2_oos:+.3f} "
          f"top3={res.topk_agreement.get(3, float('nan')):.2f} "
          f"(chance {res.topk_chance.get(3, float('nan')):.2f})")

    summary = {
        "model": args.model,
        "n_traces": len(per_trace),
        "drop_sink": not args.keep_sink,
        "layers": L, "heads": H,
        "top_receiver_heads": [{"layer": l, "head": h, "peakedness": v} for l, h, v in top],
        "peakedness_by_layer_head": peaked_mean.tolist(),
        "whitebox_position_only": {
            "spearman_oos": res.spearman, "spearman_p": res.spearman_p,
            "r2_oos": res.r2_oos,
            "topk_agreement": {str(k): v for k, v in res.topk_agreement.items()},
            "topk_chance": {str(k): v for k, v in res.topk_chance.items()},
            "curve_x": res.curve_x.tolist(), "curve_y": res.curve_y.tolist(),
        },
        "per_sentence": [{"trace_id": a, "position": b, "attention_received": c}
                         for a, b, c in zip(tids, poss, vals)],
        "traces": per_trace,
    }
    (C.DATA / f"internals_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote internals_{tag}.json")


if __name__ == "__main__":
    main()
