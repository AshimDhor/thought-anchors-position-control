"""Merge sharded rollout files into the single file stage 3 expects.

Two modes.

``--mode dedupe`` (default) stitches together shards of one pass: each
(trace_id, prefix) appears in exactly one shard, so duplicates mean a shard was
run twice and the extra copy is dropped.

``--mode concat`` combines *independent passes* over the same prefixes, pooling
their rollouts. Rollouts are unseeded, so a second pass draws fresh samples from
the same distribution and concatenating them is simply a larger R. This is the
cheap way to lower the finite-sample floor, which matters here: at R = 32 on
problems with only two or three distinct answers, a good fraction of sentences
sit below their own floor and their individual scores are noise.
"""

from __future__ import annotations

import argparse
import json

from anchors import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    ap.add_argument("--arm", default="main")
    ap.add_argument("--mode", choices=["dedupe", "concat"], default="dedupe")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional already-merged files to pool with (concat mode)")
    args = ap.parse_args()
    tag = args.model.split("/")[-1]

    parts = sorted(C.DATA.glob(f"rollouts_{args.arm}_{tag}.shard*.json"))
    if not parts:
        print("no shards found; nothing to merge")
        return
    merged = []
    for p in parts:
        recs = json.loads(p.read_text())
        merged.extend(recs)
        print(f"  {p.name}: {len(recs)} prefixes")
    for extra in args.extra:
        recs = json.loads((C.DATA / extra).read_text())
        merged.extend(recs)
        print(f"  {extra}: {len(recs)} prefixes (pooled)")

    by_key: dict[tuple[str, int], dict] = {}
    for r in merged:
        key = (r["trace_id"], r["i"])
        if key not in by_key:
            by_key[key] = dict(r)
        elif args.mode == "concat":
            tgt = by_key[key]
            tgt["answers"] = tgt["answers"] + r["answers"]
            tgt["first_sentences"] = (tgt.get("first_sentences", [])
                                      + r.get("first_sentences", []))
    out = list(by_key.values())
    n_roll = [len(r["answers"]) for r in out]
    dest = C.DATA / f"rollouts_{args.arm}_{tag}.json"
    dest.write_text(json.dumps(out))
    print(f"wrote {dest} ({len(out)} unique prefixes, "
          f"rollouts/prefix min {min(n_roll)} median {sorted(n_roll)[len(n_roll)//2]} "
          f"max {max(n_roll)})")


if __name__ == "__main__":
    main()
