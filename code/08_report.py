"""Stage 7: render the results section straight from the saved numbers.

Every figure quoted in the write-up is emitted by this script rather than typed
by hand, because transcription is exactly the kind of silent error that survives
proofreading. If a number appears in the doc and not here, it should be treated
as suspect.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from anchors import config as C

PRETTY = {
    "kl_resampling": "resampling importance  D_KL[A'||A]",
    "kl_counterfactual": "counterfactual importance (semantic filter)",
    "kl_similar": "paraphrase arm (replacement was *similar*)",
    "kl_corrected": "resampling importance, floor-corrected",
    "tv": "total variation",
    "abs_delta_acc": "|delta P(correct)|",
}


def fmt(x, n=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{n}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.PRIMARY_MODEL)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]

    s = json.loads((C.DATA / f"summary_{tag}.json").read_text())
    df = pd.read_csv(C.DATA / f"sentences_{tag}.csv")
    traces = json.loads((C.DATA / f"traces_{tag}.json").read_text())
    out = []

    out.append(f"# Results — {s['model']}\n")
    out.append(f"- traces: **{s['n_traces']}**, sentences scored: **{s['n_sentences']}**")
    out.append(f"- rollouts per prefix boundary: **{s['rollouts_per_prefix']}** "
               f"(pooled over two independent passes; Bogdan et al. use 100)")
    ns = [t["n_sentences"] for t in traces]
    out.append(f"- sentences per trace: min {min(ns)}, median {int(np.median(ns))}, max {max(ns)}")
    out.append(f"- base trace correct: {sum(t['base_correct'] for t in traces)}/{len(traces)}")
    out.append(f"- screening pass rate of selected problems: "
               f"{np.mean([t['screen_pass_rate'] for t in traces]):.2f} "
               f"(band {C.DIFFICULTY_BAND})")
    out.append(f"- resampled first sentences counted as semantically *different*: "
               f"**{s['frac_resamples_dissimilar']:.1%}**")
    if "drop_no_answer" in s:
        out.append(f"- rollouts producing no boxed answer: "
                   f"**{s['drop_no_answer']['no_answer_rate_overall']:.1%}** "
                   f"(these are labelled `<none>`, not scored as wrong; see 5b)\n")

    out.append("\n## 1. How much does position alone recover?\n")
    out.append("Position-only predictor, fit leave-one-trace-out, scored out of sample.\n")
    out.append("| measure | Spearman rho | R^2 (oos) | top-1 | top-3 | top-5 | chance (top-3) |")
    out.append("|---|---|---|---|---|---|---|")
    for m in ["kl_resampling", "kl_counterfactual", "kl_similar", "tv", "abs_delta_acc"]:
        if m not in s:
            continue
        e = s[m]
        out.append(
            f"| {PRETTY.get(m, m)} | {fmt(e['spearman_oos'])} "
            f"(p={e['spearman_p']:.1e}) | {fmt(e['r2_oos'])} | "
            f"{fmt(e['topk_agreement'].get('1'), 2)} | {fmt(e['topk_agreement'].get('3'), 2)} | "
            f"{fmt(e['topk_agreement'].get('5'), 2)} | {fmt(e['topk_chance'].get('3'), 2)} |")

    out.append("\n## 2. Is the signal above its own noise floor?\n")
    nf = s["noise_floor"]
    out.append(f"- mean KL **{fmt(nf['mean_kl'])}** against a finite-sample floor of "
               f"**{fmt(nf['mean_kl_null'])}**")
    out.append(f"- sentences whose KL exceeds their own floor: **{nf['frac_kl_above_null']:.1%}**")
    out.append(f"- mean TV {fmt(nf['mean_tv'])} against floor {fmt(nf['mean_tv_null'])}")

    if "filler" in s:
        f = s["filler"]
        out.append("\n## 3. Filler control — same slot, no content\n")
        out.append(f"- real sentence: mean KL **{fmt(f['mean_real_kl'])}**")
        out.append(f"- content-free filler in the same slot: mean KL **{fmt(f['mean_filler_kl'])}**")
        out.append(f"- gap **{fmt(f['mean_gap'])}** nats, 95% CI "
                   f"[{fmt(f['gap_ci'][0])}, {fmt(f['gap_ci'][1])}] "
                   f"(cluster bootstrap over traces)")
        out.append(f"- the real sentence exceeds its filler in "
                   f"**{f['frac_real_exceeds_filler']:.0%}** of slots "
                   f"(50% would mean the content is doing nothing)")

    if "paraphrase_contrast" in s:
        pc = s["paraphrase_contrast"]
        out.append("\n## 4. Paraphrase control — same slot, content preserved\n")
        out.append("The complement of the semantic filter that defines counterfactual "
                   "importance: rollouts whose replacement sentence meant roughly the "
                   "same thing. If importance tracks content, these should diverge "
                   "much less than the semantically different ones.\n")
        out.append(f"- semantically **different** replacements: mean KL "
                   f"**{fmt(pc['mean_kl_different'])}**")
        out.append(f"- semantically **similar** replacements: mean KL "
                   f"**{fmt(pc['mean_kl_similar'])}**")
        out.append(f"- gap **{fmt(pc['mean_gap'])}** nats, 95% CI "
                   f"[{fmt(pc['gap_ci'][0])}, {fmt(pc['gap_ci'][1])}]")
        out.append(f"- different exceeds similar in **{pc['frac_different_exceeds_similar']:.0%}** "
                   f"of slots")

    out.append("\n## 5. Where the positional structure comes from\n")
    mech = s["mechanism"]
    out.append(f"- entropy of `A_(i-1)` vs position: rho = **{fmt(mech['entropy_vs_position_rho'])}** "
               f"(the answer distribution concentrates as the trace proceeds)")
    out.append(f"- importance vs that entropy: rho = **{fmt(mech['kl_vs_entropy_before_rho'])}**")
    out.append(f"- importance vs the entropy *drop* at that step: rho = "
               f"**{fmt(mech['kl_vs_entropy_drop_rho'])}**")

    if "drop_no_answer" in s:
        d = s["drop_no_answer"]
        out.append("\n## 5b. Robustness: are truncated rollouts driving this?\n")
        out.append("A rollout that exhausts its budget mid-thought has no boxed answer. "
                   "The amount of reasoning left to write shrinks along a trace, so the "
                   "rate of those is itself a function of position -- exactly the sort of "
                   "thing that could manufacture a positional pattern. Recomputed with "
                   "those rollouts dropped rather than labelled:\n")
        out.append(f"- overall no-answer rate: **{d['no_answer_rate_overall']:.1%}**; "
                   f"its correlation with position: rho = "
                   f"**{fmt(d['no_answer_rate_vs_position_rho'])}**")
        out.append(f"- position-only predictor after dropping them: rho "
                   f"**{fmt(d['spearman_oos'])}**, R^2 {fmt(d['r2_oos'])}, "
                   f"top-3 {fmt(d['top3'], 2)}")
        out.append(f"- compare the headline row above: rho "
                   f"{fmt(s['kl_resampling']['spearman_oos'])}, "
                   f"top-3 {fmt(s['kl_resampling']['topk_agreement'].get('3'), 2)}")

    out.append("\n## 6. Robustness to the smoothing prior\n")
    out.append("KL needs a prior or it is infinite whenever a rollout produces an "
               "answer the other side never produced. Jeffreys (0.5) is the default; "
               "the conclusions should not move.\n")
    out.append("| Jeffreys alpha | rho | R^2 (oos) | top-3 |")
    out.append("|---|---|---|---|")
    for lam, v in s.get("smoothing_sensitivity", {}).items():
        out.append(f"| {lam} | {fmt(v['spearman_oos'])} | {fmt(v['r2_oos'])} | "
                   f"{fmt(v['top3'], 2)} |")

    sim_path = C.DATA / "simulation.json"
    if sim_path.exists():
        sim = json.loads(sim_path.read_text())
        out.append("\n## 7. Instrument validation (simulated ground truth)\n")
        out.append("| check | result |")
        out.append("|---|---|")
        for c in sim["checks"]:
            out.append(f"| {c['check']} | {'PASS' if c['pass'] else 'FAIL'} |")
        for r in sim["results"]:
            out.append(f"\n**{r['world']}**: position-only rho "
                       f"{fmt(r['position_only_spearman'])}; "
                       f"mean KL by position sextile {r['mean_kl_by_position_sextile']}")
            if "planted_real_minus_filler" in r:
                out.append(f"  - filler gap at the planted step "
                           f"**{fmt(r['planted_real_minus_filler'])}** vs "
                           f"**{fmt(r['offtarget_real_minus_filler'])}** elsewhere")
                out.append(f"  - planted step in top-3: {r['frac_top3_raw']:.0%} raw, "
                           f"{r['frac_top3_residual']:.0%} after residualising")

    # A concrete look at the extremes, so the reader can judge for themselves.
    out.append("\n## 8. The highest- and lowest-importance sentences\n")
    d = df.dropna(subset=["kl_resampling"]).sort_values("kl_resampling", ascending=False)
    for label, rows in [("Highest", d.head(6)), ("Lowest", d.tail(6))]:
        out.append(f"\n**{label} measured importance**\n")
        for _, r in rows.iterrows():
            out.append(f"- `KL={r.kl_resampling:.2f}` (floor {r.kl_null:.2f}, "
                       f"pos {r.position:.2f}) — {' '.join(str(r.text).split())[:150]}")

    dest = C.ROOT / "writeup" / f"results_{tag}.md"
    dest.write_text("\n".join(out))
    print("\n".join(out))
    print(f"\n[wrote {dest}]")


if __name__ == "__main__":
    main()
