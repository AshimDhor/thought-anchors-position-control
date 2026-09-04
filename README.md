# What resampling importance actually measures

Code and data for a red-team of the black-box **thought anchors** measure
(Bogdan, Macar, Nanda & Conmy) — testing whether sentence-level resampling
importance is confounded by a sentence's **position** in the reasoning trace.

**It isn't.** The measure survives that confound. What it *is* dominated by is
how undecided the model still was when the sentence arrived.

> **Headline.** Regressing importance on candidate explanations (254 sentences):
> headroom `H(A_{i-1})` explains **R² = 0.351**, against **0.027** for position
> and **0.014** for sentence category. Headroom uniquely adds **+0.334** over the
> other two combined, and is not position in disguise (ρ = −0.45 with position).

Setting: **DeepSeek-R1-Distill-Qwen-14B** on MATH-500 — the model the method was
built on. 8 traces, 254 sentences, 64 rollouts per prefix boundary.

---

## The measure

For a trace with sentences `S_0 … S_{n-1}`, let `A_i` be the distribution over
final answers when the model continues from the prefix **including** `S_i`, and
`A_{i-1}` the distribution when it stops **before** `S_i` (so the model resamples
its own replacement `T_i`):

```
resampling importance       imp_r(i) = D_KL[ p(A_{i-1}) || p(A_i) ]
counterfactual importance   imp(i)   = D_KL[ p(A_{i-1} | T_i ≁ S_i) || p(A_i) ]
```

`T_i ≁ S_i` is cosine similarity < 0.8 under `all-MiniLM-L6-v2`, matching the
original work.

**`A'_i` and `A_{i-1}` are the same object** — both are rollouts from prefix
`S_{<i}` — so one sweep over the `n+1` prefix boundaries yields every importance
in the trace.

## Three controls

| control | holds fixed | varies | asks |
|---|---|---|---|
| position-only predictor | — | — | what does a rule that never reads the text recover? |
| filler arm | position | content **removed** | does a content-free sentence move the distribution as much? |
| paraphrase arm | position | content **preserved** | when the model resamples the same meaning, does divergence vanish? |

The paraphrase arm is free: it is the complement of the semantic filter that
defines counterfactual importance — the rollouts the published measure discards.

**The filler arm was withdrawn.** A pre-registered check found the filler sits
−7.61 σ from the model's own sentences (a deliberately alien sentence sits at
−11.68 σ), so it measured *disruption*, not "same position, no content". See
`10_filler_indistribution.py`.

---

## Results

| # | finding | evidence |
|---|---|---|
| 1 | Importance is **not** positional | ρ = +0.044 (p=0.48); top-3 recovery **0.083 vs chance 0.095** |
| 2 | It tracks **how undecided the model was** | R² = **0.351** vs 0.027 (position) / 0.014 (category) |
| 3 | The categorical claim survives too | category is strongly positional (Kruskal p=1.2e-13) yet shared R² = **0.001** |
| 4 | White-box receiver heads are **recency** | **84%** of their attention mass within 1 sentence of the diagonal |
| 5 | The semantic filter does nothing detectable | different 0.103 vs similar 0.119; gap −0.016, CI [−0.081, +0.040] |
| 6 | The method doesn't reach current models | ~**11 H200-hours per trace** on Olmo-3-Think / Qwen3-8B |

![headline](figures/fig0_headline_DeepSeek-R1-Distill-Qwen-14B.png)

---

## Start here: four notebooks, no GPU required

Every notebook runs **CPU-only in seconds** from data committed to this repo. No
model download, no API key, no GPU. Outputs are saved, so you can read them on
GitHub without running anything.

| notebook | what it does | needs |
|---|---|---|
| [`01_reproduce_headline.ipynb`](notebooks/01_reproduce_headline.ipynb) | Re-derives the headline from scratch: the variance decomposition (**R² = 0.351** headroom vs 0.027 position vs 0.014 category), the position-only control, the noise floor, the paraphrase contrast | CPU, seconds |
| [`02_look_at_the_data.ipynb`](notebooks/02_look_at_the_data.ipynb) | The three judgement calls the study rests on: segmentation, answer extraction, labelling. **Re-computes an importance value by hand** and checks it against the pipeline. Shows the withdrawn filler control | CPU, seconds |
| [`03_instrument_validation.ipynb`](notebooks/03_instrument_validation.ipynb) | The six pass conditions declared *before* running, on planted ground truth. Shows the estimator is not intrinsically positional — which is what makes the null interpretable | CPU, ~1 min |
| [`04_whitebox_recency.ipynb`](notebooks/04_whitebox_recency.ipynb) | The white-box arm, and the error I found in it: the "receiver heads" turn out to be **recency heads** (84% of mass within one sentence of the diagonal) | CPU, seconds |

```bash
pip install -r requirements.txt   # or just: numpy pandas scipy matplotlib
jupyter lab notebooks/
```

> **If you only run one:** notebook 1. If you want to know whether to *believe*
> notebook 1, run notebook 3 first — it is the argument that the instrument
> works.

---

## Reproducing

**Environment.** Python 3.12, one H200-class GPU. See `requirements.txt`.
The engine runs vLLM in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) — see
`slurm/_common.sh` for the node-specific reasons, including a CUDA 11.8 / Hopper
incompatibility that requires pointing `CUDA_HOME` at the toolkit bundled in the
torch wheels.

```bash
make sim         # instrument validation on planted ground truth — start here
make traces      # select problems on answer entropy, generate base traces
make resample    # the prefix sweep            (ARM=main)
make filler      # the content control          (ARM=filler)
make analyze     # importance + all controls -> data/sentences_*.csv
make labels categories figures examples report
```

On a SLURM cluster use `slurm/` instead; `slurm/run_pipeline.sh` chains the whole
thing with `afterok` dependencies.

### Regenerating the attention tensors

The white-box arm writes per-trace attention as `data/attn_*.npy`
(`[layers, heads, sentences, sentences]`). **These are excluded from the repo** —
they total 591 MB and two exceed GitHub's 100 MB file limit. Regenerate with:

```bash
python code/11_internals.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --max-traces 8
python code/14_mingap_sweep.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
```

One forward pass per trace under eager attention; no sampling. Everything else
needed to reproduce every number in the report **is** in `data/`.

---

## What's here

| path | contents |
|---|---|
| `code/anchors/` | library: segmentation, answer normalisation, importance estimators + noise floor, position baseline, rollout engine, simulator |
| `code/00`–`15_*.py` | pipeline stages, numbered in execution order |
| `code/Makefile` | one target per stage |
| `slurm/` | cluster submission; `_common.sh` documents the node-specific fixes |
| `data/` | screening, length probes, traces, raw rollouts, per-sentence CSV, all summary JSON |
| `figures/` | the nine figures in the report |
| `notebooks/` | four CPU-only notebooks reproducing the results (see above) |

### Stages worth reading first

- **`06_simulate.py`** — validates the instrument on planted ground truth with
  **six pass conditions declared before running**. Shows the estimator is not
  intrinsically positional, which is what makes the real-data null interpretable.
- **`anchors/importance.py`** — the estimators, and the parametric-bootstrap
  **finite-sample floor**. With R rollouts a side, empirical KL is positive even
  when the two distributions are identical, and that offset varies along a trace.
- **`anchors/baselines.py`** — the position-only predictor, fit
  leave-one-trace-out so it can never memorise the trace it scores.
- **`14_mingap_sweep.py`** — the correction that turned the white-box result from
  "strongly positional" into "dominated by recency".

## Design decisions that carry weight

- **Prefixes are exact slices.** Segmentation returns character spans, so a
  rollout prompt is literally `thinking[:span.end]` — the bytes the model
  produced. Nothing is re-rendered. There is a tiling assertion.
- **`<none>` is an outcome, not a wrong answer.** A rollout that exhausts its
  budget is labelled separately; folding it into "incorrect" would hand-build a
  positional trend. *This is what exposed a broken experiment early on.*
- **Rollouts are unseeded.** A fixed `SamplingParams.seed` makes vLLM
  deterministic per prompt — 24 identical requests returned the same 8
  completions — which would impose common random numbers across neighbouring
  prefixes.
- **Windowed sampling.** 6 windows × 4 consecutive sentences measures the *same*
  sentence-level quantity on a random subsample rather than coarsening the unit:
  55 prefixes instead of 517, 42M tokens instead of 271M.
- **Cluster bootstrap** over traces for all CIs; sentences within a trace are not
  independent.

## Honest limits

- **n = 8 traces / 254 sentences.** The constraint is answer dispersion, not
  compute: entropy is exactly **zero for 84% of the 140 problems screened**.
- **61% of sentences clear their own noise floor** at R = 64. The pooled and
  variance-decomposition results are the defensible ones, not per-sentence ranks.
- **A null is not proof of absence.** `|ΔP(correct)|`, a lower-variance
  statistic, *does* show a positional effect (ρ = +0.237) — so this bounds a
  positional component rather than excluding one.
- **The white-box long-range result is unresolved** — non-monotone in minimum
  sentence gap, and at n = 8 I can't separate structure from noise.

## Pre-registration

Eight predictions with numeric confidences were written and hashed **before any
importance number existed on a real trace**. Final score: **3 right, 4 wrong, 1
partial** — including both of the two held with highest confidence. The
pre-registration and its scoring live in the write-up.

## License

MIT — see `LICENSE`.
