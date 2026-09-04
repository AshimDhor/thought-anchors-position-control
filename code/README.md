# Is thought-anchor importance measuring the sentence, or its position?

Reimplementation of the black-box resampling measure from *Thought Anchors*
(Bogdan, Macar, Nanda & Conmy), plus three controls it does not have.

## The measure

For a reasoning trace with sentences `S_0 … S_{n-1}`, let `A_i` be the
distribution over final answers when the model continues from the prefix that
**includes** `S_i`. Then `A_{i-1}` is the distribution when it continues from
the prefix that **stops before** `S_i` — so the model resamples a replacement
sentence `T_i` and carries on. The paper's two quantities are

```
resampling importance       importance_r := D_KL[ p(A'_i) || p(A_i) ]
counterfactual importance   importance   := D_KL[ p(A'_i | T_i !~ S_i) || p(A_i) ]
```

with `A'_i` and `A_{i-1}` being the *same object*: the rollouts launched from
prefix `S_{<i}`. One sweep over the `n+1` prefix boundaries therefore supplies
every importance in the trace. `T_i !~ S_i` means cosine similarity below 0.8
under `all-MiniLM-L6-v2`, which is what the paper uses.

## The controls

| control | holds fixed | varies | answers |
|---|---|---|---|
| **position-only predictor** | — | — | how much of the ranking does a rule that never reads the text recover? |
| **filler arm** | position | content (removed) | does a content-free sentence in slot `i` move the answer distribution as much as the real one? |
| **paraphrase arm** | position | content (preserved) | when the model resamples something that *means the same thing*, does the divergence go away? |

The paraphrase arm is free: it is the complement of the semantic filter that
defines counterfactual importance, i.e. the rollouts the paper discards.

## Pipeline

```
make sim        06_simulate.py    instrument validation on planted ground truth
make screen     00_screen.py      keep problems with pass rate in [0.25, 0.75]
make traces     01_traces.py      one base trace per problem, split into sentences
make resample   02_resample.py    the prefix sweep            (ARM=main)
make filler     02_resample.py    the content control         (ARM=filler)
make analyze    03_analyze.py     importance + all controls -> data/sentences_*.csv
make labels     04_labels.py      sentence function, labelled by another model
make figures    05_figures.py
make examples   07_examples.py    randomly selected raw examples
```

Run on the cluster with `sbatch slurm/<stage>.sbatch` — see `slurm/_common.sh`
for the node-specific facts (shard scheduling, the CUDA 11.8 / Hopper problem,
and why the engine runs in-process).

## Design decisions that carry weight

- **Prefixes are exact slices.** `split_sentences` returns character spans into
  the trace, so a rollout prompt is literally `thinking[:span.end]` — the bytes
  the model itself produced. Nothing is re-rendered, so no measured effect can
  be an artefact of reconstruction. There is a tiling assertion for this.
- **Segmentation follows the models' actual style.** These models write
  outlines, not prose. Bullets and numbered items are unit boundaries, and list
  markers are masked from the sentence-terminator rule.
- **`<none>` is an outcome, not a wrong answer.** A rollout that exhausts its
  budget mid-thought is labelled separately. Scoring it as incorrect would put a
  positional trend into the data by hand, since later prefixes leave less room
  to run over.
- **Every divergence is reported against its finite-sample floor,** estimated by
  parametric bootstrap under the null that the sentence did nothing. With `R`
  rollouts a side, empirical KL is positive even when the two distributions are
  identical, and the size of that offset changes along a trace.
- **The position-only predictor is fit leave-one-trace-out,** so it can never
  memorise the trace it scores.
- **Clustered bootstrap.** Sentences within a trace are not independent;
  confidence intervals resample traces, not sentences.

## Instrument validation

`06_simulate.py` runs the analysis on three simulated worlds with known ground
truth and six pass conditions declared before the numbers existed: a null world
where no sentence causes anything, a world with a planted anchor at a random
position, and one where the anchor always sits at the same position. The third
exists because the second exposed a limit of the position-only control — if real
anchors cluster positionally, residualising against position deletes real
signal, monotonically in the smoother's resolution. That is why the filler arm,
not the residualisation, is the load-bearing comparison.
