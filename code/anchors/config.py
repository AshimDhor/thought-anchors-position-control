"""Shared constants.  Everything a reader would need to reproduce the numbers."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
for _d in (DATA, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Olmo-3-7B-Think is the primary: a current reasoning model whose whole training
# pipeline is public, which matters if any of this turns into a question about
# where the behaviour comes from.  Qwen3.5-9B was the original choice, but it is
# a hybrid Mamba-attention VL architecture whose vLLM path fails to initialise on
# this node; that is an infrastructure fact, not a scientific one, and it is
# recorded here so the model choice does not look like cherry-picking.
# Primary is the model the method was actually built on. A red-team of a method
# is most convincing on the method's home ground: it forecloses the objection
# that the method was broken by moving it somewhere it was never meant to go.
# The measured case for it, from 12_length_probe.py:
#   R1-Distill-14B @16k : 96.4% close, median trace 2380 tok
#   Olmo-3-7B-Think     : median 5130 tok, in-band traces 11669 tok / 516 sent
#   Qwen3-8B            : median 4856 tok, in-band traces 11751 tok / 530 sent
# The modern models are reported separately as a scaling result, not dropped.
PRIMARY_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
SECONDARY_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODERN_MODELS = ["allenai/Olmo-3-7B-Think", "Qwen/Qwen3-8B"]

# Kept verbatim in the prompt so the boxed-answer extractor has something to find.
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

# Sampling.  Both model cards recommend these for reasoning mode; we do not tune
# them, since tuning temperature per problem would confound difficulty with
# answer dispersion.
TEMPERATURE = 0.6
TOP_P = 0.95

SCREEN_SAMPLES = 12          # rollouts per problem when measuring difficulty
ROLLOUTS_PER_PREFIX = 32     # rollouts per prefix boundary in the main run
                             # (Bogdan et al. use 100; we report the finite-sample
                             #  floor explicitly so the shortfall is visible)

# Semantic filter for counterfactual importance, following the paper.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.8          # resampled sentence counts as "different" below this

# Problems are kept only if the model is genuinely uncertain about them.  A
# resampling measure asks how much a sentence *moved* the answer distribution;
# where the model is at 0 or 100 percent there is no mass to move and the
# measure is identically zero for reasons that have nothing to do with the
# sentences.  This is a selection on the outcome variable and is stated as a
# limitation in the write-up.
DIFFICULTY_BAND = (0.25, 0.75)   # matches Bogdan et al.'s own band

# What the estimator actually needs is dispersion in the answer distribution --
# room for a sentence to move things. Pass rate is the paper's proxy for that and
# it is lossy in both directions: a problem yielding six different wrong answers
# has abundant dispersion at pass rate 0, while one that confidently repeats a
# single wrong answer has none. We select on the thing itself. Measured on 140
# problems, entropy is 0 for 84% of them, so this is the binding constraint.
MIN_ANSWER_ENTROPY = 0.4
MIN_CLOSED_RATE = 0.8

# Traces outside this range are dropped: very short ones have too few sentences
# for a positional analysis, very long ones blow up the rollout budget.
# Windowed sampling decouples sweep cost from sentence count, so the ceiling
# here can be generous: it exists only to exclude degenerate traces.
MIN_SENTENCES = 30
MAX_SENTENCES = 400

# Cost bound.  A rollout from prefix i has to write the rest of the trace, so
# the sweep costs roughly (n_sentences + 1) x R x (half a trace) tokens per
# trace -- quadratic in trace length.  Capping length is what makes the sweep
# affordable at all, and it is a real selection effect on trace length that the
# write-up states rather than hides.
MAX_THINKING_CHARS = 18000

# Subsampling the sweep.  A full sweep is (n+1) prefixes; on real traces that is
# unaffordable.  We measure the same sentence-level quantity on N_WINDOWS runs of
# WINDOW_LEN consecutive sentences, spread evenly across the trace.  Set
# N_WINDOWS = 0 to sweep everything.
N_WINDOWS = 6
WINDOW_LEN = 4

MAX_MODEL_LEN = 20480
MAX_NEW_TOKENS = 16384        # for generating base traces
ROLLOUT_MAX_TOKENS = 8192     # rollouts only finish a trace, they do not start one

# The filler arm is run on a random half of the slots rather than all of them.
# The comparison it supports is position-stratified, so a uniform random subset
# of positions costs half the compute and buys the same contrast.
FILLER_SUBSAMPLE = 0.5

SEED = 20260820
