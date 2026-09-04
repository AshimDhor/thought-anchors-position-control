from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
for _d in (DATA, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)


PRIMARY_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
SECONDARY_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODERN_MODELS = ["allenai/Olmo-3-7B-Think", "Qwen/Qwen3-8B"]

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


TEMPERATURE = 0.6
TOP_P = 0.95

SCREEN_SAMPLES = 12          # rollouts per problem when measuring difficulty
ROLLOUTS_PER_PREFIX = 32     # rollouts per prefix boundary in the main run
                             # (Bogdan et al. use 100; we report the finite-sample
                             #  floor explicitly so the shortfall is visible)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.8          # resampled sentence counts as "different" below this


DIFFICULTY_BAND = (0.25, 0.75)   # matches Bogdan et al.'s own band


MIN_ANSWER_ENTROPY = 0.4
MIN_CLOSED_RATE = 0.8

MIN_SENTENCES = 30
MAX_SENTENCES = 400


MAX_THINKING_CHARS = 18000

N_WINDOWS = 6
WINDOW_LEN = 4

MAX_MODEL_LEN = 20480
MAX_NEW_TOKENS = 16384        # for generating base traces
ROLLOUT_MAX_TOKENS = 8192     # rollouts only finish a trace, they do not start one


FILLER_SUBSAMPLE = 0.5

SEED = 20260820
