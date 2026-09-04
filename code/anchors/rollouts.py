"""Rollout engine.

One vLLM instance serves every experiment.  The only subtlety is prompt
construction: a rollout from prefix ``i`` must look to the model exactly like
its own half-finished thought, not like a reconstruction of it.  Both models we
use open the reasoning block in their generation prompt (``...assistant\\n<think>``),
so we build

    chat_template(problem, add_generation_prompt=True) + thinking_trace[:end_i]

and let the model carry on.  Because ``thinking_trace[:end_i]`` is a literal
slice of text the model itself produced, there is no re-rendering step that
could put the continuation off-distribution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class GenConfig:
    temperature: float = 0.6      # the sampling settings both model cards
    top_p: float = 0.95           # recommend for reasoning mode
    max_tokens: int = 3072
    # Deliberately None.  A fixed SamplingParams.seed makes vLLM fully
    # deterministic *per prompt*: 24 identical requests at seed=12345 returned
    # only 8 distinct completions in total (the same 8, 24 times), while
    # unseeded returned 192/192 distinct at identical throughput. Every prompt
    # in this study is unique, so a seed would not have corrupted the results --
    # but it would have imposed common random numbers across neighbouring
    # prefixes, and the estimator wants independent draws either side of the
    # comparison. Reproducibility comes from storing every raw rollout instead.
    seed: int | None = None
    stop: list[str] = field(default_factory=list)


class Engine:
    """Thin wrapper over vLLM, with a plain-transformers fallback.

    vLLM is 10-20x faster here and is what the reported runs used; the fallback
    exists so the pipeline is runnable on a machine without it, and so a broken
    install never silently becomes a scientific decision.
    """

    def __init__(self, model: str, gpu_memory_utilization: float = 0.85,
                 tensor_parallel_size: int = 1, max_model_len: int = 8192,
                 backend: str = "auto", max_num_seqs: int = 512):
        from transformers import AutoTokenizer

        self.model_name = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_model_len = max_model_len

        if backend == "auto":
            try:
                import vllm  # noqa: F401
                backend = "vllm"
            except ImportError:
                backend = "hf"
        self.backend = backend
        if backend == "hf":
            self._init_hf()
            return

        from vllm import LLM
        self.llm = LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            # Measured on this node (slurm/probe.sbatch, job 1473) over a
            # workload shaped like the real sweep -- many prompts sharing a long
            # prefix:
            #   eager    + prefix cache : 5205 tok/s
            #   cudagraph+ prefix cache : 3868 tok/s  (and 42s to load)
            #   eager    + no cache     : 2566 tok/s
            # CUDA graphs lose here because the batch is prefill-heavy, so we
            # take eager and spend the memory on KV cache instead.
            enforce_eager=True,
            # Every prefix of a trace shares a prefix with every longer one, so
            # the sweep re-reads the same tokens O(n) times.  Prefix caching
            # turns that from the dominant cost into a rounding error.
            enable_prefix_caching=True,
            # Left at the default, the sweep ran at a few hundred tokens/s
            # because too few sequences were resident at once. Raising it took
            # the same workload to ~3.5k tok/s.
            max_num_seqs=max_num_seqs,
        )

    def _init_hf(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        print(f"[Engine] vLLM unavailable; falling back to transformers for {self.model_name}")
        self.torch = torch
        self.hf = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.bfloat16, device_map="auto")
        self.hf.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def _generate_hf(self, prompts: list[str], cfg: "GenConfig", n: int,
                     batch_size: int = 32) -> list[list[str]]:
        torch = self.torch
        # Expand n as repeated prompts: simpler than num_return_sequences and
        # keeps memory predictable.
        flat = [p for p in prompts for _ in range(n)]
        outs: list[str] = []
        for start in range(0, len(flat), batch_size):
            chunk = flat[start : start + batch_size]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 truncation=True,
                                 max_length=self.max_model_len - cfg.max_tokens)
            enc = {k: v.to(self.hf.device) for k, v in enc.items()}
            with torch.no_grad():
                gen = self.hf.generate(
                    **enc, do_sample=cfg.temperature > 0,
                    temperature=max(cfg.temperature, 1e-5), top_p=cfg.top_p,
                    max_new_tokens=cfg.max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1] :]
            outs.extend(self.tokenizer.batch_decode(new, skip_special_tokens=True))
        return [outs[i * n : (i + 1) * n] for i in range(len(prompts))]

    # ---- prompt construction -------------------------------------------------

    def chat_prefix(self, question: str, instruction: str) -> str:
        """The prompt up to and including the model's opening ``<think>``.

        Models differ on who writes that tag.  Olmo-3-Think's generation prompt
        ends with ``<think>`` already; Qwen3's does not, and the model emits it
        as its first output token.  Either way a rollout has to resume *inside*
        the reasoning block, so we normalise by appending the opening tag
        ourselves when the template did not.
        """
        msgs = [{"role": "user", "content": f"{question}\n\n{instruction}"}]
        base = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        if THINK_OPEN not in base[-64:]:
            base = base + THINK_OPEN + "\n"
        return base

    def rollout_prompt(self, question: str, instruction: str, thinking_prefix: str) -> str:
        return self.chat_prefix(question, instruction) + thinking_prefix

    # ---- generation ----------------------------------------------------------

    def generate(self, prompts: list[str], cfg: GenConfig, n: int = 1) -> list[list[str]]:
        """Return ``n`` completions for each prompt, in prompt order."""
        if self.backend == "hf":
            return self._generate_hf(prompts, cfg, n)

        from vllm import SamplingParams

        params = SamplingParams(
            n=n,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
            stop=cfg.stop or None,
        )
        outs = self.llm.generate(prompts, params)
        return [[o.text for o in out.outputs] for out in outs]


def split_thinking(completion: str) -> tuple[str, str]:
    """Split a completion into (reasoning trace, post-</think> answer text).

    If the model never closed the block -- it ran out of budget mid-thought --
    the whole completion is reasoning and there is no answer section.  Those
    rollouts are counted as ``<none>``, and their rate is reported, because
    silently scoring them as wrong would manufacture exactly the kind of
    positional trend this project is trying to measure.
    """
    body = completion
    # If the model re-emitted the opening tag (because the template did not),
    # drop it so that `thinking` is the reasoning text itself and prefixes stay
    # comparable across models.
    stripped = body.lstrip()
    if stripped.startswith(THINK_OPEN):
        body = stripped[len(THINK_OPEN):].lstrip("\n")
    idx = body.find(THINK_CLOSE)
    if idx == -1:
        return body, ""
    return body[:idx], body[idx + len(THINK_CLOSE):]
