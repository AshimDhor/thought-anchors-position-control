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

    seed: int | None = None
    stop: list[str] = field(default_factory=list)


class Engine:


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

    body = completion

    stripped = body.lstrip()
    if stripped.startswith(THINK_OPEN):
        body = stripped[len(THINK_OPEN):].lstrip("\n")
    idx = body.find(THINK_CLOSE)
    if idx == -1:
        return body, ""
    return body[:idx], body[idx + len(THINK_CLOSE):]
