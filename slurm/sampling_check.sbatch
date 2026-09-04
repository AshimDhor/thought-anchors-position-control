#!/bin/bash
#SBATCH --job-name=anch-sampcheck
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
#
# Two questions, one job.
#   (a) Correctness: with a fixed SamplingParams.seed and n>1, are the n samples
#       actually different? If they were identical, every pass rate would be 0 or
#       1 and the entire difficulty filter would be meaningless. Worth 3 minutes
#       to rule out.
#   (b) Speed: the screening run came in around 20x below the benchmark. Is the
#       seed responsible, or the concurrency cap?
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
python - <<'PY'
import time
from vllm import LLM, SamplingParams
from anchors import config as C
from anchors.rollouts import Engine

llm = LLM(model=C.PRIMARY_MODEL, gpu_memory_utilization=0.85, max_model_len=8192,
          enforce_eager=True, enable_prefix_caching=True, max_num_seqs=512)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(C.PRIMARY_MODEL)
q = "If $3x+7=22$, what is $x^2-1$?"
msgs = [{"role": "user", "content": f"{q}\n\n{C.INSTRUCTION}"}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

for label, seed in [("seeded", 12345), ("unseeded", None)]:
    sp = SamplingParams(n=8, temperature=0.6, top_p=0.95, max_tokens=600, seed=seed)
    t0 = time.time()
    outs = llm.generate([prompt] * 24, sp)
    dt = time.time() - t0
    ntok = sum(len(o.token_ids) for out in outs for o in out.outputs)
    texts = [o.text for o in outs[0].outputs]
    uniq_within = len(set(texts))
    across = len(set(o.text for out in outs for o in out.outputs))
    print(f"CHECK | {label:9s} | {ntok/dt:7.0f} tok/s | distinct within one request: "
          f"{uniq_within}/8 | distinct across 24 identical requests: {across}/192")

# Throughput at realistic prompt lengths (long shared prefixes).
long_prompt = prompt + "Let me work through this. " * 300
sp = SamplingParams(n=16, temperature=0.6, top_p=0.95, max_tokens=1024)
t0 = time.time()
outs = llm.generate([long_prompt[: len(long_prompt) - 26 * i] for i in range(40)], sp)
dt = time.time() - t0
ntok = sum(len(o.token_ids) for out in outs for o in out.outputs)
print(f"CHECK | sweep-shaped, n=16, 40 prefixes | {ntok/dt:7.0f} tok/s | {ntok} tokens in {dt:.0f}s")
print("SAMPCHECK_DONE")
PY
echo "done: $(date --iso-8601=seconds)"
