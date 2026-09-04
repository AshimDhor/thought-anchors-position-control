#!/bin/bash
#SBATCH --job-name=anch-smoke
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
#
# Cheapest possible end-to-end check that the engine loads and generates on this
# node. Two minutes of GPU beats another forty-minute failure.
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
python - <<'PY'
from anchors import config as C
from anchors.rollouts import Engine, GenConfig, split_thinking
from anchors.splitting import split_sentences
from anchors.answers import final_answer

import os
MODEL = os.environ.get("MODEL", C.PRIMARY_MODEL)
print("MODEL:", MODEL)
eng = Engine(MODEL, gpu_memory_utilization=0.85,
             max_model_len=C.MAX_MODEL_LEN)
print("BACKEND:", eng.backend)
q = "What is the remainder when $7^{2026}$ is divided by 100?"
prompts = [eng.chat_prefix(q, C.INSTRUCTION)]
outs = eng.generate(prompts, GenConfig(temperature=C.TEMPERATURE, top_p=C.TOP_P,
                                       max_tokens=2048, seed=0), n=2)
for k, c in enumerate(outs[0]):
    think, tail = split_thinking(c)
    sents = split_sentences(think)
    print(f"--- completion {k}: closed={bool(tail)} sentences={len(sents)} "
          f"answer={final_answer(c)!r}")
    for s in sents[:4]:
        print(f"   [{s.index}] {' '.join(s.text.split())[:120]}")
print("SMOKE_OK")
PY
echo "done: $(date --iso-8601=seconds)"
