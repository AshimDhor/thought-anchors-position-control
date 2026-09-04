#!/bin/bash
#SBATCH --job-name=anch-probe
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
# Which engine options actually work on this node, and how fast. Prefix caching
# matters a lot here -- every prefix of a trace is a prefix of every longer one --
# so it is worth a job to find out whether we can have it.
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
cat > /tmp/anch_probe_one.py <<'PY'
import sys, time
from vllm import LLM, SamplingParams

MODEL = sys.argv[1]
eager = sys.argv[2] == "1"
pcache = sys.argv[3] == "1"
name = f"{'eager' if eager else 'cudagraph'}, {'prefix-cache' if pcache else 'no-cache'}"
t0 = time.time()
llm = LLM(model=MODEL, gpu_memory_utilization=0.80, max_model_len=8192,
          enforce_eager=eager, enable_prefix_caching=pcache)
load = time.time() - t0
# Long shared prefix with nested continuations: this is the shape of the real
# sweep, where every prefix of a trace is a prefix of every longer one.
base = "Solve the problem step by step. " + ("We consider the next case. " * 250)
prompts = [base[: len(base) - 27 * i] for i in range(48)]
t0 = time.time()
outs = llm.generate(prompts, SamplingParams(max_tokens=384, temperature=0.6,
                                            top_p=0.95, n=8))
gen = time.time() - t0
ntok = sum(len(o.token_ids) for out in outs for o in out.outputs)
print(f"RESULT | {name:28s} | load {load:6.1f}s | gen {gen:6.1f}s | {ntok/gen:8.0f} tok/s")
PY

MODEL="${MODEL:-allenai/Olmo-3-7B-Think}"
for cfg in "1 0" "1 1" "0 1"; do
  set -- $cfg
  python /tmp/anch_probe_one.py "$MODEL" "$1" "$2" 2>&1 | grep -E "RESULT|Error:|ValueError|RuntimeError" | tail -3 || echo "RESULT | $cfg | FAILED"
done
echo "PROBE_DONE"
echo "done: $(date --iso-8601=seconds)"
