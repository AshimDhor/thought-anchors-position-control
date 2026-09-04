#!/bin/bash
#SBATCH --job-name=anch-diag
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
#
# Run the engine in-process so the real traceback lands in this log instead of
# dying inside an EngineCore subprocess that reports only "initialization failed".
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_LOGGING_LEVEL=INFO

echo "=== torch/CUDA sanity ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("available", torch.cuda.is_available(), "count", torch.cuda.device_count())
print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("matmul ok", float((x @ x).sum()))
PY

echo "=== vLLM in-process ==="
python - <<'PY'
from vllm import LLM, SamplingParams
llm = LLM(model="allenai/Olmo-3-7B-Think", gpu_memory_utilization=0.80,
          max_model_len=8192, enforce_eager=True, enable_prefix_caching=False)
out = llm.generate(["2+2 equals"], SamplingParams(max_tokens=16, temperature=0))
print("GEN:", repr(out[0].outputs[0].text))
print("DIAG_OK")
PY
echo "done: $(date --iso-8601=seconds)"
