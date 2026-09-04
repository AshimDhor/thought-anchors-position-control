#!/usr/bin/env bash
# Shared setup for every batch script in this project.
#
# Cluster facts this encodes (sankhya, verified 2026-08-20):
#   partition/qos : gpu / gpu
#   GPUs          : 8 x H200 NVL, exposed as 32 shards (4 per GPU)
#   quota         : at most 4 shards concurrently per user
#   important     : the node kills long GPU processes started from an interactive
#                   shell after ~5-10 minutes, so everything runs under sbatch.
#                   Do not set CUDA_VISIBLE_DEVICES; SLURM assigns the device.
set -euo pipefail

APP_ROOT="${APP_ROOT:-/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application}"
source /home/samiran2/miniconda3/etc/profile.d/conda.sh

# Prefer the vLLM environment; fall back to the transformers-only one so a
# broken install degrades speed rather than blocking the science.
if /home/samiran2/miniconda3/envs/anchors/bin/python -c "import vllm" 2>/dev/null; then
  conda activate anchors
  BACKEND=vllm
else
  conda activate sot
  BACKEND=hf
fi

cd "${APP_ROOT}/code"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="${HF_HOME:-/home/samiran2/.cache/huggingface}"

# The system toolkit at /usr/local/cuda is CUDA 11.8, which predates Hopper:
# every JIT compile targeting sm_90a died with
#   nvcc fatal : Unsupported gpu architecture 'compute_90a'
# The torch cu130 wheels ship a complete CUDA 13.3 toolkit inside site-packages,
# so we point the JIT at that instead of the system one.
_CU13="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cu13"
if [ -x "${_CU13}/bin/nvcc" ]; then
  export CUDA_HOME="${_CU13}"
  export CUDA_PATH="${_CU13}"
  export PATH="${_CU13}/bin:${PATH}"
  export LD_LIBRARY_PATH="${_CU13}/lib:${LD_LIBRARY_PATH:-}"
fi
export TORCH_CUDA_ARCH_LIST="9.0a"
# Belt and braces: skip flashinfer's sampler kernels entirely. They are a
# throughput optimisation, not a correctness requirement.
export VLLM_USE_FLASHINFER_SAMPLER=0
# The EngineCore subprocess dies here without ever surfacing its traceback
# ("Engine core initialization failed", nothing else). Running the engine
# in-process works and makes any future failure debuggable, which is worth more
# than the small amount of overlap the subprocess buys.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

echo "nvcc        : $(command -v nvcc) ($(nvcc --version 2>/dev/null | tail -1))"

echo "host        : $(hostname)"
echo "job id      : ${SLURM_JOB_ID:-interactive}"
echo "backend     : ${BACKEND}"
echo "python      : $(which python)"
echo "cuda devices: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "started     : $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
echo "---"
