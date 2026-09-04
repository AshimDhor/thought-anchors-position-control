#!/bin/bash
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
#SBATCH --job-name=anch-resample
#SBATCH --gres=shard:4
#SBATCH --mem=128G
#SBATCH --time=UNLIMITED
#
# Stage 2. The prefix sweep (ARM=main) or the content control (ARM=filler).
# Sharded by trace so each worker keeps a trace's prefixes together, which is
# what makes vLLM's prefix caching pay off.
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
python 02_resample.py --model "${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}" --arm "${ARM:-main}" \
  --shard "${SHARD:-0}" --n-shards "${N_SHARDS:-1}" --gpu-frac "${GPU_FRAC:-0.85}" \
  --max-tokens "${ROLLOUT_MAX_TOKENS:-8192}" --pass-id "${PASS:-1}"
echo "done: $(date --iso-8601=seconds)"
