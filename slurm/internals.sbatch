#!/bin/bash
#SBATCH --job-name=anch-internals
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
#
# White-box arm: sentence-level attention, receiver-head scores, and the same
# position-only control applied to the mechanistic measure. Runs under HF
# transformers with eager attention because vLLM does not expose attention
# weights. One forward pass per trace, no sampling.
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
python 11_internals.py --model "${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}" \
  --max-traces "${MAX_TRACES:-8}" --top-heads "${TOP_HEADS:-20}"
echo "done: $(date --iso-8601=seconds)"
