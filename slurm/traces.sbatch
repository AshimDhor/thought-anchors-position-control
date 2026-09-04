#!/bin/bash
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
#SBATCH --job-name=anch-traces
#SBATCH --gres=shard:4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# Stage 1. Generate the base trace whose sentences get scored.
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
python 01_traces.py --model "${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}" --n-problems "${N_PROBLEMS:-20}" --gpu-frac "${GPU_FRAC:-0.85}"
echo "done: $(date --iso-8601=seconds)"
