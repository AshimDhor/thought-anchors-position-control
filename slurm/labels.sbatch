#!/bin/bash
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
#SBATCH --job-name=anch-labels
#SBATCH --gres=shard:4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# Stage 4. Label sentence function with a model from a different family than the
# one under study, so the labels are not the target model grading itself.
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
python 04_labels.py --target-model "${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}" --labeller "${LABELLER:-Qwen/Qwen3-8B}" --gpu-frac "${GPU_FRAC:-0.85}"

# The category analysis needs the labels, so it runs here rather than as its own
# job; it is CPU-only and takes seconds.
python 09_categories.py --model "${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
echo "done: $(date --iso-8601=seconds)"
