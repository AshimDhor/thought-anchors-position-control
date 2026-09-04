#!/bin/bash
#SBATCH --job-name=anch-indist
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
#
# Checks the filler arm's premise: is the filler text the model would plausibly
# write in that slot, or an intrusion? Pure prefill, no sampling.
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
python 10_filler_indistribution.py --model "${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}" --n-slots "${N_SLOTS:-120}"
echo "done: $(date --iso-8601=seconds)"
