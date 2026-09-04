#!/bin/bash
#SBATCH --job-name=anch-analyze
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
#
# Stage 3 plus the figures, the report and the randomly-selected examples.
# CPU-bound apart from the sentence embeddings, hence only one shard.
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
M="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-14B}"
python 03_analyze.py  --model "$M"
python 05_figures.py  --model "$M"
python 07_examples.py --model "$M"
python 08_report.py   --model "$M"
echo "done: $(date --iso-8601=seconds)"
