#!/bin/bash
#SBATCH --job-name=anch-sample
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
#
# Capture a few real traces from each model. Segmentation is the unit of
# analysis for this project, so the splitter has to be developed against text
# these models actually write.
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
python 98_sample_traces.py --model "${MODEL:-Qwen/Qwen3.5-9B}" --n "${N:-5}" --level "${LEVEL:-4}"
echo "done: $(date --iso-8601=seconds)"
