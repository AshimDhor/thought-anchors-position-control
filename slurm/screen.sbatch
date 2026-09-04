#!/bin/bash
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
#SBATCH --job-name=anch-screen
#SBATCH --gres=shard:4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#
# Stage 0. Measure per-problem pass rate so we can keep only the problems the
# model is genuinely uncertain about. Resampling importance is undefined where
# the answer distribution is a point mass.
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
python 00_screen.py --model "${MODEL:-allenai/Olmo-3-7B-Think}" --limit-math "${LIMIT_MATH:-400}" --gpu-frac "${GPU_FRAC:-0.85}"
echo "done: $(date --iso-8601=seconds)"
