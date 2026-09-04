#!/bin/bash
#SBATCH --job-name=anch-lenprobe
#SBATCH --account=samiran2
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=shard:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.out
#SBATCH --error=/home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/logs/%x_%j.err
source /home/samiran2/Ramen/1_ramen/extra/MATS_NeelNada_Application/1_application/slurm/_common.sh
python 12_length_probe.py --model "${MODEL:-allenai/Olmo-3-7B-Think}" \
  --n-problems "${N_PROBLEMS:-60}" --samples "${SAMPLES:-8}" \
  --max-tokens "${MAX_TOKENS:-12288}" --levels "${LEVELS:-4,5}" ${EXTRA:-}
echo "done: $(date --iso-8601=seconds)"
