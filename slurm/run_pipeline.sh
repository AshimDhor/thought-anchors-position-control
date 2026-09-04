#!/usr/bin/env bash
# Submit the whole post-screening pipeline as a dependency chain.
#
# The 4-shard-per-user cap means these run one at a time anyway, but chaining on
# afterok means a failure stops the chain instead of feeding empty inputs to the
# next stage.
#
# Usage: slurm/run_pipeline.sh [MODEL] [LABELLER] [N_PROBLEMS]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-allenai/Olmo-3-7B-Think}"
LABELLER="${2:-Qwen/Qwen3-8B}"
N_PROBLEMS="${3:-20}"

j1=$(sbatch --parsable --export=ALL,MODEL="$MODEL",N_PROBLEMS="$N_PROBLEMS" slurm/traces.sbatch)
echo "traces        : $j1"

j2=$(sbatch --parsable --dependency=afterok:$j1 \
      --export=ALL,MODEL="$MODEL",ARM=main slurm/resample.sbatch)
echo "resample main : $j2  (after $j1)"

j3=$(sbatch --parsable --dependency=afterok:$j2 \
      --export=ALL,MODEL="$MODEL",ARM=filler slurm/resample.sbatch)
echo "resample filler: $j3  (after $j2)"

j4=$(sbatch --parsable --dependency=afterok:$j3 \
      --export=ALL,MODEL="$MODEL" slurm/analyze.sbatch)
echo "analyze       : $j4  (after $j3)"

j5=$(sbatch --parsable --dependency=afterok:$j4 \
      --export=ALL,MODEL="$MODEL",LABELLER="$LABELLER" slurm/labels.sbatch)
echo "labels+cats   : $j5  (after $j4)"

j6=$(sbatch --parsable --dependency=afterok:$j3 \
      --export=ALL,MODEL="$MODEL" slurm/indist.sbatch)
echo "filler check  : $j6  (after $j3)"

echo
echo "watch with: squeue -u \$USER -o '%.8i %.16j %.9T %.10M %R'"
