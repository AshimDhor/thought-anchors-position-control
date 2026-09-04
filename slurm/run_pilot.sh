#!/usr/bin/env bash
# Pilot: take one quarter of the traces all the way through on real data before
# committing to the full sweep.
#
# The sweep is the expensive stage, and a bug in it is only visible downstream.
# Sharding is by trace, so shard 0 of 4 is a real, self-contained subset: run it,
# analyse it, look at the figures and the random examples, and only then launch
# the remaining shards. 97_merge_shards.py stitches them together afterwards, so
# nothing done here is thrown away.
#
# Usage: slurm/run_pilot.sh [MODEL]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-allenai/Olmo-3-7B-Think}"

j1=$(sbatch --parsable --export=ALL,MODEL="$MODEL",ARM=main,SHARD=0,N_SHARDS=4 \
      slurm/resample.sbatch)
echo "pilot main   (shard 0/4): $j1"

j2=$(sbatch --parsable --dependency=afterok:$j1 \
      --export=ALL,MODEL="$MODEL",ARM=filler,SHARD=0,N_SHARDS=4 slurm/resample.sbatch)
echo "pilot filler (shard 0/4): $j2  (after $j1)"

echo
echo "when both finish:"
echo "  python code/97_merge_shards.py --model $MODEL --arm main"
echo "  python code/97_merge_shards.py --model $MODEL --arm filler"
echo "  sbatch --export=ALL,MODEL=$MODEL slurm/analyze.sbatch"
echo "then inspect figures/ and writeup/random_examples.md before running the rest."
