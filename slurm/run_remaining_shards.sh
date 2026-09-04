#!/usr/bin/env bash
# Launch shards 1-3 after the pilot (shard 0) has been checked.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-allenai/Olmo-3-7B-Think}"

prev=""
for arm in main filler; do
  for shard in 1 2 3; do
    dep=""
    [ -n "$prev" ] && dep="--dependency=afterok:$prev"
    prev=$(sbatch --parsable $dep \
      --export=ALL,MODEL="$MODEL",ARM=$arm,SHARD=$shard,N_SHARDS=4 slurm/resample.sbatch)
    echo "$arm shard $shard/4: $prev"
  done
done
echo
echo "after all finish: merge shards, then sbatch slurm/analyze.sbatch and slurm/labels.sbatch"
