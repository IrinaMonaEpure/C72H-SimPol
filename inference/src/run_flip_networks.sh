#!/bin/bash
# Launch N independent, single-threaded shards of generate_flip_networks.py as
# separate OS processes (graph-tool deadlocks under Python multiprocessing on
# macOS, so we parallelise at the shell level instead). Resumable: re-running
# skips samples already saved.
#
# Usage (from inference/src/):
#   ./run_flip_networks.sh          # 12 shards (default)
#   ./run_flip_networks.sh 10       # 10 shards
set -euo pipefail
cd "$(dirname "$0")"

PY=/opt/homebrew/Caskroom/miniconda/base/envs/complexity72-simpol/bin/python
N=${1:-12}
mkdir -p flip_logs

echo "Launching $N single-threaded shards..."
pids=()
for k in $(seq 0 $((N - 1))); do
  OMP_NUM_THREADS=1 "$PY" generate_flip_networks.py --num-shards "$N" --shard "$k" \
    > "flip_logs/shard_${k}.log" 2>&1 &
  pids+=($!)
done
echo "Shard PIDs: ${pids[*]}"

wait
echo "All shards finished. Building per-country summaries..."
"$PY" generate_flip_networks.py --summarize
echo "ALL DONE."
