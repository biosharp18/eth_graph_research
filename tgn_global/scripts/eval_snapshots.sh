#!/usr/bin/env bash
# Evaluate the periodic snapshots of a long run: for each epoch N given,
# assemble figures/deep/<name>_ep<N>/tgn_seed{s}.pt and run DGB standard +
# the two DGB modes (the metrics that move with the epoch dial).
# Usage: eval_snapshots.sh <name> "<seeds>" "<epochs>"
set -euo pipefail
name=$1; seeds=$2; epochs=$3
D=tgn_global/figures/deep
P=/data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet
export UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv
for ep in $epochs; do
  M=$D/${name}_ep$ep; mkdir -p $M
  for s in $seeds; do cp $D/$name/s$s/tgn_seed${s}_ep$ep.pt $M/tgn_seed$s.pt; done
  uv run --group ml python -m tgn.evaluate_dgb --parquet $P --models $M --seeds $seeds --out $M/dgb.json
  uv run --group ml python tgn_global/scripts/run_dgb_modes.py --parquet $P --seeds $seeds --no-edgebank --model ${name}_ep$ep=$M --out $M/modes.json
  echo "snapshot $name ep$ep done"
done
