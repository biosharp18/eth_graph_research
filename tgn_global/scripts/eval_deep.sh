#!/usr/bin/env bash
# Evaluate one deep-MP variant trained per-seed under figures/deep/<name>/s<seed>/.
# Assembles <name>_last/ (epoch-49 weights) and <name>_best/ (combo-best) in the
# tgn_seed{s}.pt layout the eval tools expect, then runs: DGB standard, DGB
# modes (inductive_sym / test_recurring), legacy eval, deployment ranking.
# Usage: eval_deep.sh <name> "<seeds>" [extra eval flags, e.g. --k-inner 10]
set -euo pipefail
name=$1; seeds=$2; shift 2
D=tgn_global/figures/deep
P=/data1/gaorory/flow_graphs/OFAC_khop_value_graphs_k4/edges.parquet
export UV_PROJECT_ENVIRONMENT=/tmp/tgat-venv
for ck in last best; do
  mkdir -p $D/${name}_$ck
  for s in $seeds; do
    src=$D/$name/s$s/tgn_seed${s}.pt; [ $ck = last ] && src=$D/$name/s$s/tgn_seed${s}_last.pt
    cp $src $D/${name}_$ck/tgn_seed${s}.pt
  done
done
for ck in last best; do
  M=$D/${name}_$ck
  uv run --group ml python -m tgn.evaluate_dgb --parquet $P --models $M --seeds $seeds --out $M/dgb.json "$@"
  uv run --group ml python tgn_global/scripts/run_dgb_modes.py --parquet $P --seeds $seeds --no-edgebank --model $name=$M --out $M/modes.json "$@"
  uv run --group ml python -m tgn.evaluate --parquet $P --models $M --seeds $seeds --out $M/results.json "$@"
  uv run --group ml python -m tgn.ranking --parquet $P --models $M --seeds $seeds --out $M/ranking.json "$@"
done
echo "eval_deep $name done"
