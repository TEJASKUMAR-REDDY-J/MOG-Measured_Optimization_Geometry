#!/usr/bin/env bash
# Pivot Phase 2 (pre-registered in docs/EXPERIMENT_LOG.md). Additive; generated configs are committed before use.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { python -m scripts.run_experiment --config "$1" 2>&1 | grep --line-buffered -viE "RequestsDependencyWarning|warnings.warn"; }
commit_cfgs() { git add configs/pivot && { git diff --cached --quiet || git commit -q -m "$1"; }; }
run configs/pivot/exp506_p0_lm_stats.yaml
for t in 600_p2_sup 610_p2_ssl 620_p2_flow 630_p2_rl; do run configs/pivot/p2/exp${t}_tuning.yaml; done
python -m scripts.make_pivot_configs p2_extend
commit_cfgs "Add Phase 2 LR edge-extension configs from tuning"
for c in configs/pivot/p2/exp6?1_*_edge.yaml; do [ -f "$c" ] && run "$c"; done
python -m scripts.make_pivot_configs p2_eval
commit_cfgs "Add Phase 2 evaluation, checkpoint, oracle and statistics configs from tuned LRs"
for t in 600_p2_sup 610_p2_ssl 620_p2_flow 630_p2_rl; do
  base=${t%%_*}; tag=${t#*_}
  run configs/pivot/p2/exp${t}_eval.yaml
  run configs/pivot/p2/exp$((base + 3))_${tag}_ckpts.yaml
  run configs/pivot/p2/exp$((base + 5))_${tag}_stats.yaml
  run configs/pivot/p2/exp$((base + 4))_${tag}_oracle.yaml
done
run configs/pivot/exp507_p0_lm_block_oracle.yaml
run configs/pivot/p2/exp640_p2_h1_lopo.yaml
echo P2_DONE
