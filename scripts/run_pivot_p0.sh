#!/usr/bin/env bash
# Pivot Phase 0 (pre-registered in docs/EXPERIMENT_LOG.md). Additive: never modifies earlier results.
# Generated configs are committed before the run that uses them.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { python -m scripts.run_experiment --config "$1" 2>&1 | grep -viE "RequestsDependencyWarning|warnings.warn"; }
commit_cfgs() { git add configs/pivot && { git diff --cached --quiet || git commit -q -m "$1"; }; }
run configs/pivot/exp504_p0_adamw_ckpts.yaml
run configs/pivot/exp500_p0_tuning.yaml
run configs/pivot/exp510_p0_bpe_tuning.yaml
python -m scripts.make_pivot_configs p0_extend
commit_cfgs "Add Phase 0 LR edge-extension configs from exp500/exp510 tuning"
for c in configs/pivot/exp501_p0_edge.yaml configs/pivot/exp511_p0_bpe_edge.yaml; do [ -f "$c" ] && run "$c"; done
python -m scripts.make_pivot_configs p0_eval
commit_cfgs "Add Phase 0 evaluation configs from tuned LRs"
run configs/pivot/exp502_p0_eval.yaml
run configs/pivot/exp505_p0_oracle_units.yaml
run configs/pivot/exp512_p0_bpe_eval.yaml
echo P0_DONE
