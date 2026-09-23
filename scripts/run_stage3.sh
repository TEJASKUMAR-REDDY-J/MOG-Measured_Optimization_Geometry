#!/usr/bin/env bash
# Stage 3 chain, in dependency order. Each generated config is committed before the
# run that uses it, so every run records a clean git state.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { python -m scripts.run_experiment --config "$1" 2>&1 | grep -viE "RequestsDependencyWarning|warnings.warn"; }
commit_cfg() { git add configs/1m && { git diff --cached --quiet || git commit -q -m "$1"; }; }

run configs/1m/exp150_lr_tuning.yaml
python -m scripts.make_stage3_configs extend
commit_cfg "Add exp151 LR edge-extension config generated from exp150"
if python -c "import yaml,sys; sys.exit(0 if yaml.safe_load(open('configs/1m/exp151_lr_edge_extension.yaml'))['arms'] else 1)"; then
  run configs/1m/exp151_lr_edge_extension.yaml
fi
python -m scripts.make_stage3_configs eval
commit_cfg "Add exp160 baseline and exp100 atlas configs with tuned LRs"
run configs/1m/exp100_atlas.yaml
run configs/1m/exp200_oracle.yaml
python -m scripts.make_stage3_configs mog_tuning
commit_cfg "Add exp290 MOG-oracle LR tuning config"
run configs/1m/exp290_mog_lr_tuning.yaml
python -m scripts.make_stage3_configs mog_eval
commit_cfg "Add exp300 MOG-oracle evaluation config"
run configs/1m/exp160_baselines.yaml
run configs/1m/exp300_mog.yaml
echo STAGE3_DONE
