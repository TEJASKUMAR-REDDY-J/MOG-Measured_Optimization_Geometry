#!/usr/bin/env bash
# Redesign experiments (pre-registered in docs/EXPERIMENT_LOG.md). Additive: reads earlier
# results, never modifies them. Generated configs are committed before the run that uses them.
set -euo pipefail
cd "$(dirname "$0")/.."
run() { python -m scripts.run_experiment --config "$1" 2>&1 | grep -viE "RequestsDependencyWarning|warnings.warn"; }
run configs/redesign/exp430_h4_randpart_tuning.yaml
python -m scripts.make_stage3_configs h4_eval
git add configs/redesign && { git diff --cached --quiet || git commit -q -m "Add exp431 H4 control evaluation config from exp430 tuning"; }
run configs/redesign/exp431_h4_randpart.yaml
run configs/redesign/exp440_h5_vocab.yaml
run configs/redesign/exp420_additivity.yaml
run configs/redesign/exp400_h2prime_ckpts.yaml
run configs/redesign/exp410_h2prime_oracle.yaml
echo REDESIGN_DONE
