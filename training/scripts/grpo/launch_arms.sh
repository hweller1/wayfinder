#!/usr/bin/env bash
# Launch GRPO sweep arms. Usage: scripts/grpo/launch_arms.sh a b c d   (each arm: its own Modal app wayfinder-grpo-<arm>, driver log artifacts/grpo/<arm>.log)
# Defaults: 16 tasks x 8 per step (smoke: a 32x8 step needs ~78 min of 2xH200 training), 4 H100 rollouts, 4 H200 trainer per arm.
# a: default design from v3.  b: KL coef halved (0.01), lr x1.5 (7.5e-5).  c: a + precision term (-0.3 x fraction of reported files that are not gold).
# d: a's exact config starting from SFT v4.   Spend gate checked per arm before launch (refuses -> stop).
set -euo pipefail
cd "$(dirname "$0")/../.."
for arm in "$@"; do
  case "$arm" in
    a) extra="--lr 5e-5 --beta 0.02 --precision-coef 0.0 --start /data/sft/v3_merged" ;;
    b) extra="--lr 7.5e-5 --beta 0.01 --precision-coef 0.0 --start /data/sft/v3_merged" ;;
    c) extra="--lr 5e-5 --beta 0.02 --precision-coef 0.3 --start /data/sft/v3_merged" ;;
    d) extra="--lr 5e-5 --beta 0.02 --precision-coef 0.0 --start /data/sft/v4_merged" ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  GRPO_RUN="$arm" GRPO_START="$(echo "$extra" | sed -n 's/.*--start \([^ ]*\).*/\1/p')" nohup modal run wayfinder/grpo/app.py --run "$arm" --steps "${GRPO_STEPS:-40}" --workers "${GRPO_WORKERS:-4}" --train-gpus "${GRPO_TRAIN_GPUS:-4}" --tasks-per-step "${GRPO_TASKS:-16}" \
    --budget-usd 700 $extra > "artifacts/grpo/$arm.log" 2>&1 &
  echo "launched arm $arm (pid $!)"
  sleep 20
done
