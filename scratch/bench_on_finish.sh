#!/usr/bin/env bash
# Waits for the augmented training run to exit, then benchmarks its best-val
# checkpoint against the vanilla and inno2 baselines.
#
# Baselines are read from the main checkout by absolute path; nothing is written
# there. See docs/viewpoint-augmentation-results.md.
set -uo pipefail

W=/home/kna/DiffCSPNew/.claude/worktrees/agent-a6fe1f06b99f23e27
PY=/home/kna/DiffCSPNew/.venv/bin/python
MAIN=/home/kna/DiffCSPNew/runs/bench/mp20
TRAIN_PID="${1:?usage: bench_on_finish.sh <training-pid>}"
CKPT=$W/runs/aug/wyckoff_aug_best.pt
LABEL="asymm-aug"

cd "$W"
echo "=== waiting for training PID $TRAIN_PID ==="
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
echo "=== training exited at $(date -Is) ==="

if [ ! -f "$CKPT" ]; then echo "FATAL: no checkpoint at $CKPT"; exit 1; fi
"$PY" -c "
import torch
d = torch.load('$CKPT', map_location='cpu', weights_only=False)
print('best-val checkpoint: epoch', d.get('epoch'), 'val_loss', d.get('val_loss'))
"

echo "=== sampling ==="
CUDA_VISIBLE_DEVICES=0 "$PY" run.py bench/run_diffusion.py --regime wyckoff \
  --ckpt "$CKPT" --inits "$MAIN/inits.pkl" --batch_size 128 \
  --out runs/aug/pred_aug.pkl || exit 1

echo "=== scoring ==="
"$PY" run.py bench/evaluate.py --inits "$MAIN/inits.pkl" \
  --preds "$LABEL=runs/aug/pred_aug.pkl" --n_jobs 20 \
  --out runs/aug/results_aug.pkl || exit 1

echo "=== multiplicity x DoF breakdown (headline: mult>1 & DoF>=9) ==="
"$PY" run.py bench/multiplicity_analysis.py --inits "$MAIN/inits.pkl" \
  --results "$MAIN/results_inno2_converged.pkl" runs/aug/results_aug.pkl \
  --regimes "vanilla-diffcsp" "asymm-inno2 (converged)" "$LABEL" \
  --compare "vanilla-diffcsp" "$LABEL"

echo "=== same comparison against the unaugmented asymmetric baseline ==="
"$PY" run.py bench/multiplicity_analysis.py --inits "$MAIN/inits.pkl" \
  --results "$MAIN/results_inno2_converged.pkl" runs/aug/results_aug.pkl \
  --regimes "asymm-inno2 (converged)" "$LABEL" \
  --compare "$LABEL" "asymm-inno2 (converged)"

echo "=== viewpoint spread (mechanism check; baseline 0.326 / 0.490 / 0.625) ==="
CUDA_VISIBLE_DEVICES=0 "$PY" run.py bench/viewpoint_gauge.py --ckpt "$CKPT" \
  --inits "$MAIN/inits.pkl"

echo "=== clashes and cell volume (baseline: 18.8% clash, 0.61x vanilla volume) ==="
"$PY" run.py bench/structure_diagnostics.py --inits "$MAIN/inits.pkl" \
  --preds "vanilla=$MAIN/pred_cspnet.pkl" \
          "asymm-inno2=$MAIN/pred_wyckoff_inno2_converged.pkl" \
          "$LABEL=runs/aug/pred_aug.pkl"

echo "=== BENCHMARK COMPLETE $(date -Is) ==="
