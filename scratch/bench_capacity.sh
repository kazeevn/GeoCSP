#!/usr/bin/env bash
# Waits for both capacity-experiment arms to exit, then benchmarks each best-val
# checkpoint against the vanilla and inno2 baselines.
#
# Baselines are read from the main checkout by absolute path; nothing is written there.
set -uo pipefail

W=/home/kna/DiffCSPNew/.claude/worktrees/agent-a6fe1f06b99f23e27
PY=/home/kna/DiffCSPNew/.venv/bin/python
MAIN=/home/kna/DiffCSPNew/runs/bench/mp20
PID864="${1:?usage: bench_capacity.sh <pid864> <pid512>}"
PID512="${2:?usage: bench_capacity.sh <pid864> <pid512>}"

cd "$W"
echo "=== waiting for arms: h864 PID $PID864, h512 PID $PID512 ==="
while kill -0 "$PID864" 2>/dev/null || kill -0 "$PID512" 2>/dev/null; do sleep 120; done
echo "=== both arms exited at $(date -Is) ==="

run_arm () {
  local h=$1 ckpt=$2 label=$3
  echo ""
  echo "############ arm h=$h  ($label) ############"
  if [ ! -f "$ckpt" ]; then echo "MISSING checkpoint $ckpt"; return 1; fi
  "$PY" -c "
import torch
d = torch.load('$ckpt', map_location='cpu', weights_only=False)
print('best-val: epoch', d.get('epoch'), 'val_loss', d.get('val_loss'))
"
  CUDA_VISIBLE_DEVICES=0 "$PY" run.py bench/run_diffusion.py --regime wyckoff \
    --ckpt "$ckpt" --hidden_dim "$h" --num_layers 6 --inits "$MAIN/inits.pkl" \
    --batch_size 128 --out "runs/cap/pred_$label.pkl" || return 1
  "$PY" run.py bench/evaluate.py --inits "$MAIN/inits.pkl" \
    --preds "$label=runs/cap/pred_$label.pkl" --n_jobs 20 \
    --out "runs/cap/results_$label.pkl" || return 1
  echo "--- vs vanilla (headline: mult>1 & DoF>=9, baseline +8.1 / 91 vs 43) ---"
  "$PY" run.py bench/multiplicity_analysis.py --inits "$MAIN/inits.pkl" \
    --results "$MAIN/results_inno2_converged.pkl" "runs/cap/results_$label.pkl" \
    --regimes "vanilla-diffcsp" "asymm-inno2 (converged)" "$label" \
    --compare "vanilla-diffcsp" "$label"
  echo "--- layer divergence vs the full cell (this arm's own weights in both decoders) ---"
  CUDA_VISIBLE_DEVICES=0 "$PY" run.py bench/layer_divergence.py \
    --inits "$MAIN/inits.pkl" --ckpt "$ckpt" --hidden_dim "$h" --num_layers 6 2>&1 | tail -12
  echo "--- viewpoint spread (baseline 0.326 / 0.490 / 0.625) ---"
  CUDA_VISIBLE_DEVICES=0 "$PY" run.py bench/viewpoint_gauge.py --ckpt "$ckpt" \
    --inits "$MAIN/inits.pkl" --hidden_dim "$h" --num_layers 6 2>&1 | sed -n '/B. viewpoint/,$p'
}

run_arm 864 runs/cap/wyckoff_h864_best.pt "asymm-h864"
run_arm 512 runs/cap/wyckoff_h512_best.pt "asymm-h512"

echo ""
echo "############ both arms together vs baselines ############"
"$PY" run.py bench/multiplicity_analysis.py --inits "$MAIN/inits.pkl" \
  --results "$MAIN/results_inno2_converged.pkl" \
            runs/cap/results_asymm-h864.pkl runs/cap/results_asymm-h512.pkl \
  --regimes "vanilla-diffcsp" "asymm-inno2 (converged)" "asymm-h512" "asymm-h864" \
  --compare "asymm-h864" "asymm-h512"

echo "--- clashes and cell volume ---"
"$PY" run.py bench/structure_diagnostics.py --inits "$MAIN/inits.pkl" \
  --preds "vanilla=$MAIN/pred_cspnet.pkl" \
          "asymm-inno2=$MAIN/pred_wyckoff_inno2_converged.pkl" \
          "asymm-h512=runs/cap/pred_asymm-h512.pkl" \
          "asymm-h864=runs/cap/pred_asymm-h864.pkl"

echo "=== CAPACITY BENCHMARK COMPLETE $(date -Is) ==="
