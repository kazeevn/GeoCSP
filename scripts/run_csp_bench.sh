#!/usr/bin/env bash
# CSP benchmark, Wyckoff-gene track, on the MP-20 benchmark set (docs/benchmark.md).
#
#   1. 20 pyxtal.from_random draws per gene
#   2. samples: the unrefined draw (control) and DiffCSP-Geo (500e)
#   3. PET-OAM-XL relaxation + ORB (orb_conserv_inf) energies / E_hull, for the
#      samples and the ground truth, in SHARDS processes per GPU
#   4. scoring, logged to W&B
#
# Needs the csp-bench extra:
#   DIFFCSP_EXTRAS="orb wandb dev csp-bench" scripts/platforms/zeus/env_init.sh
#
# Every stage skips work whose output exists, and mlip_relax.py resumes its
# shards, so the script can be rerun after an interruption.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

GPUS=${GPUS:-1}            # comma-separated device ids for the relaxations
SHARDS_PER_GPU=${SHARDS_PER_GPU:-2}
TRIALS=${TRIALS:-20}
OUT=${OUT:-runs/bench/mp20}
GEO_CKPT=${GEO_CKPT:-runs/mp20_geo/geo_mp20_500e_best.pt}
INITS=$OUT/inits_t$TRIALS.pkl
# Several relaxation processes share each GPU; keep their CPU threads from oversubscribing.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

[[ -f $INITS ]] || uv run python bench/gen_init.py --benchset $OUT/benchset.pkl --trials "$TRIALS" --out "$INITS"
[[ -f $OUT/pred_none_t$TRIALS.pkl ]] || uv run python bench/run_none.py --inits "$INITS" --out $OUT/pred_none_t$TRIALS.pkl
[[ -f $OUT/pred_geo_t$TRIALS.pkl ]] || CUDA_VISIBLE_DEVICES=${GPUS%%,*} uv run python bench/run_diffusion.py \
    --regime geo --ckpt "$GEO_CKPT" --inits "$INITS" --batch_size 128 --out $OUT/pred_geo_t$TRIALS.pkl

# relax <name> <mlip_relax.py source args...>: SHARDS_PER_GPU shards on every GPU.
relax() {
    local name=$1; shift
    IFS=, read -ra gpus <<< "$GPUS"
    local n=$(( ${#gpus[@]} * SHARDS_PER_GPU )) i=0 pids=()
    for gpu in "${gpus[@]}"; do
        for _ in $(seq "$SHARDS_PER_GPU"); do
            CUDA_VISIBLE_DEVICES=$gpu uv run python bench/mlip_relax.py --inits "$INITS" "$@" \
                --out "$OUT/relaxed_$name.pkl" --shard "$i/$n" > "$OUT/relaxed_$name.shard$i.log" 2>&1 &
            pids+=($!); i=$((i + 1))
        done
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

relax gt --gt
relax none_t$TRIALS --preds $OUT/pred_none_t$TRIALS.pkl
relax geo_t$TRIALS --preds $OUT/pred_geo_t$TRIALS.pkl

for regime in none geo; do
    label=$([[ $regime == none ]] && echo "pyxtal-only (control)" || echo "diffcsp-geo (500e)")
    uv run python bench/evaluate_csp.py --label "$label" --inits "$INITS" \
        --relaxed $OUT/relaxed_${regime}_t$TRIALS*.pkl --gt_relaxed $OUT/relaxed_gt*.pkl \
        --out $OUT/csp_${regime}_t$TRIALS.pkl
done
