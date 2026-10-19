#!/usr/bin/env bash
# CSP benchmark, formula track, on the MP-20 benchmark set (docs/benchmark.md).
#
#   1. a composition-conditioned WyFormer samples TRIALS Wyckoff genes per formula
#   2. one pyxtal.from_random draw per gene
#   3. DiffCSP-Geo (500e) turns each draw into a structure
#   4. PET-OAM-XL relaxation + ORB (orb_conserv_inf) energies / E_hull, in
#      SHARDS_PER_GPU processes per GPU; the ground truth's relaxation is the gene
#      track's (same benchmark entries), produced by scripts/run_csp_bench.sh
#   5. scoring, logged to W&B
#
#   MODELS="uncond-cell eform" GPUS=0,1 scripts/run_csp_formula_bench.sh
#
# A model name with the suffix -cell gives the decoder the ground truth's atom count
# (the DiffCSP CSP protocol); without it the model chooses z. RELAX=0 stops after the
# raw samples and scores the classic metrics only.
#
# Needs the csp-bench extra (see scripts/run_csp_bench.sh). Every stage skips work
# whose output exists, and mlip_relax.py resumes its shards.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

GPUS=${GPUS:-0}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-2}
TRIALS=${TRIALS:-20}
BENCH=${BENCH:-runs/bench/mp20}
OUT=${OUT:-$BENCH/formula}
GEO_CKPT=${GEO_CKPT:-runs/mp20_geo/geo_mp20_500e_best.pt}
MODELS=${MODELS:-"uncond-cell eform-cell eform-cfg-cell eform"}
RELAX=${RELAX:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}

# Short name -> WyckoffTransformer W&B run (symmetry-advantage/WyckoffTransformer).
declare -A RUN=(
    [uncond]=csp-formula-hera-20261016-mp20_unconditional
    [eform]=csp-formula-hera-20261016-mp20_formation_energy
    [eform-cfg]=csp-formula-hera-20261016-mp20_formation_energy_cfg
)

relax() {  # relax <name> <inits> <mlip_relax.py source args...>
    local name=$1 inits=$2; shift 2
    IFS=, read -ra gpus <<< "$GPUS"
    local n=$(( ${#gpus[@]} * SHARDS_PER_GPU )) i=0 pids=()
    for gpu in "${gpus[@]}"; do
        for _ in $(seq "$SHARDS_PER_GPU"); do
            CUDA_VISIBLE_DEVICES=$gpu uv run python bench/mlip_relax.py --inits "$inits" "$@" \
                --out "$OUT/relaxed_$name.pkl" --shard "$i/$n" > "$OUT/relaxed_$name.shard$i.log" 2>&1 &
            pids+=($!); i=$((i + 1))
        done
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

mkdir -p "$OUT"
for m in $MODELS; do
    genes=$OUT/genes_$m.pkl inits=$OUT/inits_${m}_t$TRIALS.pkl pred=$OUT/pred_geo_${m}_t$TRIALS.pkl
    base=${m%-cell} cell=$([[ $m == *-cell ]] && echo --given_cell || true)
    [[ -f $genes ]] || CUDA_VISIBLE_DEVICES=${GPUS%%,*} uv run python bench/gen_csp_formula.py \
        --benchset "$BENCH/benchset.pkl" --wandb_run "${RUN[$base]}" --samples "$TRIALS" $cell --out "$genes"
    [[ -f $inits ]] || uv run python bench/gen_init.py --benchset "$BENCH/benchset.pkl" --trials "$TRIALS" \
        --genes "$genes" --out "$inits"
    [[ -f $pred ]] || CUDA_VISIBLE_DEVICES=${GPUS%%,*} uv run python bench/run_diffusion.py \
        --regime geo --ckpt "$GEO_CKPT" --inits "$inits" --batch_size 128 --out "$pred"
    if [[ $RELAX == 0 ]]; then
        uv run python bench/evaluate_csp.py --label "wyformer-$m + diffcsp-geo (500e), raw" --inits "$inits" \
            --preds "$pred" --wandb_tags formula_track --out "$OUT/csp_geo_${m}_t${TRIALS}_raw.pkl"
        continue
    fi
    relax "geo_${m}_t$TRIALS" "$inits" --preds "$pred"
    uv run python bench/evaluate_csp.py --label "wyformer-$m + diffcsp-geo (500e)" --inits "$inits" \
        --relaxed "$OUT"/relaxed_geo_${m}_t$TRIALS*.pkl --gt_relaxed "$BENCH"/relaxed_gt*.pkl \
        --wandb_tags formula_track --out "$OUT/csp_geo_${m}_t$TRIALS.pkl"
done
