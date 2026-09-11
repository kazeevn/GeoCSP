# Viewpoint augmentation: implementation and pilot results

Implements the "learned/statistical" option from `docs/asymmetric-unit-deficit.md` §6/§7 —
randomise which orbit member is the anchor during training, so the model has to learn the
invariance the full-cell graph gets for free.

**Status: implemented and verified; full training not completed.** The pilot could not
resolve the effect, for a reason worth recording (below), and a matched full run is ~7 h
under current GPU contention. What is established is the implementation and its
correctness gates, plus a matched-budget signal that is directionally right but weak.

---

## 1. What was implemented

`WyckoffDiffusion(augment_viewpoint=True)`, wired to `--augment-viewpoint` in
`diffcsp/cli/train.py`. Off by default.

The parent's suggested route — rewrite the ops as `M'_j = M_j @ pinv(M_r)` — **does not
work**, and it is worth saying why since the algebra looks sound. The identity
`M_j M_r^+ M_r x = M_j x` does hold. But the model stores coordinates wrapped into
`[0,1)`, so the available anchor position is `M_r x - n` for an integer `n`, and the
reconstruction then picks up `M_j M_r^+ (n,0)`. That correction is not a lattice vector,
because `pinv(R_r)` is not an integer matrix, so the cell no longer reconstructs mod 1.
Measured: max deviation 0.5 (i.e. maximal) over 142530 atom re-anchorings, against a
baseline consistency of 8.9e-08 for the stored ops. `n` also changes as the anchor
diffuses across a cell boundary, so it cannot be absorbed into a fixed op.

What is implemented instead leaves `ops` alone and moves only two things:

1. the **edge reference position** becomes the chosen member's noised position,
   `input_full_coords[rep_atom]` — which is by construction the exact image of the noised
   canonical anchor, so no wrapping issue arises;
2. the **head's output frame**. The prediction comes out in member `r`'s frame and is
   mapped back with the already-stored `ops_inv[r] = pinv(R_r)`, then projected onto the
   canonical tangent space. This is done by folding `P0 @ pinv(R_r)` into the
   `site_projectors` argument the decoder already applies, so no new code path.

Noise and target stay in the canonical frame, which keeps `loss_coord` on the same scale
as an unaugmented run. Validation runs under `eval()`, where the augmentation is inert, so
**val loss is directly comparable between augmented and unaugmented runs** and best-val
checkpoint selection is done on the deployed (unaugmented) objective.

### Sites that are skipped

Sites whose canonical projector is oblique keep their canonical anchor. `pinv(R_r) R_r` is
the orthogonal projector onto the *row* space of `R_r`, which coincides with the tangent
space only when the projector is symmetric; on oblique sites the map-back is not an
inverse and would corrupt the target. Measured over 10041 sites of the benchmark set:

| | sites | round-trip `pinv(R_r) R_r v == v` |
|---|---:|---|
| symmetric projector | 9527 | exact, max error 8.9e-16 |
| oblique projector | 514 | fails, all 514 |

The failures are exactly the oblique sites — no symmetric site fails, no oblique site
passes — and they are all DoF 1-2 special positions. This is the same population as
§5.1, 1.1% of the DoF >= 9 structures that carry the deficit, so skipping them costs
almost nothing in the regime of interest. 5893 of 6407 sites with replicas (92%) remain
augmentable.

## 2. Correctness gates

| gate | result |
|---|---|
| re-anchored ops reproduce the cell | see above — `pinv` route rejected, frame route exact |
| projector idempotent, `round(trace) == DoF` | max `\|P^2-P\|` 1.2e-08, trace exact |
| `eval()` bit-identical to unaugmented | yes, 1.29284716 vs 1.29284716 |
| `train()` differs, and only in `loss_coord` | yes: coord 0.8257 -> 0.8363, lattice 0.467132 -> 0.467123 |
| gradients flow | 53 parameter tensors with nonzero grad |
| test suite | 34 passed |

## 3. Why the pilot could not resolve the effect

The gate was "viewpoint spread must fall clearly after ~25 epochs". It cannot, because
**spread is dominated by training maturity, not by augmentation**:

| orbit size m | augmented, best @ ep 19 | unaugmented control, best @ ep 9 | converged baseline, ep 510 |
|---:|---:|---:|---:|
| 2 | 0.908 | 1.038 | **0.326** |
| 3-4 | 1.097 | 1.215 | **0.490** |
| 5-8 | 1.236 | 1.343 | **0.625** |

The two pilot columns are **not epoch-matched** — the augmented checkpoint had ~10 more
epochs than the control when measured, which on its own could account for the whole
difference. Do not read the 12% gap as an augmentation effect.

(two-viewpoint relative difference; `m == 1` is 0.000 in every run, confirming the probe)

Both 25-epoch pilots sit at 0.9-1.3 while the converged unaugmented baseline is at
0.33-0.63. So an early checkpoint is *further* from viewpoint invariance than a trained
one regardless of augmentation, and the augmentation's contribution is a ~12% relative
difference on top of that — within what an epoch mismatch alone could explain, since the
augmented pilot had ~10 more epochs than the control at the time of measurement.

The honest reading: **this pilot design cannot answer the question.** A spread comparison
is only meaningful between checkpoints trained to comparable maturity.

## 4. What the pilot does show

Augmentation is not harmful. Validation runs under `eval()` with the augmentation inert,
so these are like-for-like on the identical objective:

| epoch | augmented | unaugmented control |
|---:|---:|---:|
| 4 | 0.7448 | **0.7414** |
| 9 | **0.7293** | 0.7480 |
| 14 | **0.7037** | 0.7138 |
| 19 | 0.7948 | (control still running) |
| 24 | 0.6900 | (control still running) |

The augmented run is ahead at two of the three matched epochs, by 0.019 and 0.010. That is
**not a reliable difference**: the augmented run's own epoch-19 value (0.7948) is 0.105
above its epoch-14 value, so run-to-run noise on this objective is an order of magnitude
larger than the gap being claimed. The defensible statement is that augmentation does not
degrade the unaugmented validation loss at this budget — nothing stronger.

## 5. What is still needed

A matched full run. At 424 iterations/epoch and ~14 s/epoch uncontended (~25 s/epoch with
the concurrent PaiNN job on GPU 0), 1000 epochs is **3.9 h alone, ~7 h contended**. Then:

```bash
python run.py bench/run_diffusion.py --regime wyckoff --ckpt <aug_best.pt> \
  --inits /home/kna/DiffCSPNew/runs/bench/mp20/inits.pkl --out runs/aug/pred.pkl
python run.py bench/evaluate.py --inits /home/kna/DiffCSPNew/runs/bench/mp20/inits.pkl \
  --preds "asymm-aug=runs/aug/pred.pkl" --n_jobs 20 --out runs/aug/results.pkl
python run.py bench/multiplicity_analysis.py \
  --inits /home/kna/DiffCSPNew/runs/bench/mp20/inits.pkl \
  --results /home/kna/DiffCSPNew/runs/bench/mp20/results_inno2_converged.pkl runs/aug/results.pkl \
  --regimes "vanilla-diffcsp" "asymm-inno2 (converged)" "asymm-aug" \
  --compare "vanilla-diffcsp" "asymm-aug"
python run.py bench/viewpoint_gauge.py --ckpt <aug_best.pt>
python run.py bench/structure_diagnostics.py --preds "asymm-aug=runs/aug/pred.pkl"
```

Primary metric: the multiplicity > 1 & DoF >= 9 per-trial gap, baseline vanilla +8.1
(91 vs 43 discordant, p = 4.1e-05). Secondary: does the viewpoint spread at ep ~510 fall
below the baseline's 0.326 / 0.490 / 0.625 — that is the mechanism check, and it is only
interpretable against a checkpoint of comparable maturity.
