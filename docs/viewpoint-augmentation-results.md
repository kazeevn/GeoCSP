# Viewpoint augmentation: it works on the mechanism and does not help

Implements the "learned/statistical" option from `docs/asymmetric-unit-deficit.md` §6/§7 —
randomise which orbit member is the anchor during training, so the model has to learn the
invariance the full-cell graph gets for free.

**Result: negative.** Trained to convergence (1000 epochs, best val at epoch 810) the
augmentation does reduce viewpoint dependence — the quantity it targets — by 11-23%
relative. It does not convert that into match rate. In the bin the change was aimed at,
multiplicity > 1 and DoF >= 9, vanilla's lead **widens from +8.1 to +11.2 points**. Against
the unaugmented asymmetric baseline the change is -3.0 points there (49 vs 67 discordant,
p = 0.11), i.e. no significant difference, with a consistent slight drift the wrong way
across clashes, cell volume and DoF <= 5.

The conclusion is not that the diagnosis in §7 of the deficit note was wrong — viewpoint
dependence is real and measurable, and this reduces it. It is that **reducing it by this
much is not worth anything on this benchmark**, which makes the structural routes (§6 of
that note) the ones left standing.

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

## 3. The pilot gate was invalid — do not repeat it

The gate was specified as "viewpoint spread must fall clearly after ~25 epochs, otherwise
the implementation is broken". **That gate cannot distinguish a working implementation
from a broken one, and a failure to meet it says nothing.** It was run, it was not met,
and the correct conclusion was that the gate was wrong rather than the code.

The reason is a confound: **spread is dominated by training maturity, not by
augmentation**.

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
is only meaningful between checkpoints trained to comparable maturity. An early checkpoint
is further from viewpoint invariance than a late one *regardless* of augmentation, and the
25-epoch window sits entirely inside that transient.

The methodological lesson, since the next person will otherwise repeat it: a cheap early
gate needs a quantity that is already near its asymptote at the gate's budget. Viewpoint
spread is not — it moves by a factor of ~3 between epoch 25 and epoch 510 on the
unaugmented model alone. Correctness of this change is better established by the direct
algebraic checks in §2 and `bench/viewpoint_checks/`, which are exact, instant, and do not
depend on training at all. Use those as the gate; use spread only as an end-of-training
mechanism check against a baseline of matched maturity.

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

## 5. Environment hazard: one venv shared by every checkout

Worth recording because it can silently invalidate a run rather than fail loudly.

There is a single virtualenv, `/home/kna/DiffCSPNew/.venv`, and **every checkout points at
it by symlink** — verified with `readlink -f`:

| checkout | `.venv` |
|---|---|
| `/home/kna/DiffCSPNew` | the real venv |
| `/home/kna/DiffCSPNew/.worktrees/painn` | symlink -> the real venv |
| this worktree | symlinked to it initially; removed, see below |

`diffcsp` is installed editable, and its path is recorded in one file,
`.venv/lib/python3.12/site-packages/_editable_impl_diffcsp.pth`, which contains a single
checkout path. **`uv run` rewrites that file to the current project root on every
invocation.** So with several worktrees active it thrashes: it was observed pointing at
this worktree, then at `.worktrees/painn`, then back at the main checkout, purely
depending on who ran `uv` last.

Consequences:

- `uv run` from inside a checkout is safe — it rewrites the path to that checkout first.
- Invoking `.venv/bin/python` **directly** is not: it imports whichever checkout won the
  last write. A training run started that way can silently train the wrong code.
- Symlinking the shared venv into a worktree is actively harmful, because the first
  `uv run` there repoints the shared file and changes what *other* checkouts import.

What this worktree does instead: no `.venv` at all, and a `run.py` launcher that inserts
the worktree root at `sys.path[0]` before running the target, so imports are independent
of the `.pth` no matter who last wrote it. Verified with `diffcsp.__file__` before
training.

The actual fix is a real `uv venv` per worktree, which **no checkout has done**. It was
not applied here: repointing the shared `.pth` is a write to a resource two live agents
depend on.

## 6. Results

Trained 1000 epochs, `--augment-viewpoint`, otherwise identical to the baseline config
(512 hidden, 6 layers, batch 64, lr 1e-3). Best val loss **0.41299 at epoch 810**, against
the unaugmented baseline's **0.40472 at epoch 510** — comparable, since validation runs
under `eval()` where the augmentation is inert. Wall clock ~5 h sharing GPU 0. Benchmarked
on the same 1000 representations x 3 draws.

### The mechanism did move

Two-viewpoint relative difference in the predicted score. Both models converged, so this
is the fair comparison the pilot could not make:

| orbit size m | baseline, ep 510 | augmented, ep 810 | change |
|---:|---:|---:|---:|
| 1 (null) | 0.001 | 0.003 | — |
| 2 | 0.326 | **0.290** | -11% |
| 3-4 | 0.490 | **0.419** | -14% |
| 5-8 | 0.625 | **0.484** | -23% |

The augmentation does what it was designed to do, and the effect grows with orbit size, as
it should. But it only shaves the dependence — an equivariant denoiser scores 0, and the
augmented model is still at 0.29-0.48.

### The match rate did not follow

Per-trial, paired on identical draws:

| subset | n | vanilla | asymm-inno2 | asymm-aug |
|---|---:|---:|---:|---:|
| ALL | 1000 | 81.3% | 79.5% | 78.7% |
| multiplicity == 1 | 110 | 47.9% | 47.3% | 48.8% |
| multiplicity > 1 | 890 | 85.5% | 83.4% | 82.4% |
| &nbsp;&nbsp;...DoF <= 5 | 551 | 97.1% | 97.9% | 97.2% |
| &nbsp;&nbsp;...DoF 6-8 | 142 | 84.3% | 79.8% | 80.3% |
| &nbsp;&nbsp;**...DoF >= 9** | 197 | **53.8%** | **45.7%** | **42.6%** |

Paired McNemar in the headline bin (multiplicity > 1, DoF >= 9, 591 trials):

| comparison | gap | discordant | p |
|---|---:|---|---:|
| vanilla vs asymm-inno2 (baseline) | +8.1 | 91 vs 43 | 4.1e-05 |
| **vanilla vs asymm-aug** | **+11.2** | 103 vs 37 | 2.2e-08 |
| asymm-aug vs asymm-inno2 | -3.0 | 49 vs 67 | 0.11 |

The aug-vs-baseline difference is **not significant** (p = 0.11 in the headline bin, 0.15
overall), so the honest statement is that augmentation left the match rate where it was,
not that it damaged it. What is clear is that it did not help.

### Secondary diagnostics, all drifting slightly the wrong way

| metric (DoF >= 9, mult > 1 unless noted) | vanilla | asymm-inno2 | asymm-aug |
|---|---:|---:|---:|
| clashes below 0.9x true contact | 4.6% | 18.8% | 21.3% |
| median rel. cell-volume error | 0.0642 | **0.0522** | 0.0652 |
| per-trial match, DoF <= 5 (mult > 1) | 97.1% | 97.9% | 97.2% |

The cell-volume regression is the most notable: the unaugmented asymmetric model's clear
advantage over vanilla on lattices (0.0522 vs 0.0642) is **erased** (0.0652). None of these
individually is significant, but all three point the same way, which is consistent with the
augmentation spending capacity on an invariance that does not pay for itself.

### Reading

The §7 diagnosis stands — viewpoint dependence is real, and this reduces it. What fails is
the inference that reducing it recovers the match rate. Either an 11-23% reduction is far
too little (the residual 0.29-0.48 is still enormous next to an equivariant model's 0), or
viewpoint dependence is a *symptom* of the collapse rather than the binding constraint.
§3 of the deficit note supports the latter: the two architectures' anchor representations
diverge by 37% in hidden state and 58% in coordinate score by layer 6, and that divergence
comes from replicas carrying individuated states through depth — something augmentation
cannot supply, because no amount of viewpoint sampling gives the collapsed graph more than
one state per orbit.

That leaves the structural options in §6 of the deficit note as the ones worth trying:
conditioning the message on the replica's operation (`hj = MLP([h_{s(j)}, embed(R_j)])`),
which restores individuated neighbour states at unchanged `K x N` cost, or equivariant
Cartesian features, which make the collapse lossless by construction rather than on
average.

## 7. Reproducing

Training: 424 iterations/epoch, ~14 s/epoch uncontended and ~25 s/epoch sharing GPU 0;
the run above took ~5 h wall clock for 1000 epochs.

```bash
python run.py diffcsp/cli/train.py --model wyckoff --augment-viewpoint \
  --epochs 1000 --batch_size 64 --lr 1e-3 --no-wandb --num_workers 8 \
  --ckpt_path runs/aug/wyckoff_aug.pt --eval_freq 10 --save_freq 50
```

Then `scratch/bench_on_finish.sh <training-pid>` runs everything below, or by hand:

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
(91 vs 43 discordant, p = 4.1e-05). Secondary: the viewpoint spread against the baseline's
0.326 / 0.490 / 0.625 — the mechanism check, interpretable only against a checkpoint of
comparable maturity, which is why it belongs here and not in a pilot (§3).

Artifacts from the run reported above, in this worktree:
`runs/aug/wyckoff_aug_best.pt` (epoch 810), `runs/aug/pred_aug.pkl`,
`runs/aug/results_aug.pkl`, `logs/full_aug.log`, `logs/bench_on_finish.log`.
