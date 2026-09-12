# Is the asymmetric-unit deficit about compute? No.

Tests whether vanilla's advantage is FLOPs and per-node capacity rather than symmetry, by
widening the asymmetric-unit model until its edge-MLP FLOPs match vanilla's, against a
same-recipe control at the original width.

**Result: clean null.** The compute-matched arm and the control are indistinguishable —
−0.0 points overall (120 vs 121 discordant, p = 1.0) and +1.7 in the bin that carries the
deficit (62 vs 52, p = 0.4). Widening buys nothing. Capacity is not the mechanism.

A second finding fell out that matters more for reading the earlier results: **this recipe
produces models 7–9 points worse than the published inno2 baseline**, so the control was
load-bearing. Without it the honest-looking conclusion would have been "capacity makes
things worse", which is wrong.

---

## 1. The width was recomputed, not accepted

The proposed `h' = h·sqrt(N/K) ≈ 768` assumes edge-MLP cost scales as `h²`. It does not.
The first linear consumes `2h + 6 + D` inputs where `D = num_freqs·2·3 = 768` is the
sinusoidal edge featurisation — the same order as `h` itself — so cost per edge is

```
C(h) = h·(2h + 6 + D) + h·h = 3h² + h·(6 + D)
```

which grows slower than `h²` in relative terms: 512 → 768 multiplies MACs by 1.999, not
2.25. The rationale also used a mean of per-structure `K/N` (0.4678), where the quantity
that sets compute is the ratio of totals, `Σ(K·N)/Σ(N²) = 0.4047`; they differ because
`K/N` anti-correlates with `N` (ρ = −0.372).

| width | C(h)/C(512) | edge-MLP FLOPs vs vanilla | + node MLP | params |
|---:|---:|---:|---:|---:|
| 512 | 1.000 | 0.405 | 0.406 | 12,277,248 |
| 768 | 1.999 | 0.809 | 0.816 | 25,690,368 |
| **864** | 2.459 | **0.995** | 1.005 | 31,970,592 |
| 1024 | 3.330 | 1.348 | 1.364 | 43,953,152 |

768 would have left the arm 19% short of vanilla, enough to make a null answerable with
"you never matched the compute". The exact match is h\* = 866; **864** is the multiple of
32 and lands at 0.995×. Parameter counts predicted by `bench/flop_ratio.py` match the
trained models exactly, which validates the accounting.

Worth noting: matching vanilla's *FLOPs* takes 2.6× vanilla's *parameters*, because the
asymmetric graph has ~2.5× fewer edges to amortise them over.

## 2. Setup

Both arms `--augment-viewpoint` **off**, identical recipe: 1000 epochs, batch 64, lr 1e-3,
`eval_freq 10`, `save_freq 50`, 8 dataloader workers, co-scheduled on one GPU. Best-val
checkpoints, both at **epoch 610**.

| arm | width | params | best val loss |
|---|---:|---:|---:|
| capacity | 864 | 31,970,592 | 0.41612 |
| control | 512 | 12,277,248 | 0.42383 |
| *(published inno2 baseline)* | *512* | *12,277,248* | *0.40472 @ ep 510* |

## 3. The null

Per-trial match rate:

| subset | n | vanilla | inno2 baseline | **h512** | **h864** |
|---|---:|---:|---:|---:|---:|
| ALL | 1000 | 81.3% | 79.5% | 77.5% | 77.5% |
| multiplicity == 1 | 110 | 47.9% | 47.3% | 49.4% | 48.2% |
| multiplicity > 1 | 890 | 85.5% | 83.4% | 81.0% | 81.1% |
| &nbsp;&nbsp;...DoF <= 5 | 551 | 97.1% | 97.9% | 96.6% | 96.7% |
| &nbsp;&nbsp;...DoF 6-8 | 142 | 84.3% | 79.8% | 81.5% | 79.3% |
| &nbsp;&nbsp;**...DoF >= 9** | 197 | **53.8%** | **45.7%** | **36.9%** | **38.6%** |

Paired McNemar, h864 vs h512 — the comparison the experiment exists to make:

| subset | gap | discordant | p |
|---|---:|---|---:|
| ALL | −0.0 | 120 vs 121 | 1.0 |
| multiplicity > 1 | +0.1 | 111 vs 108 | 0.89 |
| **multiplicity > 1, DoF >= 9** | **+1.7** | 62 vs 52 | **0.4** |

2.46× the edge FLOPs, 2.6× the parameters, and the two land on top of each other. Against
vanilla both arms are *worse* than the baseline: +16.9 (129 vs 29, p = 3e-16) for h512 and
+15.2 (123 vs 33, p = 2.1e-13) for h864, against the baseline's +8.1.

## 4. The recipe confound — why the control was load-bearing

Both arms sit 7–9 points below the published inno2 baseline in the headline bin (36.9% and
38.6% vs 45.7%), and both have worse val loss (0.4238, 0.4161 vs 0.4047) despite training
1000 epochs to the baseline's 510. Something in this recipe — the exact optimiser
schedule, the `ReduceLROnPlateau` interaction with `eval_freq`, or seed handling — is
worse than whatever produced the baseline checkpoint, which is not reproducible from the
repo as it stands.

So **absolute comparisons between these arms and the published baseline are not valid**,
and neither are comparisons between them and the augmented run from the previous
experiment, which used the same recipe but a different one again from the baseline. The
h864-vs-h512 comparison *is* valid, because the two differ in exactly one thing.

### This overturns the augmentation result

The augmented run was compared against the published baseline, and its apparent −3.0 in the
headline bin was recipe, not augmentation. `asymm-h512` here is its true same-recipe
control. Paired McNemar, augmented vs that control:

| subset | gap | discordant | p |
|---|---:|---|---:|
| ALL | +1.2 | 146 vs 110 | 0.029 |
| **multiplicity == 1** (augmentation inert here) | **−0.6** | 14 vs 16 | **0.86** |
| multiplicity > 1 | +1.4 | 132 vs 94 | 0.014 |
| **multiplicity > 1, DoF >= 9** | **+5.8** | 80 vs 46 | **0.0031** |

**Viewpoint augmentation helps by +5.8 points in exactly the bin it targets, p = 0.0031** —
and does nothing at multiplicity 1 (p = 0.86), where by construction there is no orbit
member to randomise. That null is the internal control, and it is what a real effect must
look like: the benefit appears only where the augmentation is active, and scales into the
regime the deficit lives in.

Caveats: one run per arm, no seed replication, so the magnitude is soft. And it does not
rescue the architecture — the augmented model is still at 42.6% against vanilla's 53.8%.
Augmentation recovers roughly a third of the recipe-handicapped gap (+16.9 → +11.2), not
all of it.

The previous write-up's headline conclusion — "the mechanism works and it does not help" —
is **wrong on the second half**, and `docs/viewpoint-augmentation-results.md` is corrected
accordingly. The error was comparing against a baseline trained with an unknown, better
recipe rather than against a matched control. The same mistake would have made this
capacity experiment read as "capacity hurts".

## 5. Neither candidate mechanism tracks match rate

Layer-6 divergence from the full cell, each model's own weights in both decoders:

| weights | anchor state | coord score | headline-bin match |
|---|---:|---:|---:|
| vanilla | 2.51e-01 | 5.81e-01 | 53.8% |
| inno2 baseline | 6.85e-02 | — | 45.7% |
| augmented | 3.75e-02 | — | 42.6% |
| h512 | 4.60e-02 | 1.23e-01 | 36.9% |
| h864 | 4.79e-02 | 1.24e-01 | 38.6% |

Divergence spans 6.7× across these models and match rate does not follow it — the *least*
divergent model (augmented, 3.75e-02) and the most (vanilla, 2.51e-01) sit at opposite ends
of the match-rate range, in the wrong direction. Width leaves divergence essentially
unchanged (4.60 vs 4.79e-02), as it should, since width does not add nodes.

Viewpoint spread (two-viewpoint relative difference) is equally flat across widths:

| orbit size m | inno2 baseline | h512 | h864 |
|---:|---:|---:|---:|
| 2 | 0.326 | 0.318 | 0.363 |
| 3-4 | 0.490 | 0.496 | 0.468 |
| 5-8 | 0.625 | 0.639 | 0.653 |

## 6. Secondary diagnostics

At DoF >= 9, multiplicity > 1: clashes below 0.9× the true contact are 4.6% (vanilla),
18.8% (baseline), 28.9% (h512), 36.0% (h864). Median relative cell-volume error is 0.0642,
0.0522, 0.0522, 0.0590. The h864 arm is worse on both, which — given the match-rate null —
reads as the wider model fitting the training objective in a way that does not transfer,
not as a capacity effect.

## 7. Throughput: the asymmetric unit is not 14× cheaper on this distribution

Measured solo, forward+backward, batch of 64 crystals / 654 atoms:

| model | width | params | s/batch | vs asym-512 |
|---|---:|---:|---:|---:|
| asymmetric | 512 | 12,277,248 | 0.0168 | 1.00× |
| asymmetric | 864 | 31,970,592 | 0.0305 | 1.82× |
| full cell | 512 | 12,277,248 | 0.0216 | 1.28× |
| full cell | 864 | 31,970,592 | 0.0459 | 2.73× |

Two corrections to earlier claims:

- **The asymmetric unit is 1.28× faster than the full cell at matched width here, not
  14×.** The 14.14× in `wyckoff-innovations-study.md` §5 is for one hand-picked large
  crystal (N=48, K=2); over the MP-20 benchmark distribution (mean N=11.5, mean K=4.9) the
  advantage is far smaller. The compute-matched arm (asym-864, 0.0305 s) is in fact
  *slower* than vanilla itself (full-cell-512, 0.0216 s).
- An interim note in this worktree suggested the model was overhead-bound because the two
  arms cost near-identical wall clock when co-scheduled (30.0 vs 28.5 s/epoch). **That was
  an artefact of time-slicing and dataloader contention.** Measured solo, width costs
  1.82× for 2.46× FLOPs — sublinear, so partly overhead-bound, but it does track
  arithmetic. The capacity arm really did receive the compute it was given.

## 8. What this does and does not establish

Establishes: **matching vanilla's edge-MLP FLOPs, with 2.6× its parameters, recovers none
of the deficit.** Combined with the divergence table in §5, both mechanisms proposed so far
— symmetry/viewpoint dependence and compute/capacity — now have direct experimental
evidence against them.

Does **not** establish anything about the *number of hidden states*. Width adds FLOPs and
per-node capacity; it does not add nodes. The asymmetric graph still carries K node states
against vanilla's N, and that cannot be tested by widening, because a model with N
independent node states is vanilla. Testing it needs the structural route — giving
replicas individuated states at K-node cost, e.g. `hj = MLP([h_{s(j)}, embed(R_j)])` — which
is now the only untested hypothesis of the three.

The recipe gap in §4 should be closed first, or every future arm inherits the same
7–9 point handicap and the published baseline stays unreproducible.

## 9. Reproducing

```bash
python run.py bench/flop_ratio.py                    # width selection
python run.py bench/throughput.py                    # FLOP-bound vs overhead-bound

python run.py diffcsp/cli/train.py --model wyckoff --hidden_dim 864 \
  --epochs 1000 --batch_size 64 --lr 1e-3 --no-wandb --num_workers 8 \
  --ckpt_path runs/cap/wyckoff_h864.pt --eval_freq 10 --save_freq 50
# ...and the same with --hidden_dim 512

./scratch/bench_capacity.sh <pid864> <pid512>        # both arms, all diagnostics
```

Artifacts: `runs/cap/wyckoff_h{864,512}_best.pt` (both epoch 610),
`runs/cap/pred_asymm-h{864,512}.pkl`, `runs/cap/results_asymm-h{864,512}.pkl`,
`logs/cap_h{864,512}.log`, `logs/bench_capacity.log`.
