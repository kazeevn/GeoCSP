# Benchmarking crystal structure prediction

The harness in `bench/` measures how often a model recovers a known crystal from its
Wyckoff representation, and compares that against references that do less work — so
a match rate can be read against what the representation alone already gives away.

## The pipeline

```
test structure -> Wyckoff representation -> pyxtal.from_random x3 -> regime -> StructureMatcher
                 (pyxtal.from_seed)        (3 instantiations)                 (vs ground truth)
```

Every regime runs on **identical draws**: `crystal_graph_from_pyxtal` (split out of
`build_crystal_graph`) derives model inputs from one specific pyXtal instance rather
than triggering a fresh random draw per regime.

Scored with `StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)`, reported two ways:

- **match/trial** — fraction of all generated structures that match.
- **best-of-3** — fraction of representations matched by at least one of their trials.

### Stages

| script | does |
|---|---|
| `build_set.py` | samples test structures, derives Wyckoff representations, keeps matched ground truth |
| `gen_init.py` | the three `from_random` draws per representation |
| `run_none.py` | control: the unrefined draw |
| `run_diffusion.py` | `--regime cspnet \| orb \| wyckoff \| painn \| geo \| geov2 \| geo_orb` (plus `--rank_candidates`, `--orb_handoff_t`, `--noise_cutoff_t`, `--no-orbit-average`) |
| `run_relax.py` | symmetry-constrained ORB relaxation |
| `evaluate.py` | scores every regime against ground truth |
| `dof_analysis.py` | splits results by Wyckoff degrees of freedom |
| `multiplicity_analysis.py` | splits results by Wyckoff orbit multiplicity and runs paired McNemar tests |
| `structure_diagnostics.py` | measures shortest interatomic contacts, clash rates ($< 0.9 \times d_{\text{GT}}$), and relative cell-volume error |
| `layer_divergence.py` | compares full-cell vs asymmetric-unit hidden states and scores layer by layer |
| `viewpoint_gauge.py` | measures score sensitivity to which orbit member is chosen as the Wyckoff anchor |

`run_all.sh` and `run_lemat_eval.sh` drive the baseline MP-20 and LeMat comparisons; `scripts/train_geo_500e.sh`, `scripts/train_geov2_mp20.sh`, `scripts/train_geo_orb.sh`, `scripts/eval_geo_rank3.sh`, and `scripts/eval_geov2_rank3.sh` run the Geo-family benchmarks.

## What pyXtal supplies, and what it does not

`sample()` draws initial coordinates from `torch.rand` and the initial lattice from
`torch.randn`, then projects both onto the space group — coordinates through
`batch.ops`/`batch.anchor_index`, the lattice through `proj_k_to_spacegroup`. So
pyXtal supplies the **symmetry scaffold** (space group, Wyckoff orbits, affine
operations, atom count, species); the continuous values are noise. A `from_random`
draw's own coordinates are computed and then discarded.

This has a consequence for the three trials. The `sites` spec fixes the orbits, so
all three draws give an identical scaffold: for the diffusion regimes the trials are
three **sampler seeds**, while for relaxation and the control they are three
genuinely different starting geometries. The comparison is still fair — each regime
gets three attempts per representation — but the trials are not the same object.

---

## Result 1 — MP-20 test benchmark

1000 representations $\times$ 3 trials = 2,998 draws (`runs/bench/mp20/inits.pkl`, generated 2026-09-09).

### Provenance (Dates, Commits, W&B Runs)

| regime | date | code commit | W&B run (`symmetry-advantage/diffcsp`) | checkpoint |
|---|---|---|---|---|
| `pyxtal-only (control)` | 2026-09-10 | `e6fcb3d` | — | — |
| `orb-relax` | 2026-09-10 | `e6fcb3d` | — | frozen `orb-v3` (`orb_v3_direct_20_mpa`) |
| `orb-diffcsp` | 2026-09-10 | `2a2b54e` / `e6fcb3d` | [`vuvt0kab`](https://wandb.ai/symmetry-advantage/diffcsp/runs/vuvt0kab) (`classic-armadillo-67`, ep 500 best) | `runs/vuvt0kab/model-vuvt0kab:v19/orb_v3_diffcsp_mp20_opt3.pt` |
| `vanilla-diffcsp` | 2026-09-10 | `e6fcb3d` | [`zowdmqwo`](https://wandb.ai/symmetry-advantage/diffcsp/runs/zowdmqwo) (`diffcsp_mp_20_v10`, 1000e) | `data/mp-20/test_ckpt.pt` |
| `asymm-inno2` *(archived)* | 2026-09-11 | `57895fd` | local run (pre-default W&B) | `runs/mp20_wyckoff/wyckoff_inno2_best.pt` (ep 510) |
| `wyckoff-painn` *(archived)* | 2026-09-12 | `211c018` / `ac8a2f5` | local worktree run | `WyckoffPaiNN` 512x6 (ep 210) |
| `diffcsp-geo (500e)` | 2026-09-13 | `d7b1a7b` | [`7jdhrv2f`](https://wandb.ai/symmetry-advantage/diffcsp/runs/7jdhrv2f) (`expert-wood-72`, ep 320 best) | `runs/mp20_geo/geo_mp20_500e_best.pt` |
| `diffcsp-geov2 (400e)` | 2026-09-13 | `561bb8f` | [`say1phjr`](https://wandb.ai/symmetry-advantage/diffcsp/runs/say1phjr) (`toasty-wood-73`, ep 300 best EMA) | `runs/mp20_geov2/geov2_mp20_400e_best.pt` |
| `diffcsp-geo-orb (200e)` | 2026-09-13 | `561bb8f` | [`onl3oy96`](https://wandb.ai/symmetry-advantage/diffcsp/runs/onl3oy96) (`summer-dragon-74`, ep 180 best EMA, handoff $t=0.10$) | `runs/mp20_geo_orb/geo_orb_mp20_200e_best.pt` |
| `diffcsp-geo-rank3` | 2026-09-14 | `561bb8f` | [`7jdhrv2f`](https://wandb.ai/symmetry-advantage/diffcsp/runs/7jdhrv2f) ($M=3$ `orb-v3` ranking, `cutoff_t=50`) | `runs/mp20_geo/geo_mp20_500e_best.pt` |
| `diffcsp-geov2-rank3` | 2026-09-14 | `561bb8f` | [`say1phjr`](https://wandb.ai/symmetry-advantage/diffcsp/runs/say1phjr) ($M=3$ `orb-v3` ranking, `cutoff_t=50`) | `runs/mp20_geov2/geov2_mp20_400e_best.pt` |

### Aggregate Results

| regime | match/trial | best-of-3 | mean RMS (Å) | clash rate | median vol err | trainable params |
|---|---|---|---|---|---|---|
| `pyxtal-only (control)` | 22.6% | 40.2% | 0.1824 | — | — | — |
| `orb-relax` | 55.0% | 73.0% | 0.0585 | — | — | 0 (25.6M frozen) |
| `wyckoff-painn (512x6, 210e)` *(archived)* | 62.0% | 74.1% | 0.0978 | 25.7% | 8.59% | 18.25M |
| `orb-diffcsp` | 77.4% | 84.5% | 0.0415 | — | — | 561k (+25.6M frozen) |
| `diffcsp-geo-orb (handoff t=0.10)` | 78.9% | 86.7% | 0.0367 | 1.6% | **1.06%** | 12.28M (+25.6M frozen) |
| `asymm-inno2 (converged)` *(archived)* | 79.5% | 83.8% | 0.0457 | 6.8% | 4.18% | 12.28M |
| `vanilla-diffcsp` | 81.3% | 86.6% | 0.0387 | 3.7% | 6.84% | 12.28M |
| `diffcsp-geov2 (400e)` | 82.6% | 87.6% | 0.0407 | 0.7% | 3.26% | 12.28M |
| **`diffcsp-geo (500e)`** | **83.5%** | **88.4%** | **0.0373** *(0.0354 w/ cutoff 50)* | 1.5% | 3.90% | **12.28M** |
| **`diffcsp-geov2-rank3` ($M=3$ ORB rank)** | **85.7%** | **89.1%** | **0.0316** | **0.5%** | 1.47% | **12.28M** |
| **`diffcsp-geo-rank3` ($M=3$ ORB rank)** | **85.9%** | **89.6%** | **0.0329** | **0.5%** | 2.42% | **12.28M** |

`DiffCSP-Geo` strictly outperforms vanilla `DiffCSP++` across the dataset in a single-sample apples-to-apples comparison with identical parameter count (12.28M) and matched gradient steps (paired McNemar per-trial: 136 Geo wins vs 72 Vanilla wins, $p = 1.08 \times 10^{-5}$; best-of-3: 39 Geo wins vs 21 Vanilla wins, $p = 0.0273$). Adding $M=3$ test-time ORB candidate ranking (`diffcsp-geo-rank3`) lifts performance to **85.9% per-trial / 89.6% best-of-3**.

### Split by Wyckoff degrees of freedom (Best-of-3 %)

DoF is the number of free continuous coordinates summed over sites: what the model
actually has to determine. The aggregate above hides three different regimes.

| DoF | n | atoms | control | orb-relax | orb-diffcsp | vanilla | diffcsp-geo (500e) | diffcsp-geov2 (400e) | diffcsp-geo-rank3 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 125 | 4.3 | 89.6% | 100.0% | 100.0% | 100.0% | **100.0%** | **100.0%** | **100.0%** |
| 1 | 125 | 7.3 | 55.2% | 94.4% | 100.0% | 100.0% | **100.0%** | **100.0%** | 99.2% *(100.0% geov2-rank3)* |
| 2 | 191 | 9.3 | 55.0% | 98.4% | 99.0% | 99.5% | **99.0%** | **99.0%** | **99.0%** |
| 3 | 56 | 12.0 | 42.9% | 92.9% | 98.2% | 96.4% | 96.4% | **98.2%** | 96.4% |
| 4-5 | 86 | 13.6 | 41.9% | 80.2% | 96.5% | 94.2% | **97.7%** | 96.5% | 96.5% *(97.7% geov2-rank3)* |
| 6-8 | 153 | 14.4 | 28.8% | 69.9% | 89.5% | 91.5% | **94.1%** | 92.8% | 92.8% *(93.5% geov2-rank3)* |
| 9+ | 264 | 16.1 | 4.5% | 26.9% | 49.6% | 57.2% | **61.7%** | 59.5% | **67.8%** |

### Macro-partition breakdown (`DiffCSP-Geo` vs `Vanilla DiffCSP++`, single-sample)

| Metric | Subgroup | Vanilla DiffCSP++ | DiffCSP-Geo (500e) | Margin / Discordant | Significance |
|---|---|---|---|---|---|
| **Best-of-3** | **DoF $\ge$ 6** ($n=417$) | 69.78% | **73.62%** | **+3.84% (+16 crystals)** | $p = 0.0440$ |
| **Per-trial** | **DoF $\ge$ 6** ($n=1249$) | 59.07% | **63.07%** | **+4.00% (+50 trials)** | $p = 2.39 \times 10^{-4}$ |
| **Best-of-3** | **DoF < 6** ($n=583$) | 98.63% | **98.97%** | **+0.34% (+2 crystals)** | Matched / Beats |
| **Per-trial** | **DoF < 6** ($n=1749$) | 97.26% | **98.06%** | **+0.80% (+14 trials)** | $p = 0.0125$ |
| **Best-of-3** | **Overall** ($n=1000$) | 86.60% | **88.40%** | **+1.80% (+18 crystals)** | $p = 0.0273$ |
| **Per-trial** | **Overall** ($n=2998$) | 81.33% | **83.47%** | **+2.14% (+64 trials)** | $p = 1.08 \times 10^{-5}$ |

- **Below DoF 3 the potential is enough.** At DoF 0 every regime hits 100% and even
  the unrefined draw manages 89.6%: the representation fixes the coordinates and only
  the cell is free. Relaxation stays within 6 points of the generative models to DoF 2.
- **Between DoF 4 and 8 the diffusion earns its keep.** `DiffCSP-Geo` achieves 97.7% at DoF 4-5 and 94.1% at DoF 6-8, beating `DiffCSP++` by 3.5 and 2.6 percentage points respectively.
- **Above DoF 9 capacity and physical Euclidean geometry decide.** `DiffCSP-Geo` achieves 61.7% single-sample and **67.8%** with $M=3$ ORB ranking (vs 57.2% for `DiffCSP++`).
- See full technical details and ablation studies in [`docs/diffcsp-geo-study.md`](diffcsp-geo-study.md).

DoF is the right axis for the full-cell regimes, but not for the asymmetric-unit models (`asymm-inno2`, archived in [`docs/archive/wyckoff-innovations-study.md`](archive/wyckoff-innovations-study.md), and `wyckoff-painn`, archived in [`docs/archive/equivariant-painn-study.md`](archive/equivariant-painn-study.md)): the asymmetric-unit deficit tracks Wyckoff *orbit multiplicity* instead, and vanishes on the structures where every orbit has one member. See [`docs/asymmetric-unit-deficit.md`](asymmetric-unit-deficit.md) and `bench/multiplicity_analysis.py`.

---

## Result 2 — LeMat-Bulk test, does more data help?

**Date:** 2026-09-11 | **Code:** commit `e6fcb3d` (model trained at `3d198d8` / `96f73e4`) | **W&B Run:** [`mqk461a6`](https://wandb.ai/symmetry-advantage/diffcsp/runs/mqk461a6) (`feasible-firefly-70`, 20 epochs on LeMat-Bulk `fmax1` with `E_hull <= 0.1 eV`, `<= 128` atoms; checkpoint `runs/mqk461a6/cspnet_lemat_fmax1_ehull0p1_best.pt`).

1000 representations $\times$ 3 trials from held-out `test.csv.gz`, filtered to the training
distribution (`E_hull <= 0.1 eV`, `<= 128` atoms). Driven by `bench/run_lemat_eval.sh`.

| regime | match/trial | best-of-3 | mean RMS |
|---|---|---|---|
| pyxtal-only (control) | 19.7% | 40.7% | 0.1972 |
| orb-relax | 56.6% | 72.2% | **0.0371** |
| mp20-cspnet (`data/mp-20/test_ckpt.pt`) | 75.0% | 80.6% | 0.0418 |
| **lemat-cspnet (`mqk461a6`, ep 20)** | **77.4%** | **85.6%** | 0.0404 |

**Yes.** Same 12.3M CSPNet, same sampler, same evaluation — only the training set
differs, and the LeMat-trained model gains 5.0 points best-of-3. Paired McNemar:
best-of-3 77 vs 27 discordant ($p = 9.7 \times 10^{-7}$), per-trial 219 vs 145, net +74
($p = 1.2 \times 10^{-4}$).

Note **orb-relax has the best mean RMS despite the worst match rate** among trained
regimes. Relaxation either lands in the right basin and refines it precisely or
misses entirely; it has no mechanism for choosing a basin. The diffusion models match
more often and settle slightly less precisely.

The 85.6% here is not comparable to vanilla's 86.6% on MP-20 — different test set,
different filters, 68 vs 78 space groups. The like-for-like comparison is 85.6 vs 80.6
within this table. For the subsequent 100-epoch and property-conditioned `DiffCSP-GeoV2` runs on packed LeMat-Bulk, see [`docs/lemat-geov2-variants.md`](lemat-geov2-variants.md).

---

## Caveats that apply to both results

- **The sets are the round-trippable subsets.** Reaching 1000 representations screened
  2560 candidates on MP-20 and 2304 on LeMat: only ~40% survive
  `from_seed` -> `from_random` at the same site count. Neither is a uniform sample of
  its test split.
- **The ORB backbone has seen this data.** `orb_v3_direct_20_mpa` is trained on
  MPTraj + Alexandria. MP-20 derives from Materials Project and LeMat-Bulk aggregates
  MP + Alexandria + OQMD, so any orb-* number is not a clean generalization
  measurement. For Result 1 this cuts in the reported direction's favour: the adapter
  ran with an advantage vanilla did not have and still lost.
- **`lemat-cspnet` (`mqk461a6`) was not trained to convergence.** Its validation loss was still
  falling when the 20-epoch budget ended (0.2902), so 85.6% is a floor for that configuration
  rather than its ceiling. The subsequent 100-epoch `DiffCSP-GeoV2` run on the same slice (`rnj30hh2`) trained to full convergence (`val_loss = 0.3088` under the free-DoF masked GeoV2 loss); see [`docs/training-stability.md`](training-stability.md) and [`docs/lemat-geov2-variants.md`](lemat-geov2-variants.md).
- **Training loss is not match rate.** Vanilla's MP-20 validation loss (0.2505) is
  lower than the ORB adapter's (0.2551) and its match rate is also higher, and `DiffCSP-GeoV2`'s MP-20 validation loss (0.3936) is lower than `DiffCSP-Geo`'s (0.4057) while their single-sample match rates are statistically tied (82.6% vs 83.5%, $p = 0.078$) — loss curves alone should not be read as generative match rate, especially across models with different coordinate loss masks.
