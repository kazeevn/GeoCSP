# A frozen ORB potential inside the DiffCSP++ denoiser (archive)

> **Archived 2026-10-23, not maintained.** Written for sharing with collaborators; it brings
> together results previously split between [`mp20-lemat-match-rate-results.md`](mp20-lemat-match-rate-results.md)
> and §4.2 of [`diffcsp-geo-study.md`](diffcsp-geo-study.md). Later findings may supersede it.

**Dates:** experiments 2026-09-09 to 2026-09-14; this write-up 2026-10-23.
**Models:** `orb-diffcsp` (a small adapter on top of ORB) and `diffcsp-geo-orb` (a full-size
denoiser conditioned on ORB, plus a relaxation hand-off at the end of sampling).
**W&B runs** (project `symmetry-advantage/diffcsp`):
- `orb-diffcsp`: [`classic-armadillo-67` (`vuvt0kab`)](https://wandb.ai/symmetry-advantage/diffcsp/runs/vuvt0kab)
- `diffcsp-geo-orb`: [`summer-dragon-74` (`onl3oy96`)](https://wandb.ai/symmetry-advantage/diffcsp/runs/onl3oy96)

**Code:** the adapter is `diffcsp/models/cspnet_orb.py`, `diffusion_orb.py` and `orb_wrapper.py`. Its
final form landed in `2a2b54e` (2026-09-11) and has not changed since. `vuvt0kab` was trained
from a working tree based on `bf77074` that contained those changes before they were committed.
`GeoOrb` is `diffcsp/models/geo_orb_cspnet.py` and `geo_orb_diffusion.py`, committed in `561bb8f`
(2026-09-16); `onl3oy96` ran from a tree based on `54393cd`.
**New in this write-up:** the per-trial split by degrees of freedom (DoF), the paired tests and the
`orb-diffcsp` geometry diagnostics (§5.2–5.4). They were computed on 2026-10-23 at `1c5134c`
from the stored predictions (`runs/bench/mp20/results.pkl`, `results_master.pkl`,
`pred_*.pkl`), with `bench/structure_diagnostics.py` and a paired McNemar (binomial) test.

---

## TL;DR

- We tested two ways of putting a frozen, pretrained interatomic potential (ORB v3) inside the
  denoiser. At every denoising step, ORB is evaluated on the current noisy structure. Its forces,
  stress and learned atomic representations become extra inputs to the network.
- **Neither model beat its no-ORB counterpart on match rate.** The small adapter (561k trainable
  parameters) reaches 77.4% per trial and 84.5% best-of-3, against 81.3% / 86.6% for vanilla
  DiffCSP++. The full-size `GeoOrb` reaches 78.9% / 86.7%, against 83.5% / 88.4% for DiffCSP-Geo
  and 82.6% / 87.6% for GeoV2. The per-trial deficits are highly significant; the best-of-3
  deficits are mostly not.
- **ORB conditioning makes the cell much more accurate.** Median cell-volume error is 1.3%
  (adapter) and 1.1% (`GeoOrb`), against 3.3–6.8% for every model without ORB. Clash rates are
  low as well. The adapter gets these numbers with no relaxation at all, so the gain comes from
  the conditioning itself, not from the relaxation hand-off.
- The gains are all on low-DoF structures, where match rates are already near 100%. On DoF ≥ 9,
  where most failures are, ORB helps neither the match rate nor the cell volume.
- Caveat: ORB v3 was trained on data that overlaps MP-20, so every ORB number has a leakage
  advantage (§6).

---

## 1. Setting

The task is crystal structure prediction from a given **Wyckoff representation**: space group,
occupied Wyckoff orbits and their species. These fix the symmetry scaffold. The model must find
the continuous parameters: the free fractional coordinates of each orbit and the lattice.
DiffCSP++ does this by diffusing fractional coordinates on the 3-torus and the lattice in a
6-dimensional space-group-projected representation. A denoiser network predicts the
coordinate score and the lattice noise at each step.

The motivating idea: a universal interatomic potential already encodes much of the energy
landscape a denoiser would otherwise have to learn. So the denoiser could be told, at every
step, which way the atoms are being pushed, how strained the cell is, and how the potential
sees each atom's chemical environment.

## 2. Shared component: the frozen ORB backbone

[`OrbBackboneWrapper`](../../diffcsp/models/orb_wrapper.py) wraps `orb_v3_direct_20_mpa`, about
25.6M parameters, all frozen and evaluated under `no_grad`. The direct heads predict forces and
stress in a single pass. At each call it takes the current noisy structure $(A, x_t, L_t)$ and
returns:

| output | shape | processing |
|---|---|---|
| forces | $N \times 3$ | clamped to $\lVert f \rVert \le 2.0$; the denoiser receives both Cartesian and fractional components (6-d) |
| stress | $B \times 3 \times 3$ | clamped to $\le 0.05$ |
| atomic representations | $N \times 256$ | the GNN's node latent, captured by a forward hook (so the backbone runs once per call) and layer-normalised |
| graph representation | $B \times 256$ | mean of the atomic representations |

Two practical details matter at high noise:

- Cells collapsed below 5 Å³/atom never reach ORB. Their neighbour lists explode, and the potential
  would be far out of distribution anyway. These cells get a bounded analytic repulsion instead,
  with zero representations.
- orb-v3 includes ZBL pair repulsion internally.

The whole batch is featurised in one neighbour-list build and one `predict` call.

ORB runs at **every** denoising step, in training and in sampling. It costs much more than a
plain CSPNet forward pass.

## 3. The two models

### 3.1 `orb-diffcsp` — lightweight adapter ([`CSPNetORB`](../../diffcsp/models/cspnet_orb.py))

This is the original design: ORB does the heavy lifting and a small network adapts its outputs into
a score.

- **Node input.** An element embedding, the time embedding, the projected ORB atomic representation
  and the projected 6-d force are concatenated and fused by a 2-layer MLP (hidden 128).
- **Message passing.** 2 standard `CSPLayer`s (hidden 128), operating on fractional-coordinate
  differences as in DiffCSP++.
- **Coordinate head.** An unconstrained linear head.
- **Lattice head.** An MLP over pooled node features, the projected ORB graph representation, the
  projected stress, the current lattice and the time embedding.
- **Trainable parameters:** 561,408, compared with 12.28M for vanilla CSPNet.
- **Sampler:** the standard DiffCSP++ predictor–corrector sampler. No relaxation is involved.

*History, for anyone reading older commits.* The first version (`7a00540`) hard-wired the score as
$s = \gamma f_\text{frac} + v_\perp$ with $\gamma > 0$, so that a zero score implied zero force.
This was mis-signed against the sampler's update: the resulting step pointed uphill in energy
(cosine with the ORB force −0.9994). Training could only respond by driving $\gamma$ to its floor.
In the same version the "representations" were zero-padded forces, so a tenth of the adapter's
weights never received any input. Both problems were fixed in `2a2b54e`; the run reported here
uses the fixed version. The old constraint is still available behind `--enforce_zero_force`.

### 3.2 `diffcsp-geo-orb` — full-size, geometry-aware denoiser ([`GeoOrbCSPNet`](../../diffcsp/models/geo_orb_cspnet.py))

After the adapter, the question was whether ORB helps a denoiser that has full capacity. `GeoOrb`
is the DiffCSP-GeoV2 architecture (see [`diffcsp-geo-study.md`](diffcsp-geo-study.md)) with ORB
inputs added:

- **Node input.** The GeoV2 crystallographic embeddings (element 384-d, space group 64-d, orbit
  multiplicity 32-d, site DoF 32-d), the ORB atomic representation projected to 512, the 6-d force
  projected to 512, and the time embedding. These are fused to hidden 512.
- **Message passing.** 6 `GeoV2CSPLayer`s: Euclidean bond lengths with a Bessel radial basis
  ($r_\text{cut} = 8$ Å), unit bond vectors and distance-gated attention.
- **Coordinate head.** A node MLP plus a directional readout along bonds, projected onto each
  site's Wyckoff tangent space. Special positions (0-DoF sites) are locked exactly and excluded
  from the loss.
- **Lattice head.** Mean- and max-pooled nodes, the current lattice, time, the ORB graph
  representation and the raw 9-d stress. The output is projected onto the space group's lattice
  family.
- **Trainable parameters:** 12,259,444, within the 12.28M budget shared by every full-size model here.
- **Two-phase sampler** (`--orb_handoff_t 0.10`). Diffusion runs from $t = 1$ down to $t = 0.10$.
  The structure is then handed to at most 50 steps of ORB FIRE relaxation (`fmax` 0.05), with
  `FixSymmetry` and `FrechetCellFilter`. Afterwards special positions are re-locked and the lattice
  is re-projected onto the space group.

| | `orb-diffcsp` | `diffcsp-geo-orb` |
|---|---|---|
| trunk | 2 × `CSPLayer`, hidden 128 | 6 × `GeoV2CSPLayer`, hidden 512 |
| trainable / frozen params | 0.56M / 25.6M | 12.26M / 25.6M |
| ORB inputs to the coordinate path | atomic repr., forces | atomic repr., forces |
| ORB inputs to the lattice head | graph repr., stress | graph repr., stress |
| Wyckoff tangent-space projection | no (orbit averaging as in DiffCSP++) | yes |
| relaxation in the sampler | none | ORB FIRE from $t = 0.10$ |

## 4. Training

Both models were trained on MP-20 (27k train structures) on one RTX 6000 Ada (host `zeus`).

| | `orb-diffcsp` (`vuvt0kab`) | `diffcsp-geo-orb` (`onl3oy96`) |
|---|---|---|
| started | 2026-09-09 | 2026-09-13 |
| epochs × batch | 1000 × 128 | 200 × 64 |
| optimiser | Adam, lr 1e-3, `ReduceLROnPlateau` (down to 1e-4 by epoch ~400) | AdamW (wd 1e-4), lr 5e-4, 10-epoch warm-up, cosine to 1e-6, EMA 0.9999 |
| wall clock | 7.2 h | 2.1 h |
| best val loss | 0.2504 (epoch 989) | 0.3894 (epoch 180, EMA) |
| **checkpoint benchmarked** | **epoch 500, val 0.2551** | **epoch 180, EMA** |

The adapter was benchmarked at epoch 500 rather than at its best epoch. A bug in the
best-checkpoint logic, fixed in `96f73e4`, only considered epochs that were multiples of 50.
The difference in validation loss is small (0.2551 vs 0.2504).

Validation losses are not comparable between the two columns. `GeoOrb` uses the GeoV2 loss,
which is restricted to free DoF and projected onto tangent spaces. For reference, vanilla
CSPNet's best MP-20 validation loss is 0.2505. That is slightly below the adapter's, and vanilla
also matches more often.

## 5. Results — MP-20 test benchmark

**Protocol** ([`../benchmark.md`](../benchmark.md)):

- 1000 test structures are reduced to their Wyckoff representations. Each is drawn 3 times with
  `pyxtal.from_random`, giving 2,998 draws (`runs/bench/mp20/inits.pkl`).
- Every regime runs on the same draws. For the diffusion regimes, the three trials are three
  sampler seeds.
- Scoring uses `StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)`.
- **match/trial** is the fraction of draws that match the ground truth. **best-of-3** is the
  fraction of structures with at least one match. RMS is averaged over matched draws only.

**Reference regimes:**

- `vanilla-diffcsp`: DiffCSP++ CSPNet, 12.28M parameters, W&B `zowdmqwo`.
- `diffcsp-geo`: 500 epochs, best epoch 320, W&B `7jdhrv2f`.
- `diffcsp-geov2`: 400 epochs, W&B `say1phjr`. It shares `GeoOrb`'s layer and training recipe but
  has no ORB.
- `orb-relax`: no learned model. ORB relaxation of the random draw with `FixSymmetry` and
  `FrechetCellFilter`, FIRE, up to 200 steps.

### 5.1 Overall

| regime | trainable params | match/trial | best-of-3 | mean RMS (Å) | clash rate | median vol. error |
|---|---|---|---|---|---|---|
| `pyxtal-only` (control) | — | 22.6% | 40.2% | 0.1824 | — | — |
| `orb-relax` | 0 | 55.0% | 73.0% | 0.0585 | 11.3% | 7.45% |
| **`orb-diffcsp`** | 0.56M | **77.4%** | **84.5%** | 0.0415 | **1.0%** | **1.26%** |
| **`diffcsp-geo-orb`** (hand-off $t = 0.10$) | 12.26M | **78.9%** | **86.7%** | **0.0367** | **1.6%** | **1.06%** |
| `vanilla-diffcsp` | 12.28M | 81.3% | 86.6% | 0.0387 | 3.7% | 6.84% |
| `diffcsp-geov2` | 12.28M | 82.6% | 87.6% | 0.0407 | 0.7% | 3.26% |
| `diffcsp-geo` | 12.28M | 83.5% | 88.4% | 0.0373 | 1.5% | 3.90% |

A draw counts as a clash when its shortest interatomic distance is below 0.9 × the ground truth's.
Clash rate and volume error are computed on all draws, matched or not.

### 5.2 Paired comparisons (McNemar, same draws)

Each cell gives the number of draws (or structures) only the first model matched : only the
second model matched, and the two-sided *p*-value.

| comparison | per trial, all | per trial, DoF ≥ 6 | best-of-3, all | best-of-3, DoF ≥ 6 |
|---|---|---|---|---|
| `orb-diffcsp` vs `vanilla` | 100 : 219, $p = 2 \times 10^{-11}$ | 79 : 176, $p = 1 \times 10^{-9}$ | 31 : 52, $p = 0.028$ | 27 : 50, $p = 0.012$ |
| `orb-diffcsp` vs `orb-relax` | 755 : 85, $p < 10^{-100}$ | 424 : 59, $p = 4 \times 10^{-69}$ | 135 : 20, $p = 4 \times 10^{-22}$ | 106 : 16, $p = 2 \times 10^{-17}$ |
| `geo-orb` vs `geo` | 101 : 237, $p = 1 \times 10^{-13}$ | 83 : 195, $p = 1 \times 10^{-11}$ | 36 : 53, $p = 0.089$ | 32 : 49, $p = 0.075$ |
| `geo-orb` vs `geov2` | 106 : 216, $p = 9 \times 10^{-10}$ | 83 : 174, $p = 1 \times 10^{-8}$ | 36 : 45, $p = 0.37$ | 32 : 41, $p = 0.35$ |
| `geo-orb` vs `vanilla` | 124 : 196, $p = 7 \times 10^{-5}$ | 96 : 158, $p = 1 \times 10^{-4}$ | 42 : 41, $p = 1.0$ | 36 : 37, $p = 1.0$ |
| `geo-orb` vs `orb-diffcsp` | 201 : 154, $p = 0.015$ | 153 : 118, $p = 0.039$ | 56 : 34, $p = 0.026$ | 52 : 30, $p = 0.020$ |

### 5.3 Split by Wyckoff degrees of freedom

DoF is the number of free continuous coordinates summed over orbits, i.e. what the model actually
has to determine. Each cell shows match/trial / best-of-3, in %.

| DoF | n | atoms | `orb-relax` | `orb-diffcsp` | `vanilla` | `geo` | `geo-orb` |
|---|---|---|---|---|---|---|---|
| 0 | 125 | 4.3 | 96.3 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 100.0 / 100.0 | 99.2 / 100.0 |
| 1 | 125 | 7.3 | 74.5 / 94.4 | 97.6 / 100.0 | 99.5 / 100.0 | 99.2 / 100.0 | 97.1 / 98.4 |
| 2 | 191 | 9.3 | 83.1 / 98.4 | 96.7 / 99.0 | 98.4 / 99.5 | 98.4 / 99.0 | **99.1 / 100.0** |
| 3 | 56 | 12.0 | 70.8 / 92.9 | 92.9 / 98.2 | 94.6 / 96.4 | 95.8 / 96.4 | **97.0 / 98.2** |
| 4–5 | 86 | 13.6 | 53.5 / 80.2 | 88.4 / 96.5 | 89.1 / 94.2 | **94.2 / 97.7** | 86.8 / 96.5 |
| 6–8 | 153 | 14.4 | 38.8 / 69.9 | 76.5 / 89.5 | 84.3 / 91.5 | **88.7 / 94.1** | 78.9 / 90.8 |
| 9+ | 264 | 16.1 | 12.5 / 26.9 | 36.7 / 49.6 | 44.4 / 57.2 | **48.2 / 61.7** | 39.8 / 57.2 |

### 5.4 Geometry by DoF

Median relative cell-volume error $|V_\text{pred} - V_\text{gt}| / V_\text{gt}$, then clash rate:

| subset | n | `orb-relax` | `orb-diffcsp` | `vanilla` | `geo` | `geo-orb` |
|---|---|---|---|---|---|---|
| DoF ≤ 5 | 583 | 1.47% · 2.9% | 0.84% · 0.9% | 7.08% · 1.4% | 3.51% · 0.2% | **0.57%** · 0.7% |
| DoF 6–8 | 153 | 8.34% · 7.8% | **1.26%** · 1.3% | 6.17% · 2.6% | 3.62% · 0.0% | 1.98% · 5.9% |
| DoF ≥ 9 | 264 | 19.89% · 31.8% | **5.31%** · 1.1% | 6.49% · 9.5% | 6.49% · 5.3% | 6.82% · 1.1% |

## 6. Interpretation

**ORB is a good lattice oracle.** Volume error falls from 3.5–7% to below 1% on DoF ≤ 5 for both
ORB models. The adapter, which never relaxes, already reaches 1.26% overall. So the stress and
representation inputs on their own give the lattice head the equilibrium volume. The hand-off in
`GeoOrb` contributes little here. This fits ORB's training data: it was fit to the same DFT
(PBE) relaxations that define MP-20's ground truth.

**ORB does not help the coordinate search, and the search is where match rates are lost.** On
DoF ≥ 9 neither ORB model improves the volume error or the match rate. The adapter has only 4.5%
of vanilla's trainable parameters and loses 5–8 points per trial on DoF ≥ 6. At equal capacity,
`GeoOrb` still trails its no-ORB sibling GeoV2 by 3.7 points per trial ($p \approx 10^{-9}$).
Best-of-3 is statistically tied, so `GeoOrb`'s samples are less consistent rather than unable to
reach the basin.

**The relaxation hand-off is the most likely cause of `GeoOrb`'s per-trial deficit, but this was
not isolated.** Handing off at $t = 0.10$ means relaxation starts before diffusion has settled
the multi-orbit packing. On high-DoF structures it then descends into whichever local ORB basin
is nearest. The DoF 6–8 clash rate (5.9% vs 0.0% for Geo) fits this picture: a 50-step
relaxation that leaves atoms close together. However, `GeoOrb` was never benchmarked without
the hand-off. It was also trained for 200 epochs, against 400–500 for the Geo models. Its
deficit therefore cannot be split between the ORB conditioning, the hand-off and the training
budget.

**The learned diffusion beats relaxation alone by a wide margin.** Even the 561k-parameter
adapter beats `orb-relax` by 22 points per trial. The potential refines a basin precisely once
it is in one, but it has no way of choosing the basin. On MP-20 the cheaper way to use ORB turned
out to be test-time ranking: draw 3 candidates per trial from a no-ORB denoiser and keep the one
with the lowest ORB energy. That gives 85.9% / 89.6% for Geo + rank-3; see §4.3 of
[`diffcsp-geo-study.md`](diffcsp-geo-study.md).

**Caveats**

- **Data leakage.** `orb_v3_direct_20_mpa` is trained on MPtraj + Alexandria, and MP-20 derives
  from the Materials Project. The ORB models therefore had an advantage the baselines lacked, and
  they still did not win on match rate.
- **Single runs.** Each configuration was trained once, so the differences reflect one seed per model.
- **Not benchmarked:** `GeoOrb` without the hand-off, or with a later hand-off; the adapter at its
  best epoch (989); ORB conditioning on LeMat-Bulk or Alex-MP-20.

## 7. Reproduction

| step | command / file |
|---|---|
| train adapter | `uv run diffcsp-train --model orb --orb_model orb-v3 --epochs 1000 --batch_size 128 --lr 1e-3 --wandb` (README, "Train Frozen ORB MLIP Adapter") |
| train `GeoOrb` + benchmark | [`scripts/train_geo_orb.sh`](../../scripts/train_geo_orb.sh) |
| adapter / relax / vanilla benchmark | [`bench/run_all.sh`](../../bench/run_all.sh) (`--regime orb`, `run_relax.py`) |
| checkpoints | `runs/vuvt0kab/model-vuvt0kab:v19/orb_v3_diffcsp_mp20_opt3.pt` (adapter, epoch 500), `runs/mp20_geo_orb/geo_orb_mp20_200e_best.pt` (`GeoOrb`, epoch 180) |
| predictions | `runs/bench/mp20/pred_orb.pkl`, `pred_geo_orb_hybrid.pkl`, `pred_relax.pkl` |
| scores | `runs/bench/mp20/results.pkl` (adapter, relax, vanilla), `results_master.pkl` (Geo family) |
| geometry | `uv run python bench/structure_diagnostics.py --preds "orb-diffcsp=runs/bench/mp20/pred_orb.pkl" ...` |
| tests | `tests/test_geo_orb.py`; adapter tests via `uv run pytest -k orb` |
