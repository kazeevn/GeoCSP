# DiffCSP-Geo & GeoV2: Cartesian-Aware, Symmetry-Projected Crystal Structure Prediction

> **Archived 2026-10-23, not maintained.** Moved from `docs/diffcsp-geo-study.md` (file last
> changed 2026-10-16 in `0b7640d`) because it records study results; only relative links were
> updated. Later findings may supersede it.

**Project:** DiffCSP++ Research & Engineering  
**Dates:** 2026-09-13 to 2026-09-29  
**Code:** commits `d7b1a7b` / `4d39843` (`DiffCSP-Geo`), `561bb8f` (`DiffCSP-GeoV2`, `DiffCSP-GeoOrb`, test-time ORB ranking), `9a4aa98` / `f9e3b3a` (Alex-MP-20 `GeoV2`).  
**Evaluation Standard:** Materials Project 20 (MP-20) Test Subset (1,000 crystal representations $\times$ 3 trials = 2,998 draws) evaluated via `pymatgen.analysis.structure_matcher.StructureMatcher(stol=0.5, angle_tol=10.0, ltol=0.3)`.  
**WanDB Runs (project `symmetry-advantage/diffcsp`):**
- `DiffCSP-Geo` (MP-20, 500e): [`expert-wood-72` (`7jdhrv2f`)](https://wandb.ai/symmetry-advantage/diffcsp/runs/7jdhrv2f)
- `DiffCSP-GeoV2` (MP-20, 400e): [`toasty-wood-73` (`say1phjr`)](https://wandb.ai/symmetry-advantage/diffcsp/runs/say1phjr)
- `DiffCSP-GeoOrb` (MP-20, 200e): [`summer-dragon-74` (`onl3oy96`)](https://wandb.ai/symmetry-advantage/diffcsp/runs/onl3oy96)
- `DiffCSP-GeoV2` (Alex-MP-20, 150e): [`different-morning-75` (`ua1g6od4`)](https://wandb.ai/symmetry-advantage/diffcsp/runs/ua1g6od4)

---

## 1. Executive Summary

In previous investigations ([`docs/asymmetric-unit-deficit.md`](../asymmetric-unit-deficit.md), [`docs/archive/equivariant-painn-study.md`](equivariant-painn-study.md)), attempts to improve **DiffCSP++** focused on collapsing the $N$-atom conventional cell to the $K$-site asymmetric unit ($K \ll N$). While this reduced graph edges, it caused a severe match-rate deficit on high-DoF structures ($\text{DoF} \ge 6$) because collapsing orbit replicas discarded the distinct hidden states that the non-equivariant denoiser relies on, while finite-cutoff models (like PaiNN with 6 Å cutoff) starved on expanded initial cells.

**DiffCSP-Geo** resolves this challenge by retaining the full-cell graph ($N$ atoms) while addressing the four fundamental architectural flaws of vanilla DiffCSP++:
1. **Cartesian Metric Awareness (Curing Chemical Blindness)**: Extends fractional difference sinusoids $(x_j - x_i) \pmod 1$ with true Euclidean bond lengths $r_{ij} = \|\mathbf{L} \Delta \mathbf{x}_{ij, \text{mic}}\|_2$, 32-frequency Bessel radial basis functions ($r_{\text{cut}} = 8.0$ Å), and 3D unit direction vectors $\hat{\mathbf{r}}_{ij}$.
2. **Lie-Algebra Wyckoff Tangent-Space Projection**: Strictly locks 0-DoF special positions to their Wyckoff coordinates, confines noise and score updates to the stabilizer tangent space $\mathcal{T}_k = \mathbf{P}_k \mathbb{R}^3$, and masks 0-DoF positions from coordinate loss.
3. **Rich Crystallographic Conditioning**: Embeds space group $G \in \{1,\dots,230\}$, orbit multiplicity $m$, and site DoF alongside atomic number $Z$.
4. **Direct Lattice Conditioning**: Feeds the current noisy lattice representation and time embedding directly into the lattice prediction head.

---

## 2. Apples-to-Apples Comparison Results (MP-20 Benchmark)

Evaluated under strict apples-to-apples matching:
- **Parameters**: 12,283,784 for DiffCSP-Geo vs 12,277,248 for DiffCSP++ (+0.053% delta).
- **Training Budget**: 500 epochs (106,000 gradient steps) on MP-20 on GPU 0.

### 2.1 Match Rates by Degrees of Freedom (DoF)

| Category / Bracket | Crystals ($n$) | DiffCSP++ Best-of-3 | DiffCSP-Geo Best-of-3 | DiffCSP++ Per-Trial | DiffCSP-Geo Per-Trial | Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall** | **1,000** | **86.60%** (866/1000) | **88.40%** (884/1000) | **81.33%** | **83.47%** | **DiffCSP-Geo Beats (+1.80%)** |
| **DoF < 6** | **583** | **98.63%** (575/583) | **98.97%** (577/583) | **97.26%** | **98.06%** | **DiffCSP-Geo Beats (+0.34%)** |
| **DoF $\ge$ 6** | **417** | **69.78%** (291/417) | **73.62%** (307/417) | **59.07%** | **63.07%** | **DiffCSP-Geo Beats (+3.84%)** |
| ↳ *DoF 0* | 125 | 100.00% | 100.00% | 100.00% | 100.00% | Exact Parity |
| ↳ *DoF 1* | 125 | 100.00% | 100.00% | 99.47% | 99.20% | Parity |
| ↳ *DoF 2* | 191 | 99.48% | 98.95% | 98.43% | 98.43% | Parity |
| ↳ *DoF 3* | 56 | 96.43% | 96.43% | 94.64% | 95.83% | DiffCSP-Geo Beats (+1.19%) |
| ↳ *DoF 4–5* | 86 | 94.19% | **97.67%** | 89.15% | **94.19%** | **DiffCSP-Geo Beats (+3.48%)** |
| ↳ *DoF 6–8* | 153 | 91.50% | **94.12%** | 84.31% | **88.67%** | **DiffCSP-Geo Beats (+2.62%)** |
| ↳ *DoF 9+* | 264 | 57.20% | **61.74%** | 44.44% | **48.23%** | **DiffCSP-Geo Beats (+4.54%)** |

---

### 2.2 Paired McNemar Statistical Significance Tests

| Comparison Subset | Trials | DiffCSP++ Wins | DiffCSP-Geo Wins | Net Margin | Two-Sided $p$-value | Significance |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **All MP-20 Trials** | 2,998 | 72 | 136 | +64 (+2.13%) | $\mathbf{1.08 \times 10^{-5}}$ | **Significant ($p < 0.0001$)** |
| **DoF $\ge$ 6** | 1,251 | 65 | 115 | +50 (+4.00%) | $\mathbf{2.39 \times 10^{-4}}$ | **Significant ($p < 0.001$)** |
| **DoF < 6** | 1,747 | 7 | 21 | +14 (+0.80%) | $\mathbf{0.0125}$ | **Significant ($p < 0.05$)** |

---

### 2.3 Physical Quality & Geometry Diagnostics

#### A. Atomic Clash Rate ($r_{\text{pred}} < 0.9 \times r_{\text{GT}}$)
| Subset | Crystals ($n$) | Ground Truth Contact | DiffCSP++ Clash Rate | DiffCSP-Geo Clash Rate | Relative Reduction |
|:---|:---:|:---:|:---:|:---:|:---:|
| **All** | 1,000 | 2.39 Å | 3.7% | **1.5%** | **-59.5%** |
| **DoF $\le$ 5** | 583 | 2.55 Å | 1.4% | **0.2%** | **-85.7%** |
| **DoF 6–8** | 153 | 2.30 Å | 2.6% | **0.0%** | **-100.0% (Zero clashes)** |
| **DoF $\ge$ 9** | 264 | 2.09 Å | 9.5% | **5.3%** | **-44.2%** |

#### B. Median Relative Cell-Volume Error ($|V_{\text{pred}} - V_{\text{gt}}| / V_{\text{gt}}$)
| Subset | Crystals ($n$) | DiffCSP++ Volume Error | DiffCSP-Geo Volume Error | Accuracy Gain |
|:---|:---:|:---:|:---:|:---:|
| **All** | 1,000 | 0.0684 | **0.0390** | **+43.0% more accurate** |
| **DoF $\le$ 5** | 583 | 0.0708 | **0.0351** | **+50.4% more accurate** |
| **DoF 6–8** | 153 | 0.0617 | **0.0362** | **+41.3% more accurate** |

---

## 3. Training Dynamics, Parameter Capacity & Overfitting Analysis

### 3.1 Empirical Evidence: The Late-Stage Generalization Gap

In W&B Run [`expert-wood-72`](https://wandb.ai/symmetry-advantage/diffcsp/runs/7jdhrv2f), tracking validation and training losses across 500 epochs reveals clear evidence of **overfitting beginning around Epoch 320**:

| Epoch | Train Loss | Val Loss | Learning Rate | Generalization Gap ($L_{\text{val}} - L_{\text{train}}$) | Dynamics / Status |
|:---:|:---:|:---:|:---:|:---:|:---|
| **200** | 0.4030 | 0.4306 | $5.0 \times 10^{-4}$ | +0.0276 | Healthy steady convergence |
| **270** | 0.3879 | 0.4121 | $5.0 \times 10^{-4}$ | +0.0242 | Smooth optimization |
| **300** | 0.3695 | 0.4141 | $3.0 \times 10^{-4}$ | +0.0446 | Scheduler triggers LR decay |
| **320** | **0.3619** | **0.4057** | **$3.0 \times 10^{-4}$** | **+0.0438** | **Peak Validation Performance** (`geo_mp20_500e_best.pt`) |
| **360** | 0.3546 | 0.4141 | $3.0 \times 10^{-4}$ | +0.0595 | Val loss turns upward |
| **400** | 0.3553 | 0.4273 | $3.0 \times 10^{-4}$ | +0.0720 | Val loss drifts upward |
| **440** | 0.3299 | 0.4317 | $1.08 \times 10^{-4}$ | +0.1018 | Gap widens past 0.10 |
| **460** | 0.3272 | 0.4405 | $1.08 \times 10^{-4}$ | +0.1133 | Val loss reaches peak spike |
| **500** | **0.3189** | **0.4373** | **$1.0 \times 10^{-4}$** | **+0.1184** | Final checkpoint (`geo_mp20_500e.pt`) |

Between Epoch 320 and Epoch 500:
- **Train loss** continued to fall from $0.3619 \to 0.3189$ ($-11.9\%$ relative reduction).
- **Val loss** deteriorated from $0.4057 \to 0.4373$ ($+7.8\%$ relative increase).
- **The generalization gap** widened from $+0.0438 \to +0.1184$ (nearly tripled).

The benchmark pipeline evaluated [`runs/mp20_geo/geo_mp20_500e_best.pt`](file:///home/kna/DiffCSPNew/runs/mp20_geo/geo_mp20_500e_best.pt), which preserved the model at its peak generalization state at Epoch 320, shielding the final benchmark metrics from the late-stage overfitting.

### 3.2 Inductive Bias vs. Parameter Capacity

Why did DiffCSP-Geo saturate and overfit on MP-20 when trained past 320 epochs, while vanilla DiffCSP++ was trained for 500–600 epochs without early saturation?

1. **Vanilla DiffCSP++ Was Capacity-Starved Due to Chemical Blindness:**
   Vanilla CSPNet has zero awareness of Euclidean Ångström bond lengths: it operates strictly on fractional difference sinusoids $(x_j - x_i) \pmod 1$ and a pooled 6D lattice vector. To predict chemically plausible structures and avoid atomic collisions, it had to devote vast amounts of its 12.28M parameter capacity to implicitly reconstructing Euclidean metric geometry and Lennard-Jones/Pauli repulsion forces. This made it learn extremely slowly, requiring 500–600 epochs just to fit rough chemistry.

2. **DiffCSP-Geo Has Strong Physical Inductive Bias:**
   DiffCSP-Geo explicitly computes:
   - Minimum-image Euclidean bond vectors $\mathbf{r}_{ij} = \Delta \mathbf{x}_{ij, \text{mic}}\mathbf{L}$.
   - 32-frequency Bessel radial basis functions ($r_{\text{cut}} = 8.0$ Å).
   - 3D unit bond direction vectors $\hat{\mathbf{r}}_{ij}$.
   - Lie-algebra stabilizer tangent space projections $\mathbf{P}_j = R_j \mathbf{P}_a R_j^{-1}$.
   Because metric physics and crystallographic symmetries are built directly into the computational graph, the network does not waste capacity learning coordinate-to-metric mappings.

3. **12.28M Parameters is Oversized for MP-20 (~27k Crystals):**
   The MP-20 training set contains ~27,000 structures. A 12.28M parameter model possesses $\approx 455$ parameters per crystal structure. With strong inductive bias, DiffCSP-Geo fully captured the underlying distribution in **~300 epochs (~65,000 gradient steps)**. Continuing training beyond this point allowed the unconstrained 12.28M parameters to fit idiosyncratic sample noise.

### 3.3 Flaws in the Original `geo` Training Schedule

Two implementation choices in [`diffcsp/cli/train.py`](file:///home/kna/DiffCSPNew/diffcsp/cli/train.py) exacerbated this overfitting for `--model geo`:
1. **Scheduler Stepped on Training Loss (`scheduler.step(avg_loss)`):**
   `scheduler.step(avg_loss)` passed the training loss to `ReduceLROnPlateau`. When training loss descent slowed around epoch 300, the scheduler reduced the learning rate ($5\times 10^{-4} \to 3\times 10^{-4} \to 1.8\times 10^{-4} \to 1\times 10^{-4}$). Decaying the learning rate when validation loss is already rising traps the model in narrow local minima on the training set, accelerating memorization.
2. **Zero Weight Decay and Absence of EMA:**
   - Optimization for `--model geo` used bare `torch.optim.Adam` without weight decay ($L_2 = 0$).
   - Without Exponential Moving Average (EMA) smoothing, late-stage weights tracked high-variance stochastic batch noise.

*(Note: Stepping `ReduceLROnPlateau` on validation loss for legacy models and downscaling `GeoCSPNet` to 4M–6M parameters (`384x4`) on MP-20 remain untried ideas — see [`docs/ideas/training-and-capacity-refinements.md`](../ideas/training-and-capacity-refinements.md).)*

---

## 4. Follow-Up Variants: DiffCSP-GeoV2, DiffCSP-GeoOrb, and Test-Time ORB Ranking

**Date:** 2026-09-13 to 2026-09-14 (committed in `561bb8f` on 2026-09-16; Alex-MP-20 run in `9a4aa98` / `f9e3b3a` on 2026-09-29).

### 4.1 DiffCSP-GeoV2 (`--model geov2`, W&B `say1phjr`)
[`GeoV2CSPNet`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_v2_cspnet.py) and [`GeoV2Diffusion`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_v2_diffusion.py) extend `DiffCSP-Geo` within the same $\le 12.28\text{M}$ parameter budget (`edge_dim=482`) by incorporating:
- **Training stabilization:** `AdamW` ($\text{weight\_decay}=10^{-4}$), linear warmup + cosine annealing LR schedule, and `EMAModel` ($\text{decay}=0.9999$).
- **Distance-gated attention aggregation:** Softmax edge attention multiplied by a learned sigmoid distance gate so close coordinating neighbors are not diluted by distant cell atoms in crowded cells.
- **Directional bond force readout:** Edge scalar predictions projected along unit bond vectors $\hat{\mathbf{r}}_{ij} \mathbf{L}^{-1}$, summed with a 2-layer node coordinate MLP and projected by $\mathbf{P}_j$.
- **Annealed Langevin corrector:** Smoothly decays corrector step size as $t \to 0$.

Trained for 400 epochs on MP-20 (W&B [`toasty-wood-73` / `say1phjr`](https://wandb.ai/symmetry-advantage/diffcsp/runs/say1phjr)), `DiffCSP-GeoV2` reached a lower validation loss than `DiffCSP-Geo` (**0.3936** at epoch 300 vs **0.4057** at epoch 320), halved the atomic clash rate (**0.7%** vs **1.5%** overall, **0.0%** for $\text{DoF} \le 8$), and reduced median cell-volume error to **3.26%** (vs **3.90%** for `DiffCSP-Geo` and **6.84%** for vanilla). Single-sample MP-20 match rate is **82.59% per-trial / 87.60% best-of-3** (paired McNemar vs `DiffCSP-Geo`: $p = 0.078$, statistically indistinguishable overall).

### 4.2 DiffCSP-GeoOrb & Two-Phase Hybrid Sampling (`--model geo_orb`, W&B `onl3oy96`)
[`GeoOrbCSPNet`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_orb_cspnet.py) and [`GeoOrbDiffusion`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_orb_diffusion.py) condition a 6-layer `GeoV2CSPLayer` denoiser ($\le 12.28\text{M}$ trainable parameters) on frozen `orb-v3` atomic representations (256-d), Cartesian/fractional forces (6-d), and virial stresses (9-d), and implement **Innovation 6 (Two-Phase Hybrid Sampling)** via `--orb_handoff_t 0.10` (handing off the diffusion trajectory at $t \le 0.10$ to 50 steps of symmetry-constrained `FixSymmetry` + `FrechetCellFilter` ORB FIRE relaxation).

Trained for 200 epochs on MP-20 (W&B [`summer-dragon-74` / `onl3oy96`](https://wandb.ai/symmetry-advantage/diffcsp/runs/onl3oy96), best val loss **0.3894** at epoch 180), hybrid sampling achieved the lowest median cell-volume error (**1.06%** overall, **0.57%** on $\text{DoF} \le 5$) and **0.0367 Å** matched RMS, but lower per-trial match rate (**78.92% per-trial / 86.70% best-of-3**, trailing `DiffCSP-Geo` by 4.5 points per-trial, $p = 9.7 \times 10^{-14}$), as early handoff at $t = 0.10$ traps high-DoF structures in local MLIP basins before diffusion finishes resolving multi-orbit packing.

### 4.3 Test-Time ORB Candidate Ranking (`--rank_candidates 3`)
Instead of modifying the denoiser or handing off early to local relaxation, `bench/run_diffusion.py` supports drawing $M=3$ parallel trajectories per trial (with `--noise_cutoff_t 50 --step_lr 8e-6`) and selecting the candidate with the lowest single-point `orb-v3` energy per atom (`scripts/eval_geo_rank3.sh`, `scripts/eval_geov2_rank3.sh`, evaluated 2026-09-14):

| Regime | Per-Trial Match | Best-of-3 Match | DoF $\ge 9$ Best-of-3 | Mean RMS (Å) | Clash Rate | Median Vol Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `vanilla-diffcsp` | 81.32% | 86.60% | 57.20% | 0.0387 | 3.7% | 6.84% |
| `diffcsp-geo` (500e, best ep 320) | 83.46% | 88.40% | 61.74% | 0.0373 | 1.5% | 3.90% |
| `diffcsp-geov2` (400e, best ep 300) | 82.59% | 87.60% | 59.47% | 0.0407 | 0.7% | 3.26% |
| `diffcsp-geo-orb` (200e, handoff $t=0.10$) | 78.92% | 86.70% | 57.20% | 0.0367 | 1.6% | **1.06%** |
| **`diffcsp-geo-rank3` ($M=3$ ORB rank)** | **85.89%** | **89.60%** | **67.80%** | 0.0329 | **0.5%** | 2.42% |
| **`diffcsp-geov2-rank3` ($M=3$ ORB rank)** | **85.66%** | **89.10%** | **64.77%** | **0.0316** | **0.5%** | 1.47% |

### 4.4 Scaling `DiffCSP-GeoV2` to Alex-MP-20 (W&B `ua1g6od4`)
On 2026-09-29 (`9a4aa98` / `f9e3b3a`, `scripts/train_geov2_alex_mp20.sh`), `DiffCSP-GeoV2` (`512x6`, batch size 128, $\text{lr}=5\times 10^{-4}$, 150 epochs $\approx 712\text{k}$ steps) was trained on **Alex-MP-20** (607,683 train / 67,521 val structures with $\le 20$ atoms and $E_{\text{hull}} < 0.1\text{ eV}$). With $22\times$ more training structures than MP-20, the 12.28M model showed **zero overfitting** across all 150 epochs (W&B [`different-morning-75` / `ua1g6od4`](https://wandb.ai/symmetry-advantage/diffcsp/runs/ua1g6od4)): validation loss decreased monotonically to **0.3035** at epoch 140 (train loss **0.2975**) and finished at **0.3058** at epoch 150. See [`docs/lemat-geov2-variants.md`](../lemat-geov2-variants.md) for the subsequent 4-GPU LeMat-Bulk `GeoV2` runs.

---

## 5. Artifacts & Code Reference
- Model implementations: [`diffcsp/models/geo_cspnet.py`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_cspnet.py), [`diffcsp/models/geo_v2_cspnet.py`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_v2_cspnet.py), [`diffcsp/models/geo_orb_cspnet.py`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_orb_cspnet.py)
- Diffusion wrappers: [`diffcsp/models/geo_diffusion.py`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_diffusion.py), [`diffcsp/models/geo_v2_diffusion.py`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_v2_diffusion.py), [`diffcsp/models/geo_orb_diffusion.py`](file:///home/kna/DiffCSPNew/diffcsp/models/geo_orb_diffusion.py)
- Training & eval scripts: [`scripts/train_geo_500e.sh`](file:///home/kna/DiffCSPNew/scripts/train_geo_500e.sh), [`scripts/train_geov2_mp20.sh`](file:///home/kna/DiffCSPNew/scripts/train_geov2_mp20.sh), [`scripts/train_geo_orb.sh`](file:///home/kna/DiffCSPNew/scripts/train_geo_orb.sh), [`scripts/eval_geo_rank3.sh`](file:///home/kna/DiffCSPNew/scripts/eval_geo_rank3.sh), [`scripts/eval_geov2_rank3.sh`](file:///home/kna/DiffCSPNew/scripts/eval_geov2_rank3.sh), [`scripts/train_geov2_alex_mp20.sh`](file:///home/kna/DiffCSPNew/scripts/train_geov2_alex_mp20.sh)
- Unit tests: [`tests/test_geo.py`](file:///home/kna/DiffCSPNew/tests/test_geo.py), [`tests/test_geo_v2.py`](file:///home/kna/DiffCSPNew/tests/test_geo_v2.py), [`tests/test_geo_orb.py`](file:///home/kna/DiffCSPNew/tests/test_geo_orb.py)
- Checkpoints: `runs/mp20_geo/geo_mp20_500e_best.pt` (ep 320), `runs/mp20_geov2/geov2_mp20_400e_best.pt` (ep 300), `runs/mp20_geo_orb/geo_orb_mp20_200e_best.pt` (ep 180), `runs/alex_mp20_geov2/geov2_alex_mp20_150e.pt` (ep 140)
- Benchmark results: `runs/bench/mp20/results_geo_500e.pkl`, `runs/bench/mp20/results_geov2.pkl`, `runs/bench/mp20/results_master.pkl`, `runs/bench/mp20/results_geov2_rank3.pkl`

