# Riemannian Flow Matching (RFM) on the Wyckoff Subspace

**Status:** Untried idea (extracted from [`docs/archive/architectural-innovations.md`](../archive/architectural-innovations.md), Innovation 5).  
**References:** FlowMM (*ICML 2024*), CrystalFlow (*2025*), uFlowCSP (*2026*).

---

## 1. Motivation & Bottleneck

All current diffusion models in the repository (`CSPDiffusion`, `CSPDiffusionORB`, `GeoDiffusion`, `GeoV2Diffusion`, `GeoOrbDiffusion`) rely on a **1,000-step predictor-corrector SDE sampler** (`diffcsp/models/diffusion.py`, `diffcsp/models/geo_v2_diffusion.py`).
- A standard 1,000-representation $\times$ 3-trial benchmark requires **3,000,000 GNN forward evaluations** (6,000,000 with predictor + corrector per step, or 18,000,000 when using $M=3$ candidate ranking).
- Wrapped-normal score matching (`d_log_p_wrapped_normal` in `diffcsp/core/schedulers.py`) has high target variance as $t \to 0$ ($\sigma_t \to 0.005$).

---

## 2. Proposed Formulation

Replace the 1,000-step predictor-corrector SDE with **Riemannian Flow Matching (RFM)**, defining deterministic probability paths along geodesics of the symmetry-constrained crystal manifold $\mathbb{T}^{\text{DoF}} \times \mathfrak{g}_G$:

1. **Lattice Flow on the Space-Group Lie Subalgebra:**
   Define the lattice trajectory in the 6D crystal-family Lie algebra $\mathfrak{g}_G = \mathbf{P}_G^{\text{lattice}} \mathbb{R}^6$:
   $$\mathbf{k}_t = (1 - t)\mathbf{k}_0 + t\mathbf{k}_1, \qquad \mathbf{L}_t = \text{v2m}(\mathbf{k}_t), \qquad t \in [0, 1]$$
   where $\mathbf{k}_0 \sim \mathcal{N}(0, \mathbf{P}_G^{\text{lattice}})$ and $\mathbf{k}_1 = \mathbf{P}_G^{\text{lattice}} \text{m2v}(\text{de\_so3}(\mathbf{L}_{\text{GT}}))$. The conditional target velocity is constant along the trajectory:
   $$\mathbf{v}_{\text{lat}}^* = \mathbf{k}_1 - \mathbf{k}_0 \in \mathfrak{g}_G$$

2. **Coordinate Flow on the Wyckoff Torus Subspace:**
   For each Wyckoff anchor $a$ with tangent projector $\mathbf{P}_a$ ($\text{DoF}_a > 0$), define the shortest-path geodesic interpolation on the flat torus subspace $\mathcal{A}_a = \mathbf{x}_0^{(a)} + \mathcal{T}_a$:
   $$\Delta \mathbf{x}_a = \mathbf{P}_a \left( (\mathbf{x}_{1}^{(a)} - \mathbf{x}_{0}^{(a)} + 0.5) \bmod 1.0 - 0.5 \right)$$
   $$\mathbf{x}_t^{(a)} = \left( \mathbf{x}_0^{(a)} + t \, \Delta \mathbf{x}_a \right) \bmod 1.0$$
   with full-cell replica coordinates $\mathbf{x}_t^{(j)} = (R_j \mathbf{x}_t^{(a)} + \mathbf{t}_j) \bmod 1.0$. The conditional target velocity in the anchor tangent space is simply the constant displacement vector:
   $$\mathbf{v}_{\text{coord}, a}^* = \Delta \mathbf{x}_a \in \mathcal{T}_a$$
   Zero-DoF sites ($\text{DoF}_a = 0$) remain locked at $\mathbf{x}_0^{(a)}$ for all $t \in [0, 1]$ and are masked from the coordinate loss.

3. **Training Objective & ODE Sampling:**
   - Regress the decoder's predicted velocity $(\hat{\mathbf{v}}_{\text{lat}}, \hat{\mathbf{v}}_{\text{coord}})$ directly against $(\mathbf{v}_{\text{lat}}^*, \mathbf{v}_{\text{coord}}^*)$ via mean-squared error, with optional time-dependent loss weighting.
   - Sample via 20–50 deterministic ODE steps (Euler, Midpoint, or RK4) projecting onto $\mathcal{A}_a$ and $\mathfrak{g}_G$ at each step.

---

## 3. Expected Impact & Primary Files

- **20x–50x Inference Speedup:** Reduces sampling from 2,000 network evaluations per draw (1,000 predictor + 1,000 corrector) to 20–50 ODE evaluations.
- **Smoother Training Objective:** Eliminates wrapped-normal score blowup near $t \to 0$.
- **Files to touch:** [`diffcsp/core/schedulers.py`](../../diffcsp/core/schedulers.py), [`diffcsp/models/geo_v2_diffusion.py`](../../diffcsp/models/geo_v2_diffusion.py).
