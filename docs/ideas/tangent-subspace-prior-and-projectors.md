# Orthonormal Tangent-Space Projectors & Uniform Subspace Priors

**Status:** Untried fixes (diagnosed in [`docs/asymmetric-unit-deficit.md`](../asymmetric-unit-deficit.md) §5.1 and §5.2 on 2026-09-11, commits `f06f75d` and `ec06e6d`).  
**Scope:** Affects [`WyckoffDiffusion`](../../diffcsp/models/wyckoff_diffusion.py), [`GeoDiffusion`](../../diffcsp/models/geo_diffusion.py), [`GeoV2Diffusion`](../../diffcsp/models/geo_v2_diffusion.py), [`GeoOrbDiffusion`](../../diffcsp/models/geo_orb_diffusion.py), and (for §2) [`CSPDiffusion`](../../diffcsp/models/diffusion.py).

---

## 1. Oblique Projector Noise Mis-Specification (`Pa @ eps`)

### Diagnosis
In `WyckoffDiffusion`, `GeoDiffusion`, `GeoV2Diffusion`, and `GeoOrbDiffusion`, tangent-space coordinate noise is injected via the anchor's linear rotation block `Pa = batch.ops[batch.anchor_index, :3, :3]`:
```python
rand_x_anchor = (Pa @ rand_x_raw[batch.anchor_index].unsqueeze(-1)).squeeze(-1)
```
Whereas vanilla `CSPDiffusion` (`diffcsp/models/diffusion.py`) uses `ops_inv = pinv(rotation_matrix)`.

For a symmetric projector ($M = M^\top$), `pinv(M) == M` and the two coincide. However, `pyxtal`'s `wp.ops[0].rotation_matrix` is idempotent ($M^2 = M$) and **not always symmetric**: the Reynolds average $\frac{1}{|H_k|} \sum_{R \in H_k} R$ is orthogonal only when $R$ is orthogonal, which fails in the fractional basis of trigonal and hexagonal space groups. For a site such as $(x, 2x, z)$:
```
M = [[1, 0, 0], [2, 0, 0], [0, 0, 1]]    (M @ M == M, but M != M.T)
Per-component noise std:
  Pa @ eps       -> (1.00, 2.00, 1.00) * sigma
  pinv(Pa) @ eps -> (0.45, 0.00, 1.00) * sigma
```
Because `d_log_p_wrapped_normal(sigmas * rand_x_anchor, sigmas)` assumes each coordinate component has standard deviation `sigma`, having a component with standard deviation `2 * sigma` mis-specifies the score target and alters the effective noise schedule on those sites.

On the MP-20 benchmark set, oblique projectors occur in **240 of 4,903 sites (4.9%)**, concentrated in **22.0% of $\text{DoF} < 9$ structures** (and 1.1% of $\text{DoF} \ge 9$ structures).

### Proposed Fix
Construct an **orthonormal tangent-space projector** $\mathbf{P}_a^{\perp} = \mathbf{U}_a \mathbf{U}_a^\top$ (where the columns of $\mathbf{U}_a \in \mathbb{R}^{3 \times \text{DoF}_a}$ form an orthonormal basis for the column space $\text{col}(\mathbf{P}_a)$, or equivalently $\mathbf{P}_a (\mathbf{P}_a^\top \mathbf{P}_a)^+ \mathbf{P}_a^\top$), or parameterize noise and score targets directly in the $\text{DoF}_a$-dimensional parameter space before mapping to fractional coordinates.

---

## 2. Non-Uniform Sampler Prior on Diagonal Wyckoff Subspaces

### Diagnosis
All diffusion samplers in the codebase (`CSPDiffusion`, `WyckoffDiffusion`, `GeoDiffusion`, `GeoV2Diffusion`, `GeoOrbDiffusion`) initialize fractional coordinates at $t = T$ by projecting a uniform 3D hypercube $\mathbf{u} \sim \text{Uniform}([0, 1)^3)$:
```python
x_rand = torch.rand([batch.num_nodes, 3], device=self.device)
x_anchor = ((Pa @ x_rand[batch.anchor_index].unsqueeze(-1)).squeeze(-1) + anchor_x0) % 1.0
```
Projecting a uniform cube onto a linear subspace is **not uniform on the torus subspace** unless the subspace is axis-aligned:

| Site Form | Std (Uniform $= 0.289$) | Min Decile (Uniform $= 0.100$) | Max Decile |
| :--- | :---: | :---: | :---: |
| $(x, y, z)$ | 0.289 | 0.099 | 0.101 |
| $(x, 0, 1/2)$ | 0.289 | 0.099 | 0.101 |
| $(x, x, z)$ | 0.204 | 0.020 | 0.180 |
| $(x, x, x)$ | 0.166 | **0.004** | 0.217 |

For diagonal 1-DoF sites like $(x, x, x)$ (where $P_a = \frac{1}{3}\mathbf{1}\mathbf{1}^\top$, so the projected coordinate is $\frac{u_1 + u_2 + u_3}{3}$, an Irwin–Hall distribution), the outer deciles near 0 and 1 are **25x under-sampled**. The forward diffusion process has the matching defect: the per-component noise standard deviation along $(x, x, x)$ is $\sigma / \sqrt{3}$, so at $\sigma_{\text{end}} = 0.5$ the terminal distribution retains a ~19% first Fourier mode instead of converging to uniform on $[0, 1)$.

### Proposed Fix
1. **Sampler Initialization:** Draw independent uniform parameters $\mathbf{z}_a \sim \text{Uniform}([0, 1)^3)$ and apply an integer-lattice-preserving basis of $\mathcal{T}_a$ (e.g. setting $x = z_1$ directly for $(x, x, x)$ or $(x, 2x, z)$ by extracting the independent free coordinate columns of `Pa` rather than averaging three uniforms).
2. **Forward Process Scaling:** Scale tangent noise so that each independent periodic degree of freedom on $\mathcal{T}_a / \mathbb{Z}^{\text{DoF}_a}$ receives full $\sigma_t$ variance, ensuring convergence to the uniform torus distribution at $t = T$.
