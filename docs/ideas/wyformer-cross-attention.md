# WyFormer Latent Cross-Attention & Analytic Repulsion Guidance

**Status:** Untried ideas (extracted from [`docs/archive/architectural-innovations.md`](../archive/architectural-innovations.md), Innovations 3 & 4).  
**References:** WyFormer (*ICML 2025*), WyckoffDiff-Adaptor (*2025*).

---

## 1. WyFormer Latent Cross-Attention

### Background
In [`diffcsp/models/geo_cspnet.py`](../../diffcsp/models/geo_cspnet.py) and [`diffcsp/models/geo_v2_cspnet.py`](../../diffcsp/models/geo_v2_cspnet.py), the first half of Innovation 4 (discrete crystallographic embeddings for atomic number $Z$, space group $G$, orbit multiplicity $m$, and site DoF) has been implemented. However, the WyFormer $\to$ DiffCSP++ pipeline still communicates solely through serialized JSON Wyckoff genes (`spacegroup`, `species`, `wyckoff letters`), discarding WyFormer's internal learned representations.

*(Note: Embedding the raw Wyckoff letter token itself is deliberately avoided because Euclidean normalizer origin shifts relabel Wyckoff letters without changing geometry; see [`docs/asymmetric-unit-deficit.md`](../asymmetric-unit-deficit.md) §7.)*

### Proposed Mechanism
1. **Site Point-Group Symmetry Embedding:**
   Embed the crystallographic site-symmetry point group $\text{PointGroup}_k$ (one of the 32 crystallographic point groups of the stabilizer $H_k \le G$), which is invariant to origin-shift relabelings where the raw Wyckoff letter is not.
2. **Penultimate Transformer Cross-Attention:**
   Export WyFormer's penultimate per-site transformer hidden states $\mathbf{H}_{\text{WyFormer}} \in \mathbb{R}^{K \times d_{\text{trans}}}$ alongside the discrete Wyckoff gene, and insert cross-attention blocks into the GNN denoiser:
   $$\mathbf{h}_i \leftarrow \mathbf{h}_i + \text{MultiHeadAttention}\left(\text{query}=\mathbf{h}_i, \text{key}=\mathbf{H}_{\text{WyFormer}}, \text{value}=\mathbf{H}_{\text{WyFormer}}\right)$$
   This gives the continuous coordinate denoiser direct access to the global stoichiometry and coordination context learned by the autoregressive transformer.

---

## 2. Analytic Steric Repulsion Guidance in Full-Cell Geo/GeoV2 Diffusion

### Background
[`diffcsp/models/repulsion.py`](../../diffcsp/models/repulsion.py) implements [`RepulsivePotential`](../../diffcsp/models/repulsion.py#L75), a smooth conservative covalent-radius pair potential:
$$E_{\text{rep}}(r_{ij}) = k_{\text{rep}} \frac{(r_c - r_{ij})^3}{3 r_c}, \qquad r_{ij} < r_c = \eta (R_i + R_j)$$
with exact analytical Cartesian forces and virial stresses. Currently, `RepulsivePotential` is only invoked inside [`OrbBackboneWrapper`](../../diffcsp/models/orb_wrapper.py) (`--model orb` / `--model geo_orb`), not in standalone `GeoDiffusion` or `GeoV2Diffusion`.

### Proposed Mechanism
During late-stage reverse diffusion ($t \le t_{\text{rep}}$, e.g. final 100 steps) in `GeoDiffusion.sample` / `GeoV2Diffusion.sample`, add a symmetry-projected repulsive drift term from `RepulsivePotential`:
1. Evaluate Cartesian repulsive forces $\mathbf{f}_{\text{rep}, j} = -\nabla_{\mathbf{r}_j} E_{\text{rep}}$ on minimum-image interatomic distances below $\eta (R_i + R_j)$ ($\eta \approx 0.70$).
2. Transform $\mathbf{f}_{\text{rep}, j}$ to fractional coordinates via $\mathbf{L}_t^{-1}$, pull back to the Wyckoff anchor via `batch.ops_inv`, average over the orbit, and project onto the stabilizer tangent space $\mathbf{P}_a$.
3. Apply the resulting clash-free displacement during predictor/corrector steps on high-DoF structures ($\text{DoF} \ge 9$, where the remaining 1.5%–2.7% clash rate resides) without requiring an expensive MLIP forward pass.
