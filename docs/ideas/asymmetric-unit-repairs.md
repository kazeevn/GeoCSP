# Repairing the Asymmetric-Unit Graph ($K \times N$)

**Status:** Untried ideas (extracted from [`docs/asymmetric-unit-deficit.md`](../asymmetric-unit-deficit.md) §6–§7 and [`docs/archive/equivariant-painn-study.md`](../archive/equivariant-painn-study.md) §6.2).

---

## 1. Context

Collapsing the $N$-atom conventional cell graph ($N \times N$ edges) to the $K$-site asymmetric unit ($K \times N$ bipartite edges in [`WyckoffCSPNet`](../../diffcsp/models/wyckoff_cspnet.py)) delivers a **14x forward speedup** on large cells and up to **32x edge reduction**, while matching or beating full-cell vanilla DiffCSP++ on $\text{DoF} \le 5$.

However, as established in [`docs/asymmetric-unit-deficit.md`](../asymmetric-unit-deficit.md) (2026-09-11, commits `f06f75d` and `ec06e6d`), the asymmetric-unit graph suffers an **8.1-point per-trial match-rate deficit** on structures with $\text{DoF} \ge 9$ and orbit multiplicity $m > 1$. Layer-by-layer analysis proved that:
- Layer 1 is mathematically identical between full-cell and asymmetric-unit graphs ($6.1 \times 10^{-8}$ relative L2 difference).
- Divergence begins at Layer 2 because every replica $j$ of an anchor $s(j)$ is assigned the same unrotated scalar hidden state $\mathbf{h}_j = \mathbf{h}_{s(j)}$, losing the $m$ distinct perspective states that full-cell CSPNet implicitly ensembles over.

While full-cell [`DiffCSP-Geo`](../diffcsp-geo-study.md) and [`DiffCSP-GeoV2`](../lemat-geov2-variants.md) sidestepped this by keeping all $N$ atoms in the graph (requiring `--max_atoms 128` truncation on LeMat-Bulk), the following three untried ideas aim to recover full-cell high-DoF accuracy directly on the fast $K \times N$ asymmetric-unit graph.

---

## 2. Three Untried Routes

### Route A: Replica Operation Conditioning in `WyckoffCSPLayer`
In [`diffcsp/models/layers.py`](../../diffcsp/models/layers.py), `WyckoffCSPLayer` already computes individualized edge displacements `frac_diff = (xj - xi) % 1.0`, so each incoming edge knows which replica $j$ it originated from. However, `hj = site_features[source_sites]` discards the replica identity in the source node representation.
- **Proposal:** Condition the source node feature on the space-group affine operation $(R_j, \mathbf{t}_j)$ that maps anchor $s(j)$ to replica $j$:
  $$\mathbf{h}_j = \text{MLP}\left(\left[\mathbf{h}_{s(j)}, \text{Flatten}(R_j), \mathbf{t}_j\right]\right)$$
- **Cost:** Preserves the exact $K \times N$ edge count and $K$-node update cost while allowing the network to distinguish orbit replicas from Layer 2 onward.

### Route B: Orbit Representative (Anchor Gauge) Randomization During Training
In [`docs/asymmetric-unit-deficit.md`](../asymmetric-unit-deficit.md) §7, the viewpoint gauge diagnostic (`bench/viewpoint_gauge.py`) showed that swapping which atom in a Wyckoff orbit is designated as the anchor changes `WyckoffCSPNet`'s predicted coordinate score by **32.6% at $m = 2$** and **62.5% at $m = 5\text{--}8$** (whereas global Wyckoff-setting augmentation is a no-op on 69% of high-DoF multi-orbit structures because their alternative settings are pure origin shifts).
- **Proposal:** During training, independently draw a random representative index $r_k \in \{1, \dots, m_k\}$ for each Wyckoff orbit $k$, set atom $r_k$ as the anchor, and conjugate the orbit operations and projector:
  $$(R_i, \mathbf{t}_i) \leftarrow (R_i, \mathbf{t}_i) \circ (R_{r_k}, \mathbf{t}_{r_k})^{-1}, \qquad \mathbf{P}_k \leftarrow R_{r_k} \mathbf{P}_k R_{r_k}^{-1}$$
- **Cost:** Zero architectural overhead or inference cost; trains the asymmetric-unit network to be invariant to the $\prod_k m_k$ anchor gauge choices.

### Route C: Long-Cutoff or Fourier-Augmented Equivariant `WyckoffPaiNN`
In [`docs/archive/equivariant-painn-study.md`](../archive/equivariant-painn-study.md) (2026-09-12, commit `c4793d7`), `WyckoffPaiNN` eliminated the anchor viewpoint gauge exactly by rotating replica vector channels via the closed-form Cartesian space-group isometry $\vec{\mathbf{v}}_j = \vec{\mathbf{v}}_a \mathbf{M}_j$ ($\mathbf{M}_j = \mathbf{L}^{-1} R_j^\top \mathbf{L} \in O(3)$). However, its overall match rate lagged `WyckoffCSPNet` (62.0% vs 76.6% per-trial at 210 epochs) because PaiNN relied exclusively on a finite Euclidean radial cutoff ($r_{\text{cut}} = 6.0\text{ \AA}$), starving message passing during early reverse diffusion ($t \sim 1000 \to 700$) when unit cells are expanded or atoms are dispersed.
- **Proposal:** Combine `WyckoffPaiNN`'s exact $O(3)$ replica vector rotation $\vec{\mathbf{v}}_j = \vec{\mathbf{v}}_a \mathbf{M}_j$ with either:
  1. An expanded radial cutoff ($r_{\text{cut}} = 10.0\text{--}12.0\text{ \AA}$) across periodic images, or
  2. Global periodic torus Fourier features ($\sin/\cos(2\pi \mathbf{f} \odot \Delta \mathbf{x}_{ij})$) in the scalar filter stream alongside the Euclidean Bessel RBFs (mirroring the dual torus + Euclidean edge representation that succeeded in `GeoCSPNet`).
