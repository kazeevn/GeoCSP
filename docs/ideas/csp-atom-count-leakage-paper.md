# Paper plan: the atom count in ML crystal structure prediction benchmarks

**Status:** idea / plan, written 2026-10-21 at DiffCSPNew `028ea24`. No experiment below has been
run *for this paper*; existing results are cited with their provenance. Everything marked
**to do** is planned only.

## Title options

1. *Prediction or solution? The cell content as an oracle input in crystal structure prediction benchmarks*
2. *How much does the atom count tell? Formula-only evaluation of generative crystal structure prediction*
3. *Structure prediction from the formula alone: re-evaluating MP-20 CSP without the ground-truth cell content*

## Claim (one paragraph)

The standard ML CSP benchmark (MP-20, MPTS-52, Perov-5, Carbon-24) gives the model the full
atom list of the ground-truth cell, hence its atom count N and number of formula units Z.
The field describes the task as "predict the structure of a composition", but for that task N
is an oracle input: it is a property of the answer, it discriminates polymorphs (ZnS:
zincblende has 2 atoms per primitive cell, wurtzite 4), and it is informative about the space
group and the prototype. In fact, a non-learned lookup gets a large part of its accuracy from
it. The benchmark therefore measures **structure solution with a known cell content** (legitimate when a
diffraction pattern has been indexed and Z inferred from the density), not **formula-only
prediction**, where classical CSP codes (USPEX, CALYPSO, AIRSS) scan Z as part of the
search. No paper we surveyed states which of the two it measures, calls the input leaky, or
reports a formula-only MP-20 match rate. We define both tasks precisely, quantify the
information N carries, re-evaluate open models under formula-only evaluation at a fixed sample budget,
and release the protocol.

## Problem definitions

- **(A) Structure solution with known cell content.** Input: the multiset of species of the
  ground-truth cell (N, Z). This is the current protocol. Physically, it corresponds to an
  indexed pattern: V comes from indexing, and Z = ρ V N_A / M comes from a measured density ρ
  and formula mass M, or from atomic-volume rules.
- **(B) Formula-only CSP.** Input: the reduced formula. Z is unknown and the model must
  scan or predict it. Output: a ranked list of K candidates of any size, with **the budget K
  shared across Z** (e.g. K = 20 total, not 20 per Z). The candidate cap is
  N_max = 20 atoms for MP-20 (52 for MPTS-52), which must be stated. A candidate is a hit if
  `StructureMatcher` (DiffCSP tolerances: ltol 0.3, stol 0.5, angle_tol 10) matches it to the ground truth.
  Comparison is on primitive cells, so a conventional cell or supercell of the right
  structure counts as a hit, and a structure with the wrong primitive Z is a miss. We
  report Recall@K for K ∈ {1, 5, 20} and an atom-weighted budget (Σ atoms over candidates),
  since relaxation cost grows with N.
- **(C) Formula + measured cell volume (middle ground).** Input: the reduced formula and V
  of the primitive cell, with Z left for the model to infer. This uses a density or an
  atomic-volume rule, in the same way an experimentalist would. It quantifies how much of (A)
  is recoverable from realistic measurements. DFT (PBE) volumes are biased relative to
  experiment, which is a caveat for this setting.

The precise statement for the paper is that (A) is well posed for structure solution, and
leaky for "predict the ground state of a formula". The benchmark conflates the two.

## Evidence base (already established)

| # | result | provenance |
|---|---|---|
| E1 | Every MP-20 CSP paper surveyed gives N: DiffCSP, DiffCSP++ (CSPML templates), FlowMM ("the user provides the number of constituent atoms"), CrystalFlow, OMatG, EquiCSP, Chemeleon (N "derived from the test set structures"), CrystaLLM benchmark prompts (cell composition; a Z scan exists only in its web app), uFlowCSP ("the number of atoms is known"). None calls this leakage or reports a formula-only MP-20 number. CrystalFormer-CSP takes a formula but is evaluated elsewhere. | Literature survey, 2026-10-21. arXiv 2309.04475, 2402.03992, 2406.04713, 2307.04340, 2609.09799, 2512.18251. github.com/jiaor17/DiffCSP, github.com/jiaor17/DiffCSP-PP |
| E1a | DiffCSP sampling takes `batch.atom_types` and `batch.num_atoms` from the test loader. Its `conf/data/mp_20.yaml` sets `niggli: true`, `primitive: False`. DiffCSP++'s CSPML retrieval (`scripts/csp_from_template.py`) keys templates on the sorted per-element **counts of the cell**, although the paper describes this as "the same compositional ratio". CSPML alone reaches 70.51% on MP-20, against 70.58% for DiffCSP++ w/ CSPML (Table 5 / Table 2). | Code read 2026-10-21. The yaml was fetched from GitHub `main` |
| E2 | MP-20 test (9,046 stored cells): Z = 1 / 2 / 3 / 4 / >4 for 4,805 / 3,136 / 189 / 861 / 55 cells, so 46.9% have Z > 1. 171 reduced formulas occur with more than one cell size within the test split. Only 64 stored cells reduce further (`get_primitive_structure(tol=0.01)`), so the input is effectively the primitive cell. | Scratch `zcheck.py`, rerun 2026-10-21 |
| E3 | Space-group lookup by anonymised stoichiometry over MP-20 train+val, evaluated on MP-20 test, top-1 / 5 / 20: **stoichiometry only 33.8 / 64.0 / 88.9%**; **+ primitive N 45.3 / 77.5 / 93.5%**; + conventional-cell N 55.0 / 83.5 / 94.5%. The plug-in conditional MI is I(sg; N_prim \| stoich) = 0.93 bits, against H(sg \| stoich) = 3.55 bits. The plug-in estimate is biased upward. | Scratch `wyformer_audit.py` (conventional N) and `leak_prim.py` (primitive N = N_conv / centring factor), 2026-10-21 |
| E4 | On the 1000-entry benchmark subset, a prototype lookup gives these ground-truth Wyckoff-gene recalls at top-1 / 5 / 20: **with N 33.0 / 70.3 / 89.1%**, and **without N 19.1 / 45.3 / 74.9%**. Without N, the top-1 candidate has the ground truth's atom count in only 38.9% of cases. The ground-truth prototype occurs in train+val for 94.5% of formulas. | Scratch `prototype_baseline.py` and `leak_prelim.py`, 2026-10-21. Benchmark set as in [`../benchmark.md`](../benchmark.md) |
| E5 | Our pipeline with N given (`uncond-cell`: WyFormer → DiffCSP-Geo → PET-OAM-XL → ORB) reaches classic R@1 / 5 / 20 = 24.7 / 49.5 / 66.4% and energy-ranked relaxed R@1 = 52.8%. With the oracle gene, Geo reaches 83.3% raw and 88.2% energy-ranked R@1. | W&B `symmetry-advantage/diffcsp` `5mh3v9wl`, scored at `4b609b5` on 2026-10-21. The oracle gene results are `1t20dz31` (control `zw2t1pj8`), code `33e3224`. See [`../archive/csp-benchmark-results.md`](../archive/csp-benchmark-results.md) |
| E6 | When WyFormer chooses z itself, only 25.6% (`uncond`) and 21.9% (`eform`) of its samples have the ground truth's atom count. | Scratch `wyformer_audit.py` §2, rerun 2026-10-21 on `runs/bench/mp20/formula/genes_{uncond,eform}.pkl` |
| E7 | A stoichiometry-frequency prior predicts the primitive Z of MP-20 test entries with top-1 / 2 / 3 = 65.0 / 90.2 / 98.0%. Always guessing Z = 1 is right for 53.7%. | Scratch `leak_prim.py`, 2026-10-21 |
| E8 | Across all MP-20 splits, 2,494 reduced formulas have entries with more than one primitive cell size. 1,415 test entries (15.6%) belong to such formulas. | Scratch `poly_split.py`, 2026-10-21 |

These are the published MP-20 single-sample match rates (all with N given): DiffCSP 51.49, FlowMM 61.39,
CrystalFlow 62.02 (k = 1; 78.34 at k = 20), DiffCSP++ w/ CSPML 70.58, and DiffCSP++ w/ GT template 80.27.

**Before submission, the scratch scripts must be committed** under `bench/leakage/` and their
outputs logged to W&B. They currently live only in the session scratchpad. Primitive N in E3, E7
and E8 is derived from the conventional cell and the centring letter. It should be recomputed from the stored
cells, since 64 of those are non-primitive.

## Experiments

Compute is in GPU-hours on one RTX 6000 Ada unless stated otherwise. "Exists" means a cited result above.

### X1. How much does N reveal? (information-theoretic and lookup analysis)
- **Hypothesis:** given the anonymised stoichiometry, N carries a substantial share of the
  remaining information about the space group, the prototype and polymorph identity.
- **Data:** the MP-20 train+val reference, scored on test. Repeat on MPTS-52 (time split) and on
  Alex-MP-20 / LeMat-Bulk as larger references.
- **Metrics:** top-k lookup accuracy for sg, prototype and gene, with and without N_prim (and
  N_conv, which additionally reveals the lattice centring). We also estimate I(Y; N | stoich) with a
  bias-corrected estimator (Miller–Madow or NSB) and a held-out cross-entropy version, i.e. the
  log-loss reduction of a smoothed lookup.
- **Expected:** +10–12 points sg top-1 from N_prim (E3), and a larger gain at the prototype level (E4).
- **Status:** sg on MP-20 **done (preliminary)**. Prototype, MI estimators, MPTS-52 and Alex **to do**.
- **Compute:** CPU only, under an hour.

### X2. Re-evaluate published models in formula-only (B)
- **Hypothesis:** at a fixed total budget, Z-scanning lowers match rates materially, by more
  for weaker models. The ranking of models may change.
- **Models:** we need open checkpoints, and their availability must be checked for each model. The candidates are
  DiffCSP (jiaor17/DiffCSP), FlowMM (facebookresearch/flowmm), CrystalFlow and DiffCSP++
  (CSPML retrieval with a Z-free key). We can retrain DiffCSP with `diffcsp/` (the `cspnet` regime) if no checkpoint
  is available. All of these models condition on an atom list, so a Z-scan only needs the input to be
  `reduced formula × Z` for Z with N ≤ N_max.
- **Allocation policies at total budget K:** (a) uniform over feasible Z; (b) proportional to
  the stoichiometry Z prior of E7; (c) all samples at the prior's argmax Z; (d) energy-adaptive (X5).
  We also report the per-Z oracle (the setting (A) number) as the upper bound.
- **Metric:** Recall@K, K ∈ {1, 5, 20}, as defined in (B), plus the atom-weighted budget.
- **Expected:** a rough bound is R_B@1 ≈ P(Ẑ = Z) · R_A@1. With E7's 65% top-1 Z prior,
  DiffCSP would drop from 51 to roughly 33–40%.
- **Data:** full MP-20 test (9,046), matching the literature. We also report the 1000-entry subset.
- **Compute:** sampling scales with Σ_Z N. At K = 20 over about 3 Z values, this costs about 1.5–2× a standard
  20-sample run per model. That is roughly 5–20 GPU-h per model without relaxation.
- **Status:** **to do**. It needs free GPUs.

### X3. Trivial baselines with and without N
- **Hypothesis:** a non-learned prototype lookup with N recovers a large fraction of the
  learned models' reported match rates, and without N it loses more than the learned models do.
- **Method:** turn the lookup's top-k genes (E4) into structures with `from_random` + Geo or
  a fixed lattice/volume prior, then score against the structure (not just the gene). Use CSPML-style retrieval with
  the published key (counts) against a reduced-ratio key. That second comparison isolates the N
  effect inside DiffCSP++.
- **Metric:** the structure match rate, R@1/5/20, on the full MP-20 test in (A) and (B).
- **Expected:** the gene-level lookup is at 33.0% top-1 with N (E4), close to our learned WyFormer (E5, 20.1% gene
  R@1). We expect the structure-level lookup to land in the range of published learned models at k = 1.
- **Status:** gene-level **done (preliminary)**. Structure-level and CSPML ablation are **to do**.
  CSPML is CPU plus a small MLP; Geo and relaxation take a few GPU-h.

### X4. Our WyFormer → Geo pipeline in (A), (B) and (C)
- **Models:** `uncond-cell`, `eform-cell` and `eform-cfg-cell` cover (A). `uncond` and `eform-cfg` (z
  chosen by the model) cover (B). A cell-size-conditioned WyFormer, which samples Z from its own
  prior or from E7's lookup, covers (B) with an explicit Z-scan and (C) with Z inferred from V.
- **Status:** (A) `uncond-cell` **done** (E5). `eform-cell` and `eform-cfg-cell` are **in progress**
  (`runs/bench/mp20/formula/`). The z-free regimes are **in progress / queued**, and their numbers are pending.
  Training of the cell-size-conditioned model is **in progress** on hera (W&B
  `WyckoffTransformer/csp-cell-hera-20261021-mp20_unconditional_cell`). (C) is **to do**.
- **Caveat:** z-free cells up to 80 conventional atoms dominate relaxation time. 40–48% of samples have
  more than 40 atoms, and a 60–80-atom cell takes a median 85 s. The (B) cap must match the
  dataset's N_max.
- **Compute:** about 13.5 h on 2 GPUs per regime with relaxation (measured for `uncond-cell`).

### X5. Does energy ranking pick the right Z?
- **Hypothesis:** MLIP energy per atom ranks candidates across Z reasonably well, because the
  ground truth is a minimum for any Z. Two things work against it: large-Z samples are more
  often saddle points (`uncond-cell` saddle rate 52.6%), and spurious low-energy supercells of
  wrong structures appear.
- **Method:** pool relaxed candidates across Z and rank by ORB E_hull. We report the ranked
  R@1 in (B) and the fraction of formulas whose top-ranked candidate has the right primitive Z. The
  "lower polymorph at another Z" bucket of the three-way breakdown is kept separate.
- **Status:** **to do**. It depends on the X4 z-free relaxations.

### X6. Polymorph split
- **Hypothesis:** the gap between (A) and (B) is largest for formulas whose known polymorphs differ in
  Z (E8: 1,415 test entries). On those formulas, N largely names the target polymorph.
- **Method:** stratify X2–X4 by whether the formula has one or several cell sizes in MP-20 and by
  ground-truth Z. We also run a deliberate test: for formulas with several polymorphs, ask (A) for
  each polymorph's N and check whether the "same formula" task yields different answers by N alone.
- **Status:** counts **done (preliminary)**, stratified evaluation **to do**. It costs CPU on top of X2–X4.

### X7 (optional). How recoverable is Z in setting (C)?
- Infer Z from the ground-truth V and Hofmann / Kempster–Lipscomb atomic volumes, then report the accuracy.
  If it is high, (C) is nearly (A), which supports the "legitimate for solution" framing.
- **Status:** **to do**. CPU only.

## Figures and tables

1. **Schematic:** the information flow in (A), (B) and (C), using the ZnS zincblende/wurtzite example.
2. **Table 1:** a literature audit with columns for paper, input (atom list / N / formula), source of N,
   whether this is stated as a limitation, and the reported MP-20 MR (E1).
3. **Figure 2:** stacked bars of lookup accuracy (sg / prototype / gene) for stoich, stoich + N_prim
   and stoich + N_conv. The panel includes the MI estimates (X1).
4. **Table 2 (headline):** R@1/5/20 for each model in (A) against (B) under every allocation policy, plus the lookup
   baselines (X2, X3).
5. **Figure 3:** match rate against the total budget K, with one curve each for (A) and (B) per model.
6. **Figure 4:** the per-Z breakdown, i.e. recall by ground-truth Z and the polymorph-split strata (X6).
7. **Table 3:** energy ranking across Z, the right-Z rate and the three-way breakdown (X5).

## Related work and positioning

- **Benchmark critiques of match-based evaluation:** arXiv 2509.12178 ("All that structure
  matches does not glitter") critiques the match metric and the datasets. We address the
  *input*, not the metric, and the two are complementary. CSPBench evaluates classical and ML CSP
  algorithms, and before writing we need to check what input it gives (formula or cell content).
- **Matbench Discovery:** frames the issue as task–metric alignment (retrospective against prospective). We
  make the analogous point for CSP: the protocol must match the claimed use case.
- **Classical CSP:** USPEX, CALYPSO and AIRSS scan Z (or the number of formula units) inside the
  search budget. Their papers report cost per Z, and this is the natural precedent for (B).
- **Formula-only ML CSP:** CrystalFormer-CSP and CrystaLLM's web app sample Z. In those cases,
  the formula-only number exists but not on MP-20.
- **Experimental structure solution:** powder diffraction with generative models (e.g. arXiv
  2409.04727) is setting (A)/(C) by construction, and we cite it as the legitimate home of (A).

## Anticipated objections and responses

- **"N is experimentally known."** This is true for structure *solution*: indexing gives V, and density
  or volume rules give Z. We agree, and define (A) as that task with (C) as its realistic
  version. But the papers motivate the benchmark as predicting structures of new compositions,
  where no pattern exists. In that case N is a property of the unknown answer, and classical CSP
  codes scan it. X7 measures how much of N a volume measurement recovers.
- **"Z is mostly 1 or 2, so this is a small effect."** E2 shows that 47% have Z > 1, E7 that a
  prior is right only 65% of the time, and E3 that N_prim adds 11.5 points of sg top-1 to a lookup. The paper
  stands on the measured gap, whatever its size, and a small gap for strong models is itself a result.
- **"Models could just scan Z, so this is trivial."** Yes, and our point is that this is never done or
  reported, and that it costs budget. X2 quantifies the cost.
- **"StructureMatcher already reduces to primitive cells."** It does, so Z enters through the
  *input*, not the scoring. Our protocol keeps the matcher unchanged.
- **"The DiffCSP++/CSPML gain is from metric learning, not from N."** The CSPML ablation in X3
  answers this directly.
- **"MP-20 is DFT and contains metastable entries."** This is orthogonal to the input, and the
  existing caveat in [`../benchmark.md`](../benchmark.md) already covers it.

## Risks

- **The effect may be small for strong models.** If (B) costs only a few points, the paper becomes a
  protocol and baseline note. Mitigation: lead with the information analysis and the lookup
  baselines, which do not depend on model strength.
- **The subset is biased.** The 1000-entry benchmark subset keeps only structures whose
  `from_random` conventional cell has the stored cell's site count. Only **0.8%** of it has a centred lattice,
  against 58% of MP-20 test. Headline numbers must use the full test set, and the subset can only
  be used for our pipeline with an explicit caveat.
- **Conventional and primitive N get confused.** Our first audit keyed on the conventional count
  (E3, 55.0%), which additionally leaks the centring. DiffCSP-protocol models receive the primitive count, so the
  paper must use N_prim (45.3%) for claims about the literature.
- **Checkpoints may be unavailable or unreproducible.** Retraining DiffCSP and FlowMM is affordable, but published
  numbers may not reproduce, so we report both the published and the reproduced numbers.
- **A bounded Z-scan has its own cap.** The cap N_max = 20 still leaks a little, since it is a dataset property.
  We state it, and we test sensitivity with N_max = 40.
- **Results may go out of date.** z-free numbers from `runs/bench/mp20/formula/` are pending. Anything quoted must
  carry the date, commit and W&B run.

## Timeline (staged)

1. **Week 1 (CPU):** commit the scratch analyses as `bench/leakage/`, recompute N_prim from stored
   cells, run X1 on MP-20/MPTS-52, the X3 CSPML ablation and X7, and log everything to W&B. Draft Table 1.
2. **Weeks 2–3 (GPU, after the current benchmark runs):** check checkpoint availability, then run
   X2 for DiffCSP and FlowMM on full MP-20 test with the four allocation policies. The X4 z-free
   regimes finish, and the cell-size-conditioned WyFormer is evaluated.
3. **Week 4:** X5 energy ranking across Z, X6 stratification, X3 structure-level lookup.
4. **Week 5:** CrystalFlow, DiffCSP++ and MPTS-52 extensions, the figures, and the writing.
   Before writing, decide between a workshop note (if the gap is under about 5 points) and a full
   paper.
