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
| `mlip_relax.py` | CSP benchmark: PET-OAM-XL relaxation, ORB energies and LeMat-Bulk E_hull of samples or ground truth ([below](#csp-benchmark--reconstructing-wyckoff-genes)) |
| `evaluate_csp.py` | CSP benchmark: Classic and MLIP-assisted Match Rate (Recall@N, ranked Recall@N, RMSD, three-way breakdown), saddle points, polymorph counting; logs to W&B |
| `csp_metrics.py` | the metric definitions, and the matching protocol `evaluate.py` also uses |
| `gen_csp_formula.py` | CSP benchmark, formula given: a composition-conditioned WyFormer samples the Wyckoff genes ([below](#csp-benchmark--csp-formula-given)); `gen_init.py --genes` draws them |

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

## CSP benchmark — Reconstructing Wyckoff genes

Implements the evaluations of [Reconstructing Wyckoff genes](ideas/csp_benchmarking.md#reconstructing-wyckoff-genes)
in [`ideas/csp_benchmarking.md`](ideas/csp_benchmarking.md): the model is given a Wyckoff gene and samples
structures conforming to it. CSP from the formula alone is [below](#csp-benchmark--csp-formula-given).
Driven by `scripts/run_csp_bench.sh`; the code and W&B call this the *gene track*
(`track=gene` in the W&B config since `43da227`; the first runs predate the field).

```
gene -> from_random x20 -> regime -> PET-OAM-XL relax -> ORB energy -> E_hull (LeMat-Bulk ORB hull)
                                     (WyckoffTransformer   (orb_conserv_inf)
                                      stepwise + rattle)
```

**Relaxation.** WyckoffTransformer's `cryspr.relaxer.stepwise_relax_stages` with PET-OAM-XL:
fix-cell warm-up, cell + positions under `FixSymmetry`, then symmetry released, then a rattle
(0.05 Å, 1% strain, kept if it wins 1 meV/atom); BFGS with `FrechetCellFilter`, `fmax` 0.05 eV/Å, 500 steps per
stage, 1200 s per structure. Two readouts:

- **sym**: the end of the symmetry-constrained stages, the *fixed-symmetry MLIP relaxation*
  of the ideas doc. It is still on the gene's Wyckoff orbits, so it is the **primary readout**.
- **free**: what the free relaxation and the rattle kept. Reported alongside.

**Energies.** ORB `orb-v3-conservative-inf-omat` single points, and E_hull against
LeMat-Bulk's published hull for the same model (`LeMaterial/LeMat-Bulk-MLIP-Hull`, split
`orb_conserv_inf`, revision `70d505bb`, 194,240 entries). Energies and hull are therefore
self-consistent and comparable with LeMat-Bulk-MLIP / LeMat-GenBench. Relaxation and energy
scoring use different MLIPs, PET and ORB, as the ideas doc's considerations ask. Of its
reference-dataset treatments, two are computed for the ground truth:

- `gt_dft`: **DFT structures and MLIP energies** (LeMat-GenBench style), ORB on the DFT structure;
- `gt_relaxed`: **references relaxed with the MLIP**, ORB on the ground truth's own PET relaxation.

**Metrics** (`evaluate_csp.py`). Samples are taken in trial order and N ∈ {1, 5, 20}; "the
first N samples" are trials 0 … N−1, and a sample that failed counts as a miss. Matching is
the DiffCSP protocol, `StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)`.

| ideas-doc section | metric | definition |
|---|---|---|
| [Classic Match Rate](ideas/csp_benchmarking.md#classic-match-rate-1) | Recall@N | a gene counts if one of its first N raw samples matches the ground truth |
| | RMSD | mean match RMS of the raw samples that match |
| [MLIP-assisted Match Rate](ideas/csp_benchmarking.md#mlip-assisted-match-rate-1) | Recall@N | the same, on the relaxed readouts |
| | ranked Recall@N | the N lowest-ORB-energy relaxed samples out of all 20 |
| | RMSD before/after relaxation | mean match RMS of the raw sample and of its relaxation, over samples where both match |
| | three-way breakdown | per gene, over the first N samples (sym readout): **(a) matched**; else **(b) putative lower polymorph** if the lowest sample energy is below the ground truth's by more than a margin (0 or 10 meV/atom); else **(c) failure**. Against `gt_dft` and `gt_relaxed` |
| | saddle point | the concern raised there: the free readout is lower than the sym readout, in PET energy, by more than 1 meV/atom, so the fixed-symmetry minimum is a saddle point. Reported for samples and for ground truths |
| [Polymorph counting](ideas/csp_benchmarking.md#polymorph-counting-1) | polymorph count | per gene, unique sym readouts with E_hull ≤ 0.1 eV/atom that are not saddle points. Ground-truth matches count as ordinary polymorphs, neither rewarded nor punished; `count_novel` excludes them. Flat budgets N and DoF-dependent budgets (WyckoffTransformer trial schedules `0:1,2:2,*:3` and `0:1,2:3,5:5,8:8,*:10`) |

Uniqueness uses pymatgen's default `StructureMatcher` tolerances (ltol 0.2, stol 0.3,
angle_tol 5), not the DiffCSP ones. At stol 0.5 the DiffCSP protocol matches zincblende to
rocksalt NaCl (rms 0.43), so counting with it would merge distinct polymorphs.
Clustering is greedy in ascending energy. StructureMatcher is not transitive, so that
order is part of the definition.

**Setup.** `DIFFCSP_EXTRAS="orb wandb dev csp-bench" scripts/platforms/zeus/env_init.sh`. The
`csp-bench` extra installs WyckoffTransformer editable from `../WyckoffTransformer` and UPET.
`mlip_relax.py` records both repositories' commits and the versions (or git commits) of the
packages behind the energies. See [the Zeus environment](platforms/zeus/environment.md).

**Caveats, beyond those below.**
- The relaxing potential has seen MP. The "OAM" in PET-OAM-XL is OMat24 + sAlex + MPtrj,
  and MPtrj is MP's own relaxation trajectories, so MP-20 ground truths are close to its
  training data. ORB-v3-omat is trained on OMat24 alone, which is built from Alexandria
  structures; Alexandria overlaps MP heavily.
- E_hull ≤ 0.1 is an ORB statement, and the ORB-vs-DFT error is largest on exactly the
  novel structures polymorph counting rewards.
- A putative lower polymorph is lower than the ground truth in ORB, after PET relaxation.
  Against `gt_dft` the breakdown is biased towards it, because the samples sit at a PET minimum
  and the ground truth at a DFT minimum. `gt_relaxed` removes that bias and is the stricter reference.

## CSP benchmark — CSP (formula given)

Implements the [Classic](ideas/csp_benchmarking.md#classic-match-rate) and
[MLIP-assisted Match Rate](ideas/csp_benchmarking.md#mlip-assisted-match-rate) of
[Crystal Structure Prediction (CSP)](ideas/csp_benchmarking.md#crystal-structure-prediction-csp): the model is
given the formula of each benchmark entry, not its Wyckoff gene. [Polymorph counting](ideas/csp_benchmarking.md#polymorph-counting)
is computed on the flat budgets N ∈ {1, 5, 20} only. Driven by `scripts/run_csp_formula_bench.sh`;
the code and W&B call this the *formula track* (`track=formula`, tag `formula_track`).

```
formula -> WyFormer: sg ~ p(sg | composition), gene ~ constrained decoder -> from_random -> DiffCSP-Geo -> PET relax -> ORB
           (x20 i.i.d. samples per formula)
```

**Tasks.** The 1000 MP-20 benchmark entries of [Reconstructing Wyckoff genes](#csp-benchmark--reconstructing-wyckoff-genes)
(983 distinct formulas), each scored against its own ground truth, so that benchmark is this one
with an oracle gene. The ideas doc *assumes* the test set holds the lowest-energy polymorph. Here,
in 93.4% of entries the ground truth is the lowest-`e_above_hull` MP-20 test polymorph of its
formula; 28.8% of the formulas also occur in MP-20 train+val (as other polymorphs).

**Gene sampling** (`bench/gen_csp_formula.py`). Each sample is an independent draw: the
space group from the backbone's start head, renormalised over the groups that can express
the composition, then the gene from WyckoffTransformer's composition-constrained decoder
(`wyckoff_transformer.csp.ConstrainedDecoder`, `strategy="sample"`, temperature 1, no
calibration) in that group. Duplicates are kept, and the trial order is the sampling
order. Two cell-size settings:

- **z free**: the model chooses the number of formula units, up to 80 atoms in the
  conventional cell (the largest in MP-20 train+val);
- **cell given** (`--given_cell`, regime suffix `-cell`): the decoder may only finish at the
  ground truth's atom count. This is what the DiffCSP CSP protocol gives a model.

A backbone conditioned on formation energy is given the formation energy of the hull at the
formula's composition, built from MP-20 train+val only (elements at zero): "on the hull",
without looking at the test structure.

**Metrics.** Those [above](#csp-benchmark--reconstructing-wyckoff-genes), per formula instead of
per gene, plus, from each sample's gene: Recall@N of the ground truth's space group and of its
Wyckoff gene, and Recall@N with the 20 samples ranked by the model's joint log-probability
log p(sg) + log p(gene | sg) ("likelihood-ranked"). Two genes are equal when one of the space
group's normalizer relabellings (PyXtal's `Group.get_alternatives()`) maps one onto the other
(`csp_metrics.canonical_gene`, since `4b609b5`); `gene_recall_exact_labels` keeps the earlier
comparison of labels as written. DoF-dependent budgets are per gene and do not apply; the
space-group checks compare against the sample's own gene. The ground truth's PET relaxation is shared with the Wyckoff-gene benchmark.

**Note.** `wyformer-csp`'s `build_condition_vector` (WyckoffTransformer `41f741b`) does not
append the null-indicator column that a classifier-free-guidance backbone expects, so it
refuses those models. `gen_csp_formula.py` builds the vector itself for them.

## Results

Archived, not maintained:

- [`archive/mp20-lemat-match-rate-results.md`](archive/mp20-lemat-match-rate-results.md) —
  Result 1 (MP-20 match rates, `match/trial` and `best-of-3`, every DiffCSP regime) and
  Result 2 (LeMat-Bulk, does more data help?), 2026-09-09 … 09-14.
- [`archive/csp-benchmark-results.md`](archive/csp-benchmark-results.md) — the CSP benchmark,
  in the sections of the ideas doc: CSP, formula given (partial as of 2026-10-21, W&B `5mh3v9wl`)
  and Reconstructing Wyckoff genes (MP-20, 2026-10-16 … 18, W&B `zw2t1pj8`, `1t20dz31`).

## Caveats that apply to all results

- **The sets are the round-trippable subsets.** Reaching 1000 representations screened
  2560 candidates on MP-20 and 2304 on LeMat: only ~40% survive
  `from_seed` -> `from_random` at the same site count. Neither is a uniform sample of
  its test split.
- **The ORB backbone has seen this data.** `orb_v3_direct_20_mpa` is trained on
  MPTraj + Alexandria. MP-20 derives from Materials Project and LeMat-Bulk aggregates
  MP + Alexandria + OQMD, so any orb-* number is not a clean generalization
  measurement.
