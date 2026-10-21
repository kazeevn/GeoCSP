# CSP benchmark results on MP-20 (archive)

> **Archived 2026-10-20, updated 2026-10-21, not maintained.** A snapshot of the CSP benchmark
> results. The evaluations are those proposed in
> [`../ideas/csp_benchmarking.md`](../ideas/csp_benchmarking.md); this file follows its sections and
> terms. The protocols as implemented are current and live in [`../benchmark.md`](../benchmark.md)
> ([CSP, formula given](../benchmark.md#csp-benchmark--csp-formula-given),
> [Reconstructing Wyckoff genes](../benchmark.md#csp-benchmark--reconstructing-wyckoff-genes)).
> Later runs, and code changes after `4b609b5`, may supersede what is written here. CSP from the
> formula is complete for the three backbones with the atom count given; the three regimes in
> which the model chooses z were cancelled on 2026-10-21 (see below).

Both evaluations use the 1000 MP-20 benchmark entries of
[Result 1](mp20-lemat-match-rate-results.md) (983 distinct formulas), the PET-OAM-XL relaxation,
ORB `orb_conserv_inf` energies and the LeMat-Bulk-MLIP-Hull revision `70d505bb`. The ground
truths were relaxed once (2026-10-16), and that relaxation serves every regime. Reconstructing
the ground truth's own Wyckoff gene is CSP from the formula with an oracle gene, so the
Wyckoff-gene `diffcsp-geo (500e)` numbers are the reference for the formula-given ones.

Terms, as in the ideas doc:
- The **sym readout** is the fixed-symmetry MLIP relaxation; the **free readout** is what the
  symmetry-released relaxation and rattle kept.
- The **three-way breakdown** is (a) matched / (b) putative lower polymorph / (c) failure.
- Its two references are `gt_dft` (DFT structures, MLIP energies; LeMat-GenBench style) and
  `gt_relaxed` (references relaxed with the MLIP).
- A **saddle point** is a sample whose free readout is more than 1 meV/atom below its sym
  readout, in PET energy.

In the code and W&B, CSP from the formula is the *formula track* (Result 4 in earlier versions
of this file) and Reconstructing Wyckoff genes the *gene track* (Result 3).

## Crystal Structure Prediction (CSP), formula given

The evaluations of [Crystal Structure Prediction (CSP)](../ideas/csp_benchmarking.md#crystal-structure-prediction-csp),
on flat budgets N ∈ {1, 5, 20}. The model gets each entry's formula only: a
composition-conditioned WyFormer draws 20 i.i.d. (space group, Wyckoff gene) samples per formula,
and DiffCSP-Geo (500e) turns each gene's `from_random` draw into a structure. Protocol:
[`../benchmark.md`](../benchmark.md#csp-benchmark--csp-formula-given).

### Setup and provenance

WyFormer backbones, all from the `csp-formula-hera-20261016` campaign ([W&B](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016)),
trained on hera at WyckoffTransformer `b25687e`, MP-20, `composition_conditioning=true`,
`condition_on_cell_size=false`, `predict_start=true`, 1M epochs with the best validation
checkpoint used:

| short name | W&B run (`symmetry-advantage/WyckoffTransformer`) | checkpoint | conditioning |
|---|---|---|---|
| `uncond` | [`csp-formula-hera-20261016-mp20_unconditional`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016-mp20_unconditional) | `best_model…:v6` (epoch 8000) | composition |
| `eform` | [`csp-formula-hera-20261016-mp20_formation_energy`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016-mp20_formation_energy) | `best_model…:v8` (epoch 12000) | composition + formation energy |
| `eform-cfg` | [`csp-formula-hera-20261016-mp20_formation_energy_cfg`](https://wandb.ai/symmetry-advantage/WyckoffTransformer/runs/csp-formula-hera-20261016-mp20_formation_energy_cfg) | `best_model…:v7` (epoch 767000) | the same, with condition dropout 0.1 (sampled at guidance scale 1) |

Regimes are `<model>-cell` (the ground truth's atom count given, the DiffCSP protocol) and
`<model>` (z chosen by the model, cells of at most 80 atoms).

| regime | genes | draws, Geo, relaxation | scoring | DiffCSPNew | WyckoffTransformer | W&B run (`symmetry-advantage/diffcsp`) | status |
|---|---|---|---|---|---|---|---|
| `uncond-cell` | 2026-10-19/20 | 2026-10-20 | 2026-10-21 | `43da227` (genes), `60ba231` (the 94 repaired formulas, draws, Geo, relaxation), `4b609b5` (scoring) | `41f741b` | [`5mh3v9wl`](https://wandb.ai/symmetry-advantage/diffcsp/runs/5mh3v9wl); first scored 2026-10-20 at `60ba231` as [`ptmvqmv2`](https://wandb.ai/symmetry-advantage/diffcsp/runs/ptmvqmv2), with exact-label gene matching | complete |
| `eform-cell` | 2026-10-19/20 | 2026-10-20 | 2026-10-21 | `43da227` (genes), `60ba231` (repair, draws, Geo, relaxation), `028ea24` (scoring; evaluation code as in `4b609b5`) | `41f741b` | [`zu75p9yz`](https://wandb.ai/symmetry-advantage/diffcsp/runs/zu75p9yz) | complete |
| `eform-cfg-cell` | 2026-10-19/20 | 2026-10-21 | 2026-10-21 | `43da227` (genes), `60ba231` (repair), `028ea24` (draws, Geo, relaxation, scoring) | `41f741b`; `db7cefe` for the relaxation | [`xrc3zzwv`](https://wandb.ai/symmetry-advantage/diffcsp/runs/xrc3zzwv) | complete |
| `uncond`, `eform-cfg`, `eform` (z chosen by the model) | 2026-10-19/20 | — | — | `1deb6fb-dirty` (≈`43da227`), repaired at `60ba231` | same | — | cancelled 2026-10-21; gene files kept |

Files are in `runs/bench/mp20/formula/`; `orchestrate_20261020.sh` is the driver invocation, and
`csp_geo_uncond-cell_t20.rescore-geneq.pkl` the 2026-10-21 scoring. Each gene file's metadata
records its repair. The first sampling pass rejected 94 of the 1000 formulas: pymatgen's reduced
formulas such as `Li(TiS2)3` don't parse in WyckoffTransformer. `60ba231` re-sampled exactly those
formulas. Seeds are per entry, so the files equal a full rerun at `60ba231`. Note that
`gen_csp_formula.py` reads the git commit when it finishes, so the gene files of the first pass
record `60ba231` though they ran the code of `43da227`. The two differ only in the formula string
and in `--repair`.

The z-free regimes were cancelled once the analyses below showed the bottleneck is WyFormer's
sampler and space-group head, which they share: a two-day relaxation of genes from the same sampler
would not have answered a new question. Their gene files stay in `runs/bench/mp20/formula/`.
WyckoffTransformer `main` moved from `41f741b` to `db7cefe` (documentation and a merged
Hugging Face inference branch) while `eform-cfg-cell` waited to be relaxed; its relaxation records
the later commit.

Samples per regime (1000 formulas × 20, failed `from_random` draws count as misses): `uncond-cell`
19,923, `eform-cell` 19,938, `eform-cfg-cell` 19,913. Per formula, the 20 samples hold 9.3, 9.4 and
9.7 distinct genes (up to equivalent relabellings) in 3.3–3.4 space groups, median 12 atoms.

### Classic Match Rate

[Classic Match Rate](../ideas/csp_benchmarking.md#classic-match-rate): Recall@N of the raw samples.
Ranking by WyFormer's likelihood is not in the ideas doc; it orders the 20 samples by
log p(sg) + log p(gene | sg).

| Recall@1 / 5 / 20 (%) | `uncond-cell` | `eform-cell` | `eform-cfg-cell` | oracle gene (R@1 / R@20) |
|---|---|---|---|---|
| raw samples, sampling order | 24.7 / 49.5 / 66.4 | **29.0 / 53.9 / 69.7** | 23.2 / 50.5 / 66.6 | 83.3 / 93.4 |
| raw samples, ranked by WyFormer likelihood | 34.7 / 48.1 / 66.4 | **40.0** / 53.9 / 69.7 | 35.3 / 48.9 / 66.6 | |

Mean match RMS of the raw samples that match: 0.115, 0.111, 0.114.

### MLIP-assisted Match Rate

[MLIP-assisted Match Rate](../ideas/csp_benchmarking.md#mlip-assisted-match-rate): Recall@N of the
relaxed readouts, unranked and ranked by ORB energy; RMSD before and after relaxation; the
three-way breakdown.

| Recall@1 / 5 / 20 (%) | `uncond-cell` | `eform-cell` | `eform-cfg-cell` | oracle gene (R@1 / R@20) |
|---|---|---|---|---|
| sym readout | 25.7 / 50.5 / 66.5 | 29.7 / 54.6 / 68.8 | 23.2 / 51.2 / 67.4 | 84.0 / 94.0 |
| sym readout, ranked by ORB energy | 52.8 / 60.7 / 66.5 | **54.8 / 62.2 / 68.8** | 51.6 / 58.6 / 67.4 | **88.2** / 94.0 |
| sym readout, ranked by WyFormer likelihood | 36.2 / 49.1 / 66.5 | 39.8 / 53.1 / 68.8 | 35.3 / 48.4 / 67.4 | |
| free readout | 29.8 / 55.0 / 73.3 | 32.3 / 59.7 / 74.6 | 25.6 / 55.5 / 73.3 | 84.0 / 94.0 |
| free readout, ranked by ORB energy | 53.0 / 65.0 / 73.3 | 54.0 / 65.4 / **74.6** | 52.4 / 62.8 / 73.3 | 88.0 / 94.0 |
| free readout, ranked by WyFormer likelihood | 37.8 / 53.1 / 73.3 | 42.1 / 57.9 / 74.6 | 37.7 / 52.1 / 73.3 | |

RMSD before and after MLIP relaxation, over samples that match both before and after: sym
readout 0.097 → 0.090, 0.093 → 0.085, 0.093 → 0.087 (4,821, 5,612, 4,608 pairs); free readout
0.091 → 0.061, 0.086 → 0.056, 0.085 → 0.061 (`uncond-cell`, `eform-cell`, `eform-cfg-cell`).

- **Conditioning on the training-hull formation energy helps a little.** `eform-cell` leads
  `uncond-cell` by 2–5 points on every row. The guidance-trained `eform-cfg-cell`, sampled at
  guidance scale 1 (plain conditional), is no better than `uncond-cell`; higher scales were not tried.

- **Finding the gene is the bottleneck, not reconstructing it.** With the gene given, Geo reaches
  88.2% energy-ranked R@1; from the formula, 52.8%. At R@20, structure recall (66.5%) tracks
  the recall of the ground truth's space group (68.9%, [below](#wyckoff-gene-recovery)).
- **Ranking matters more than relaxing.** Relaxation adds about 1 point at fixed N. Ranking the 20
  relaxed samples by ORB energy doubles R@1 (25.7 → 52.8); ranking by WyFormer's own
  likelihood gives about a third of that gain (→ 36.2).
- **The free readout finds structures outside the sampled gene.** At R@20 it recovers 73.3% against
  the sym readout's 66.5%: 55% of free readouts leave the gene's space group, and some of them
  relax into the ground truth. The same holds for the other two backbones: 53–57% of free readouts
  leave the gene's space group.

Three-way breakdown, % of formulas, sym readout, N = 1 | 5 | 20:

| reference, margin | `uncond-cell` | `eform-cell` | `eform-cfg-cell` |
|---|---|---|---|
| `gt_relaxed`, 0 | 25.7/3.9/70.4 \| 50.5/7.3/42.2 \| 66.5/9.5/24.0 | 29.7/4.5/65.8 \| 54.6/8.5/36.9 \| 68.8/9.2/22.0 | 23.2/3.8/73.0 \| 51.2/7.5/41.3 \| 67.4/7.8/24.8 |
| `gt_relaxed`, 10 meV/atom | 25.7/2.6/71.7 \| 50.5/4.3/45.2 \| 66.5/6.0/27.5 | 29.7/2.8/67.5 \| 54.6/4.9/40.5 \| 68.8/5.8/25.4 | 23.2/2.3/74.5 \| 51.2/4.6/44.2 \| 67.4/5.4/27.2 |
| `gt_dft`, 10 meV/atom | 25.7/2.9/71.4 \| 50.5/4.8/44.7 \| 66.5/6.8/26.7 | 29.7/3.2/67.1 \| 54.6/5.4/40.0 \| 68.8/6.4/24.8 | 23.2/2.6/74.2 \| 51.2/4.8/44.0 \| 67.4/5.7/26.9 |

For 5.4–6.0% of formulas, a sample is a putative lower polymorph, at least 10 meV/atom below the
PET-relaxed ground truth in ORB, against 2.2% for the oracle gene. These are candidates for
inspection, and they inherit the caveats of the Wyckoff-gene benchmark. Samples of a wrong gene
are also more often saddle points (below), and a saddle point's sym energy is not a minimum.

Saddle points and metastability:

| | ground truth | `uncond-cell` | `eform-cell` | `eform-cfg-cell` | oracle-gene Geo samples |
|---|---|---|---|---|---|
| sym readout E_hull ≤ 0.1 eV/atom | 95.3% | 34.1% | 36.7% | 32.7% | 88.4% |
| sym readout E_hull, median (eV/atom) | 0.010 | 0.249 | 0.219 | 0.259 | 0.015 |
| raw sample E_hull ≤ 0.1 eV/atom | 95.2% | 21.9% | 24.7% | 21.2% | 69.4% |
| saddle point | 6.0% | 52.6% | 50.4% | 52.9% | 9.1% |
| saddle point, among metastable sym readouts | 5.8% | 10.7% | 11.2% | 9.8% | 7.1% |
| sym readout left the gene's space group (spglib, symprec 0.1) | | 17.5% | 17.2% | 18.8% | 4.2% |

Most samples are saddle points of their gene under PET-OAM-XL, but the metastable ones mostly
are not.

### Polymorph counting

[Polymorph counting](../ideas/csp_benchmarking.md#polymorph-counting) per formula, flat budgets:
unique sym readouts with E_hull ≤ 0.1 eV/atom that are not saddle points. The formulas come from
the test set, and the model does not choose its own budget.

Each cell is count / ≥1 / novel / ≥1 novel: the mean number of distinct polymorphs per formula,
the share of formulas with at least one, and the same two for polymorphs that do not match the
ground truth.

| budget | `uncond-cell` | `eform-cell` | `eform-cfg-cell` |
|---|---|---|---|
| N = 1 | 0.31 / 30.7% / 0.12 / 12.1% | 0.33 / 33.0% / 0.11 / 11.1% | 0.29 / 29.4% / 0.12 / 12.5% |
| N = 5 | 0.84 / 60.6% / 0.43 / 28.1% | 0.90 / 65.7% / 0.44 / 28.9% | 0.84 / 59.4% / 0.44 / 28.9% |
| N = 20 | 1.98 / 77.5% / 1.35 / 43.8% | 1.92 / 80.3% / 1.27 / 44.7% | 1.89 / 77.0% / 1.29 / 44.0% |

"novel" means not matching the ground truth, as for Wyckoff genes below. It is not novelty
against MP. With 20 samples, the formula-given samples hold more non-ground-truth metastable
polymorphs per formula than the oracle gene's do per gene (1.27–1.35 vs 0.95): they spread over
several genes.

### Wyckoff gene recovery

Not in the ideas doc: how often the samples' Wyckoff genes recover the ground truth's, in sampling
order. Genes are compared up to the space group's normalizer relabellings (`csp_metrics.canonical_gene`,
since `4b609b5`); the exact-label comparison used at first misses equivalent labellings.

| Recall@1 / 5 / 20 (%) | `uncond-cell` | `eform-cell` | `eform-cfg-cell` |
|---|---|---|---|
| ground truth's space group | 42.0 / 58.0 / 68.9 | 43.3 / 60.5 / 71.1 | 40.6 / 59.1 / 69.3 |
| ground truth's Wyckoff gene | 20.1 / 39.1 / 52.4 | 23.8 / 43.5 / 56.7 | 18.6 / 38.2 / 52.1 |
| … compared on exact labels | 13.7 / 31.8 / 46.3 | 14.9 / 34.7 / 49.3 | 11.7 / 29.7 / 45.8 |

For `uncond-cell`: per sample, 20.5% of the genes are the ground truth's (13.6% on exact labels).
Of the 5,157 raw samples that match the ground-truth structure, 67.7% carry its gene (41.8% on exact labels) and
84.2% its space group. The remaining 16.5% match from the right space group but a different gene;
why has not been analysed. StructureMatcher at the DiffCSP tolerances (stol 0.5) is loose enough to
match distinct structures.

### The formation-energy condition

`eform` and `eform-cfg` are given the formation energy of the MP-20 train+val hull at the
formula's composition (`hull_formation_energy` in `bench/gen_csp_formula.py`). Against each
ground truth's own MP-20 `formation_energy_per_atom`, over the 1000 tasks:

| condition − ground-truth E_form | |
|---|---|
| median / mean | +0.006 / +0.046 eV/atom |
| 10th / 90th percentile | −0.051 / +0.136 eV/atom |
| within ±0.05 eV/atom | 60.6% |
| ground truth below the training hull | 54.3% |
| condition = 0 (no compound of that chemical system in train+val) | 5.1% |

The training hull misses competing phases that the full MP hull has, so for about half the formulas
the condition sits above the ground truth's energy, by up to about 0.1–0.15 eV/atom.
Untried alternatives: the ground truth's own formation energy (a leaky oracle and upper
bound), or a value below the hull.

### Cost

Relaxation cost is set by cell size. In a 10-formula pilot of `uncond` (z chosen by the model,
2026-10-19), cells of 60–80 atoms were 41 of 168 samples and took 84% of the PET time:
median 85 s each against 2 s below 12 atoms. In the full z-free gene files, 40–48% of
samples have more than 40 atoms. `uncond-cell` took about 13.5 h end to end on two RTX 6000 Ada:
about 3.5 h of Geo on one GPU, and 6 relaxation processes at a median 6.7 s per structure.

## Reconstructing Wyckoff genes

The evaluations of [Reconstructing Wyckoff genes](../ideas/csp_benchmarking.md#reconstructing-wyckoff-genes):
the model is given the ground truth's Wyckoff gene. Protocol:
[`../benchmark.md`](../benchmark.md#csp-benchmark--reconstructing-wyckoff-genes).

### Setup and provenance

The same 1000 MP-20 genes as Result 1, × 20 `from_random` draws = 19,997 samples per regime
(`runs/bench/mp20/inits_t20.pkl`, generated 2026-10-16; 3 draws failed).

| regime | samples | relaxed | scored | DiffCSPNew | WyckoffTransformer | W&B run (`symmetry-advantage/diffcsp`) |
|---|---|---|---|---|---|---|
| `pyxtal-only (control)` | `run_none.py` | 2026-10-16/17 | 2026-10-18 | `525c232` + then-uncommitted code, committed as `33e3224` | `a3e8512` | [`zw2t1pj8`](https://wandb.ai/symmetry-advantage/diffcsp/runs/zw2t1pj8) |
| `diffcsp-geo (500e)` | `runs/mp20_geo/geo_mp20_500e_best.pt` ([`7jdhrv2f`](https://wandb.ai/symmetry-advantage/diffcsp/runs/7jdhrv2f), ep 320) | 2026-10-17 | 2026-10-18 | same | same | [`1t20dz31`](https://wandb.ai/symmetry-advantage/diffcsp/runs/1t20dz31) |

The run's benchmark code is the code in `33e3224`, except for two changes made after it: `mlip_relax.py`'s package-provenance metadata and the removal of the overlay handling. The ground truths were relaxed once (2026-10-16) and serve both regimes. Environment: the
PYTHONPATH overlay described in the Zeus environment's *History* paragraph. Its ORB energies
match the current locked setup to within 4e-5 eV/atom. Four relaxation processes shared one
RTX 6000 Ada; median time per structure was 10.0 s for the control and 3.5 s for Geo.

It was scored by `evaluate_csp.py` as of `33e3224`, where "the first N samples" were the first N
*surviving* samples. Since `43da227` they are trials < N, so a failed draw is a miss. With 3 failed
draws out of 19,997 samples per regime, the numbers are unaffected at the precision given.

### Classic Match Rate

[Classic Match Rate](../ideas/csp_benchmarking.md#classic-match-rate-1): Recall@N of the raw samples.

| Recall@N (%) | control R@1 | R@5 | R@20 | Geo R@1 | R@5 | R@20 |
|---|---|---|---|---|---|---|
| raw samples | 22.9 | 52.1 | 71.9 | 83.3 | 89.5 | 93.4 |

### MLIP-assisted Match Rate

[MLIP-assisted Match Rate](../ideas/csp_benchmarking.md#mlip-assisted-match-rate-1): Recall@N of the
relaxed readouts, unranked and ranked by ORB energy; RMSD before and after relaxation; the
three-way breakdown; and saddle points, the concern raised there.

| Recall@N (%) | control R@1 | R@5 | R@20 | Geo R@1 | R@5 | R@20 |
|---|---|---|---|---|---|---|
| sym readout | 54.7 | 79.7 | 89.2 | 84.0 | 90.8 | 94.0 |
| sym readout, ranked by ORB energy | **79.8** | 85.4 | 89.2 | **88.2** | 90.6 | 94.0 |
| free readout | 54.3 | 79.8 | 89.6 | 84.0 | 90.7 | 94.0 |
| free readout, ranked | 76.4 | 84.0 | 89.6 | 88.0 | 90.2 | 94.0 |

RMSD before and after MLIP relaxation, the mean match RMS over samples that match both before
and after (sym readout): control 0.158 → 0.029 (4,179 pairs); Geo 0.033 → 0.022 (16,349 pairs).

- **Relaxation does the control's job, not Geo's.** PET relaxation lifts the unrefined
  draws from 22.9% to 54.7% R@1 but moves Geo by only 0.7 points: Geo's samples are already
  in the right basin, and relaxation only polishes them (RMS 0.033 → 0.022).
- **Ranking by energy is worth more than relaxing, for Geo.** Picking the lowest-energy of 20
  relaxed samples gives 88.2% R@1, against 84.0% for the first sample. For relaxed random
  draws ranking is the main effect (54.7% → 79.8%).
- **Geo's margin shrinks but survives**: 88.2 vs 79.8 at ranked R@1, 94.0 vs 89.2 at R@20,
  with 20× the relaxation budget given to both.

Three-way breakdown, % of genes, sym readout:

| reference, margin | control N=1 | N=20 | Geo N=1 | N=20 |
|---|---|---|---|---|
| `gt_relaxed`, 0 | 54.7 / 1.3 / 44.0 | 89.2 / 2.1 / 8.7 | 84.0 / 3.7 / 12.3 | 94.0 / 3.5 / 2.5 |
| `gt_relaxed`, 10 meV/atom | 54.7 / 0.7 / 44.6 | 89.2 / 1.1 / 9.7 | 84.0 / 1.4 / 14.6 | 94.0 / 2.2 / 3.8 |
| `gt_dft`, 10 meV/atom | 54.7 / 1.0 / 44.3 | 89.2 / 1.2 / 9.6 | 84.0 / 1.7 / 14.3 | 94.0 / 2.5 / 3.5 |

With 20 samples, Geo finds a putative lower polymorph, at least 10 meV/atom below the ground
truth (in ORB, after PET relaxation), for 2.2% of genes. These are candidates for inspection, not
discoveries: see the caveats in [`../benchmark.md`](../benchmark.md#csp-benchmark--reconstructing-wyckoff-genes).

Saddle points and metastability:

| | ground truth | control samples | Geo samples |
|---|---|---|---|
| sym readout E_hull ≤ 0.1 eV/atom | 95.3% | 52.8% | 88.4% |
| sym readout E_hull, median (eV/atom) | 0.010 | 0.082 | 0.015 |
| raw sample E_hull ≤ 0.1 eV/atom | 95.2% (DFT structure) | 2.3% | 69.4% |
| saddle point | 6.0% | 30.3% | 9.1% |
| saddle point, among metastable sym readouts | 5.8% | 9.0% | 7.1% |

**6% of MP-20 ground truths are saddle points under PET-OAM-XL.** This answers, for this
potential, the ideas doc's open question of whether the DFT-relaxed structures in the training
database are also saddle points. When a metastable sym readout is a saddle point, the cause may be
the potential rather than the structure; ground truths show the rate is non-negligible even for
DFT-relaxed structures. The PET-relaxed ground truth still matches the DFT structure in 99.3% of
genes.

### Polymorph counting

[Polymorph counting](../ideas/csp_benchmarking.md#polymorph-counting-1): unique sym readouts per
gene with E_hull ≤ 0.1 eV/atom that are not saddle points, on flat budgets N and on DoF-dependent
budgets. Each cell is count / ≥1 / novel / ≥1 novel.

| budget | control | Geo |
|---|---|---|
| N = 1 | 0.49 / 48.7% / 0.04 / 4.3% | 0.83 / 82.9% / 0.09 / 8.8% |
| N = 20 | 1.59 / 86.1% / 0.71 / 25.4% | 1.96 / 92.8% / 0.95 / 18.7% |
| DoF-dependent `0:1,2:2,*:3` (2.43 samples/gene) | 0.71 / 64.8% / 0.13 / 10.1% | 1.04 / 89.0% / 0.22 / 13.9% |
| DoF-dependent `0:1,2:3,5:5,8:8,*:10` (5.65) | 1.11 / 77.7% / 0.39 / 19.4% | 1.48 / 91.7% / 0.56 / 17.6% |

"novel" means a cluster whose representative does not match the ground truth, even under the
loose DiffCSP tolerances. It is **not** novelty against MP or any other database, which this
benchmark does not check. At N = 20, Geo yields more distinct non-ground-truth polymorphs
per gene (0.95 vs 0.71), but random draws reach at least one in more genes (25.4% vs 18.7%):
Geo concentrates on fewer genes.

### Notes on this run

- **Failures.** All 9 failed relaxations (control only, 0.045%) are raw draws whose atoms
  overlap: e.g. 1.65 Å³/atom with a = 1.97 Å. PET's neighbour list (metatomic-ase via
  nvalchemiops) stops at 1000 neighbours per atom and raises `NeighborOverflowError`. The
  same happens in the current environment. They count as misses.
- **Relaxations are not bit-reproducible on GPU.** Six smoke-run structures, re-relaxed in
  this run with identical inputs and seeds, differ by 0.1–0.6 meV/atom in PET energy, and one
  landed in a different basin (E_hull 0.017 vs 0.084). Where the rattle ends up can differ by
  hundreds of meV/atom. Aggregate metrics are unaffected in expectation; single-sample
  outcomes are not reproducible.
- **The sym readout does not always keep the gene's space group by spglib's reckoning**
  (symprec 0.1): 10.9% of control and 4.2% of Geo sym readouts report a different
  space-group number. FixSymmetry should preserve the gene's symmetry, so these may be
  detections of a higher symmetry, but they have not been analysed.
