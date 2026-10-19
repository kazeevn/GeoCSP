# Crystal Structure Prediction (CSP)
CSP problem can be formulated in two different ways:
1. Given the chemical formula, find the crystal structure with this chemical composition with the globally minimal energy.
2. Given the chemical formula, find all the metastable (E_hull <=0.1 eV/atom) structures.

Proposed evaluations follow.

## Classic Match Rate
*Assume* that the test dataset contains the lowest-energy polymorph. Then, sample N structures from the model, and check whether they contain the lowest-energy polymorph from the test dataset. Use N=1, 5, 20. Use DiffCSP structure matching protocol. Compute RMSD for matches.

Pros:
- Literature uses it, direct baseline results are available
Cons:
- Misses the generated structures with energy lower than the test dataset
- Combines global and local minima search. Finding the precise location of a local minimum is not very important, as it can be later approximated with a MLIP - while finding a different basin is the hard part

## MLIP-assisted Match Rate
*Assume* that the test dataset contains the lowest-energy polymorph. Then, sample 20 structures from the model, relax them with a MLIP and check whether they contain the lowest-energy polymorph from the test dataset - Recall@N for N=1,5,20. Use DiffCSP structure matching protocol. Compute RMSD for matches before and after MLIP relaxation. Use MLIP energies to rank the generated structures, check Recall@N for the ranked structures.

Pros:
- Separates global and local minima search
- MLIP energies allow to detect if the model has found a new lowest-energy polymorph; we can measure a three-way breakdown: (a) matched GT, (b) unmatched but E_{MLIP} < E_{GT} (putative lower polymorph), and (c) unmatched & higher energy (failure)
Cons:
- MLIP introduces systematic uncertainty
- Reference dataset treatment requires trade-offs:
  - DFT structures and DFT energies make us fully exposed to the DFT-vs-MLIP systematic uncertainty
  - DFT structures and MLIP energies (LeMat-Genbench style) partially cancel the DFT-vs-MLIP energy shift.
    - Comparing MLIP energies of MLIP-relaxed structures with MLIP energies of DFT relaxed structures means the E_hull will be underestimated as the new structures lie at exact MLIP local minima
    - Comparing MLIP energies of unrelaxed generated structures with MLIP energies of DFT relaxed structures doesn't encounter the specific bias direction, but still has systematic uncertainty as it will reward the generator for guessing the MLIP-vs-DFT shifts
  - Relaxing references with MLIP will completely tether us to the MLIP and its systematic uncertainty

Considerations:
 - Use different MLIPs for relaxation and energy scoring

## Polymorph counting
Don't use a test dataset. Sample N structures from the model, relax them with a MLIP and count the number of unique polymorphs with E_hull <= 0.1. Neither reward nor punish production of the structures from the reference dataset.

Considerations:
- Where does number of samples per chemical composition come from?
  - Small reasonable flat budget, N ∈ {1, 5, 20, 100}
  - Model chooses the number of structures to sample, can refuse if there are no metastable polymorphs. This requires a careful reward structure: refusing to sample a non-viable formula should be expected, but at the same time less rewarding than correctly sampling a viable formula.
- Where do the chemical formulas come from?
  - Test dataset
  - De novo generation

Pros:
- A very physically meaningful metric
Cons:
- MLIP introduces a systematic uncertainty, it's likely to perform worse on truly novel structures (shared with all MLIPS)
- StructureMatcher is not strictly speaking transitive
- StructureMatcher novelty is strict - it will count structures which belong to the same experimental material.

Variant: count novelty with [templates](/home/kna/WyckoffTransformer/research_notebooks/check_template_novelty.ipynb) or [disordered parents](https://github.com/kazeevn/OpenDisorder/).

# Reconstructing Wyckoff genes
> **Implemented** (2026-10-16): `bench/mlip_relax.py` and `bench/evaluate_csp.py`, with PET-OAM-XL relaxation and ORB `orb_conserv_inf` energies / LeMat-Bulk E_hull. The current protocol and its choices are in [`../benchmark.md`](../benchmark.md#csp-benchmark--wyckoff-gene-track); first results (2026-10-18) in Result 3 there. The formula-only CSP track above is implemented for its classic and MLIP-assisted match rates (2026-10-19, `bench/gen_csp_formula.py`, WyFormer genes -> DiffCSP-Geo; [`../benchmark.md`](../benchmark.md#csp-benchmark--formula-track)); formula-level polymorph counting with a model-chosen budget is still an idea.

a. Given a Wyckoff gene, find the lowest-energy structure
b. Given a Wyckoff gene, find all the metastable polymorphs conforming to the gene

## Classic Match Rate
*Assume* that the test dataset contains the lowest-energy polymorph. Then, sample N structures from the model, and check whether they contain the lowest-energy polymorph from the test dataset. Use N=1, 5, 20. Use DiffCSP structure matching protocol. Compute RMSD for matches.

Pros:
- Literature uses it, direct baseline results are available
Cons:
- Misses the generated structures with energy lower than the test dataset
- Combines global and local minima search. Finding the precise location of a local minimum is not very important, as it can be later approximated with a MLIP - while finding a different basin is the hard part
- If a MLIP is used inside the gene->structure operation, it opens the issues of leakage, as the MLIP has likely been trained on the test dataset

## MLIP-assisted Match Rate
*Assume* that the test dataset contains the lowest-energy polymorph. Then, sample 20 structures from the model, relax them with a fixed-symmetry MLIP and check whether they contain the lowest-energy polymorph from the test dataset - Recall@N for N=1,5,20. Use DiffCSP structure matching protocol. Compute RMSD for matches before and after MLIP relaxation. Use MLIP energies to rank the generated structures, check Recall@N for the ranked structures.

Same Pros and Cons as in CSP, plus:

Concerns:
- A fixed-symmetry MLIP relaxation might lead to a saddle point. How do we treat a saddle point with E_hull < 0.1? A saddle-point structure is not going to exist in the real world. But on the other hand, the structure is going to relax into a lower-energy structure, just with a different Wyckoff gene. Practically speaking, we can use our existing rattle protocol to check for this, or MLIP phonons. The issues are that we don't test the non-fixed-symmetry relaxations for saddle points; we didn't test the training DB, what if the DFT-relaxed structures in it are also saddle points?

## Polymorph counting
Don't use a test dataset. Sample N structures from the model, relax them with a fixed-symmetry MLIP and count the number of unique polymorphs with E_hull <= 0.1. Neither reward nor punish production of the structures from the reference dataset.

Considerations:
- Where does number of samples per gene come from?
  - Small reasonable DoF-dependent budget, N ∈ [1, 10]
  - Model chooses the number of structures to sample, can refuse if there are no metastable polymorphs. This requires a careful reward structure: refusing to sample a non-viable gene should be expected, but at the same time less rewarding than correctly sampling a viable gene. Practically speaking, viability can be reasonably checked by Wyckoff->energy regressor.
- Where do the genes come from?
  - Test dataset
  - De novo generation

Pros:
- A physically meaningful metric
Cons:
- MLIP introduces a systematic uncertainty, it's likely to perform worse on truly novel structures (shared with all MLIPS)
- StructureMatcher is not strictly speaking transitive
- StructureMatcher novelty is strict - it will count structures which belong to the same experimental material.
- Not as meaningful as in the CSP setting, as we don't expect many genuine metastable polymorphs to share the Wyckoff representation

Variant: count novelty with [disordered parents](https://github.com/kazeevn/OpenDisorder/).
