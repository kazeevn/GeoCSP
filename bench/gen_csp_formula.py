"""CSP benchmark, formula track, stage 1: WyFormer samples Wyckoff genes for each formula.

The formula track gives a model only the composition of each benchmark entry. A
composition-conditioned WyFormer backbone proposes Wyckoff genes, and the rest of the
pipeline -- from_random draw, a gene->structure regime, PET relaxation, ORB scoring --
is the gene track's (docs/benchmark.md, "CSP benchmark").

Each of a formula's samples is an independent ancestral draw from the model:

1. the space group, from the backbone's start head p(sg | composition), restricted to
   the groups that can express the composition in at most --max_atoms atoms;
2. the gene, from WyckoffTransformer's composition-constrained decoder in that group,
   with the cell size (z) left to the model, as `wyformer-csp` does.

Duplicates are kept: a gene drawn twice gets two from_random draws, as the gene track
gives every gene several. The trial order is the sampling order; each sample carries
its log-probabilities, so evaluate_csp.py can also rank the samples by likelihood.

A backbone conditioned on formation energy is given the formation energy of the hull at
the formula's composition, built from the training and validation splits only
(`--condition hull`): the "on the hull" target of WyckoffTransformer's docs/csp_mode.md,
mode (b), without looking at the test structure.

    uv run python bench/gen_csp_formula.py \\
        --wandb_run csp-formula-hera-20261016-mp20_unconditional \\
        --out runs/bench/mp20/formula/genes_wyformer-uncond.pkl
"""
import argparse
import datetime
import itertools
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlip_relax import git_commit  # noqa: E402

warnings.filterwarnings("ignore")


def training_hull(dataset: str, splits) -> dict:
    """Formation energies per atom by chemical system, from the WyFormer dataset cache."""
    from pymatgen.core import Composition
    from wyckoff_transformer.dataset_cache import dataset_cache_dir, load_split

    by_system = defaultdict(list)
    for split in splits:
        df = load_split(dataset_cache_dir(dataset), split, columns=["composition", "formation_energy_per_atom"])
        for comp, e in zip(df["composition"], df["formation_energy_per_atom"]):
            c = Composition({str(k): v for k, v in comp.items()})
            by_system[frozenset(str(el) for el in c.elements)].append((c, float(e)))
    return by_system


def hull_formation_energy(formula: str, by_system: dict) -> float:
    """Formation energy per atom of the training hull at *formula*'s composition."""
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition

    comp = Composition(formula)
    els = [str(e) for e in comp.elements]
    # Formation energies put every element at zero.
    entries = [PDEntry(Composition(e), 0.0) for e in els]
    for k in range(1, len(els) + 1):
        for system in itertools.combinations(els, k):
            entries += [PDEntry(c, e * c.num_atoms) for c, e in by_system.get(frozenset(system), [])]
    return float(PhaseDiagram(entries).get_hull_energy_per_atom(comp))


class GeneSampler:
    """Space group from the start head, then composition-constrained decoding in it."""

    def __init__(self, trainer, device, max_atoms: int, temperature: float):
        from wyckoff_transformer.csp import ConstrainedDecoder

        self.t, self.device, self.max_atoms, self.temperature = trainer, device, max_atoms, temperature
        if not getattr(trainer, "predict_start", False):
            raise ValueError("the backbone has no start head (predict_start) to sample the space group from")
        if trainer.condition_on_cell_size:
            raise ValueError("a backbone conditioned on the cell size cannot choose z in one pass")
        # Class i of the start head is the i-th space group in ascending order.
        self.space_groups = sorted(trainer.tokenisers[trainer.start_name])
        self.decoder = ConstrainedDecoder(
            model=trainer.model, cascade_order=trainer.cascade_order, cascade_is_target=trainer.cascade_is_target,
            tokenisers=trainer.tokenisers, token_engineers=trainer.token_engineers, masks=trainer.masks_dict,
            stops=trainer.stops_dict, max_sequence_len=trainer.max_sequence_length, device=device,
            model_dir=trainer.run_path)

    def _feasible(self, ratio) -> dict:
        """Space group -> cell sizes it can build *ratio* at, within max_atoms."""
        from wyckoff_transformer.cli.csp import start_tensor_for

        unit = sum(ratio.unit_counts)
        out = {}
        for sg in self.space_groups:
            start = start_tensor_for(self.t, sg, self.device)
            zs = ratio.feasible_z(self.decoder.combinatorics(sg, start), self.t.max_sequence_length,
                                  self.max_atoms // unit)
            if zs:
                out[sg] = zs
        return out

    @torch.no_grad()
    def sample(self, formula: str, n: int, condition_values, seed: int, max_attempts: int = 5,
               n_atoms: int | None = None) -> list:
        """*n* genes for *formula*, in sampling order; None where decoding failed.

        *n_atoms* fixes the cell content, as the DiffCSP CSP protocol gives it; by
        default the model chooses z.
        """
        from wyckoff_transformer.cli.csp import build_condition_vector, start_tensor_for
        from wyckoff_transformer.csp import CompositionRatio, candidates_to_pyxtal

        t = self.t
        ratio = CompositionRatio.for_formula(formula, t.tokenisers["elements"])
        scalar = None
        if condition_values is not None:
            scalar = t.transform_condition(t.build_condition_from_values(condition_values, 1, device=self.device))
        if t.classifier_free_guidance:
            # wyformer-csp's build_condition_vector (WyckoffTransformer 41f741b) omits the
            # null-indicator column a guidance-trained model expects; the trainer appends it.
            from wyckoff_transformer.composition import composition_vector

            comp = composition_vector(ratio.element_tokens, ratio.counts_at(1), t.n_elements, device=self.device,
                                      condition_on_cell_size=False).unsqueeze(0)
            cond = t.with_null_indicator(torch.cat([p for p in (scalar, comp) if p is not None], dim=-1))
        else:
            cond = build_condition_vector(t, ratio, None, scalar, self.device)
        feasible = self._feasible(ratio)
        if n_atoms is not None:
            z = n_atoms // sum(ratio.unit_counts)
            feasible = {sg: [z] for sg, zs in feasible.items() if z in zs}
        if not feasible:
            return [None] * n

        log_p = torch.log_softmax(t.model.forward_start(1, cond=cond).float()[0] / self.temperature, -1).cpu()
        mask = torch.tensor([sg in feasible for sg in self.space_groups])
        log_p = torch.where(mask, log_p, torch.tensor(-np.inf))
        mass = float(log_p.logsumexp(0).exp())
        log_p = log_p - log_p.logsumexp(0)

        cpu = torch.Generator().manual_seed(seed)
        dev = torch.Generator(device=self.device).manual_seed(seed)
        sg_of_trial = [self.space_groups[i] for i in torch.multinomial(log_p.exp(), n, replacement=True, generator=cpu)]
        genes = [None] * n
        for sg in sorted(set(sg_of_trial)):
            trials = [i for i, s in enumerate(sg_of_trial) if s == sg]
            start = start_tensor_for(t, sg, self.device)
            comb = self.decoder.combinatorics(sg, start)
            got = []
            # A path can die at the site cap without completing the formula; draw again.
            for _ in range(max_attempts):
                need = len(trials) - len(got)
                if need <= 0:
                    break
                cands = self.decoder.decode(
                    start=start, sg_number=sg, ratio=ratio, allowed_z=feasible[sg], max_z=max(feasible[sg]),
                    n_candidates=need, strategy="sample", temperature=self.temperature, cond=cond, generator=dev)
                got += [(c, r) for c, r in zip(cands, candidates_to_pyxtal(cands, t, start, comb)) if r is not None]
            # decode() returns its draws most likely first; restore an exchangeable order.
            got = [got[j] for j in torch.randperm(len(got), generator=cpu).tolist()]
            for trial, (c, rep) in zip(trials, got):
                genes[trial] = {
                    "rep": {k: rep[k] for k in ("group", "species", "numIons", "sites")},
                    "sg": sg, "z": int(c.z), "n_sites": int(c.n_sites), "n_atoms": int(sum(rep["numIons"])),
                    "log_p_sg": float(log_p[self.space_groups.index(sg)]), "log_p_gene": float(c.log_prob),
                    "feasible_sg_mass": mass,
                }
        return genes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--benchset", default="runs/bench/mp20/benchset.pkl")
    ap.add_argument("--wandb_run", required=True, help="WyckoffTransformer W&B run holding the backbone")
    ap.add_argument("--wandb_entity", default="symmetry-advantage")
    ap.add_argument("--wandb_project", default="WyckoffTransformer")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--condition", choices=["hull", "none"], default="hull",
                    help="value for a formation-energy-conditioned backbone; ignored by an unconditional one")
    ap.add_argument("--hull_dataset", default="mp_20")
    ap.add_argument("--hull_splits", nargs="+", default=["train", "val"])
    ap.add_argument("--max_atoms", type=int, default=80,
                    help="largest conventional cell; 80 is the largest in MP-20 train+val")
    ap.add_argument("--given_cell", action="store_true",
                    help="give the ground truth's atom count, as the DiffCSP CSP protocol does, "
                         "instead of letting the model choose z")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repair", action="store_true",
                    help="re-sample only the formulas that failed in an existing --out; seeds are per "
                         "entry, so the result is what a full rerun would give")
    a = ap.parse_args()

    from pymatgen.core import Composition
    from wyckoff_transformer.cli.csp import load_trainer

    with open(a.benchset, "rb") as f:
        entries = pickle.load(f)
    if a.limit:
        entries = entries[: a.limit]
    # Plain element-count formulas: pymatgen's reduced_formula groups units in parentheses,
    # e.g. Li(TiS2)3, which WyckoffTransformer's parse_formula rejects.
    formulas = ["".join(f"{el}{int(n)}" for el, n in e["gt"].composition.reduced_composition.as_dict().items())
                for e in entries]
    old = None
    if a.repair:
        with open(a.out, "rb") as f:
            old = pickle.load(f)
        todo = sorted(old["meta"]["errors"])
    else:
        todo = range(len(entries))

    device = torch.device(a.device)
    trainer = load_trainer(device, wandb_run=a.wandb_run, wandb_entity=a.wandb_entity, wandb_project=a.wandb_project)
    features = list(trainer.condition_features)
    if features and features != ["formation_energy_per_atom"]:
        raise ValueError(f"no conditioning rule for {features}")
    conditions = [None] * len(entries)
    if features:
        if a.condition == "none":
            raise ValueError("the backbone is conditioned on formation energy; pass --condition hull")
        by_system = training_hull(a.hull_dataset, a.hull_splits)
        conditions = [{"formation_energy_per_atom": hull_formation_energy(f, by_system)}
                      for f in tqdm(formulas, desc="training hull")]
        if old is not None and any(abs(c["formation_energy_per_atom"] - o["formation_energy_per_atom"]) > 1e-9
                                   for c, o in zip(conditions, old["meta"]["conditions"])):
            raise ValueError("the training-hull conditions differ from the run being repaired")

    sampler = GeneSampler(trainer, device, a.max_atoms, a.temperature)
    genes = old["genes"] if old is not None else [[None] * a.samples for _ in entries]
    errors = {}
    for i in tqdm(todo, desc="sampling genes"):
        try:
            genes[i] = sampler.sample(formulas[i], a.samples, conditions[i], seed=a.seed * 1_000_003 + i,
                                      n_atoms=entries[i]["gt"].num_sites if a.given_cell else None)
        except Exception as e:  # an element outside the vocabulary, say: the formula gets no samples
            errors[i] = f"{type(e).__name__}: {e}"
            genes[i] = [None] * a.samples

    import wandb

    run = wandb.Api().run(f"{a.wandb_entity}/{a.wandb_project}/{a.wandb_run}")
    models = [art for art in run.logged_artifacts() if art.type == "model"]
    wt = Path(sys.modules["wyckoff_transformer"].__file__).resolve().parent
    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"), "args": vars(a),
        "diffcsp_commit": git_commit(Path(__file__).resolve().parent), "wyckoff_transformer_commit": git_commit(wt),
        "model": {"run": a.wandb_run, "artifact": models[-1].name if models else None,
                  "trained_at_commit": run.config.get("code", {}).get("commit"),
                  "condition_features": features, "composition_conditioning": trainer.composition_conditioning},
        "formulas": formulas, "conditions": conditions, "errors": errors,
    }
    if old is not None:
        meta["repaired"] = {"entries": list(todo), "previous_errors": old["meta"]["errors"],
                            "previous_meta": {k: v for k, v in old["meta"].items() if k not in ("formulas", "conditions")}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"genes": genes, "meta": meta}, f)

    flat = [g for gs in genes for g in gs if g is not None]
    print(f"{len(flat)} genes for {len(formulas)} formulas ({len(errors)} formulas failed) -> {a.out}")
    if flat:
        print(f"  atoms per cell: median {np.median([g['n_atoms'] for g in flat]):.0f}, "
              f"max {max(g['n_atoms'] for g in flat)}; share above 40: "
              f"{np.mean([g['n_atoms'] > 40 for g in flat]):.1%}")


if __name__ == "__main__":
    main()
