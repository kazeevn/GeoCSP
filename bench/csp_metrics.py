"""Metrics for the Wyckoff-gene CSP benchmarks (see docs/benchmark.md).

Pure functions over per-sample records, so that bench/evaluate_csp.py stays a
thin driver and the definitions can be unit-tested without an MLIP. Every
energy here is in eV/atom; the samples of one gene share its composition, so
per-atom energies of a gene's samples and of its ground truth are comparable.
"""
import functools
from collections.abc import Callable, Sequence

import numpy as np

# The DiffCSP structure matching protocol.
STOL, ANGLE_TOL, LTOL = 0.5, 10.0, 0.3

# Tolerances for deciding that two generated structures are the same polymorph:
# pymatgen's StructureMatcher defaults. The DiffCSP protocol is too loose for it:
# at stol=0.5 it matches zincblende to rocksalt NaCl (rms 0.43).
UNIQUE_STOL, UNIQUE_ANGLE_TOL, UNIQUE_LTOL = 0.3, 5.0, 0.2

# Metastability threshold of the benchmark, eV/atom above the hull.
METASTABLE_E_HULL = 0.1

# A fixed-symmetry minimum is called a saddle point when releasing the
# symmetry (and rattling) lowers its energy by more than this, eV/atom. Same
# margin as WyckoffTransformer's RATTLE_ACCEPT_EV_PER_ATOM.
SADDLE_MARGIN = 1e-3


def is_broken(structure) -> bool:
    """A lattice no matcher or potential should be handed."""
    try:
        vol = structure.lattice.volume
        if vol < 0.1 or np.isnan(vol) or np.isinf(vol):
            return True
        return bool(any(np.isnan(structure.lattice.abc)) or any(x <= 0 for x in structure.lattice.abc))
    except Exception:
        return True


def match_rms(gt, pred, stol=STOL, angle_tol=ANGLE_TOL, ltol=LTOL):
    """RMS distance of *pred* to *gt* under StructureMatcher, or None if no match."""
    if pred is None or is_broken(pred):
        return None

    from pymatgen.analysis.structure_matcher import StructureMatcher

    m = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
    try:
        r = m.get_rms_dist(gt, pred)
        return None if r is None else float(r[0])
    except Exception:
        return None


def recall_at_n(matched: Sequence[Sequence[bool]], ns: Sequence[int]) -> dict[int, float]:
    """Fraction of genes with a match among their first N samples.

    Args:
        matched: Per gene, whether each sample matched, in sampling order.
        ns: Budgets to report.
    """
    return {n: float(np.mean([any(m[:n]) for m in matched])) for n in ns}


def ranked_recall_at_n(
    matched: Sequence[Sequence[bool]], energies: Sequence[Sequence[float]], ns: Sequence[int]
) -> dict[int, float]:
    """Recall@N where the N kept samples are the N lowest in energy.

    A sample with no energy (failed relaxation) ranks last. Ties keep sampling
    order, so the ranking is deterministic.
    """
    out = {}
    for n in ns:
        hits = []
        for m, e in zip(matched, energies):
            key = [np.inf if x is None or np.isnan(x) else x for x in e]
            order = np.argsort(key, kind="stable")[:n]
            hits.append(any(m[j] for j in order))
        out[n] = float(np.mean(hits))
    return out


def three_way(matched_any: bool, e_min: float | None, e_gt: float | None, margin: float = 0.0) -> str:
    """Classify one gene's samples against its ground truth.

    Returns:
        ``"matched"`` if a sample matched the ground truth; otherwise
        ``"lower"`` if the lowest sample energy is below the ground truth's by
        more than *margin* (a putative lower-energy polymorph); otherwise
        ``"failure"``. A gene without a usable energy on either side, and no
        match, is a failure.
    """
    if matched_any:
        return "matched"
    if e_min is None or e_gt is None or np.isnan(e_min) or np.isnan(e_gt):
        return "failure"
    return "lower" if e_min < e_gt - margin else "failure"


def breakdown(outcomes: Sequence[str]) -> dict[str, float]:
    """Fractions of each three-way outcome."""
    n = len(outcomes)
    return {k: (sum(o == k for o in outcomes) / n if n else float("nan")) for k in ("matched", "lower", "failure")}


def is_saddle(e_sym: float | None, e_free: float | None, margin: float = SADDLE_MARGIN) -> bool | None:
    """Whether the fixed-symmetry minimum relaxed further once released, by *margin*.

    Energies are per atom, from the potential that did the relaxation.
    """
    if e_sym is None or e_free is None:
        return None
    return bool(e_free < e_sym - margin)


def unique_polymorphs(
    structures: Sequence, energies: Sequence[float], same: Callable[[object, object], bool]
) -> list[int]:
    """Greedy StructureMatcher clustering into unique structures.

    Structures are visited in ascending energy, and each joins the first
    representative it matches or becomes a new one. StructureMatcher is not
    transitive, so the clustering depends on the visiting order; fixing it to
    energy order makes it deterministic and makes each representative the
    lowest-energy member of its cluster.

    Returns:
        Indices into *structures* of the cluster representatives.
    """
    reps: list[int] = []
    for i in np.argsort(energies, kind="stable"):
        if not any(same(structures[i], structures[j]) for j in reps):
            reps.append(int(i))
    return reps


def dof_budget(dof: int, schedule: str) -> int:
    """Samples a gene with *dof* positional DoF gets under a WyckoffTransformer trial schedule.

    *schedule* is ``"dof:trials"`` pairs, e.g. ``"0:1,2:2,*:3"``; see
    ``wyckoff_transformer.evaluation.protocol.parse_trial_schedule``.
    """
    from wyckoff_transformer.evaluation.protocol import parse_trial_schedule, trials_for_dof

    return trials_for_dof(dof, parse_trial_schedule(schedule))


@functools.cache
def letter_permutations(group: int) -> tuple[dict[str, str], ...]:
    """The Wyckoff-letter relabellings of space group *group* that describe the same structure.

    PyXtal's ``Group.get_alternatives()``: each row of ``Transformed WP`` lists the images of
    the letters a, b, c, ... under one coset of the space group's normalizer. They form a group
    for all 230 space groups (WyckoffTransformer, docs/wyckoff_augmentation_audit.md), so the
    minimum over them is a canonical form. Rows that permute nothing are collapsed.
    """
    from pyxtal.symmetry import Group

    rows = [tuple(r.split()) for r in Group(group).get_alternatives()["Transformed WP"]]
    return tuple(dict(zip(rows[0], row)) for row in dict.fromkeys(rows))


def canonical_gene(rep: dict, equivalent: bool = True) -> tuple:
    """A pyXtal representation as (space group, sorted (species, Wyckoff label) pairs).

    With *equivalent*, two genes are equal when a normalizer relabelling maps one onto the
    other, e.g. Ba 1a, Ti 1b, O 3c and Ba 1b, Ti 1a, O 3d in Pm-3m. Without it, labels are
    compared as written, which counts such pairs as different genes. Either way, a structure
    described in two different space groups (a subgroup setting, say) has two genes.
    """
    sites = [(str(sp), lab) for sp, labs in zip(rep["species"], rep["sites"]) for lab in labs]
    if not equivalent:
        return rep["group"], tuple(sorted(sites))
    # A label is the multiplicity followed by one letter; relabelling keeps the multiplicity.
    return rep["group"], min(tuple(sorted((sp, lab[:-1] + p[lab[-1]]) for sp, lab in sites))
                             for p in letter_permutations(rep["group"]))
