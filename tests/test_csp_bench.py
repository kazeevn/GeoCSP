"""Metric definitions of the Wyckoff-gene CSP benchmark (bench/csp_metrics.py, bench/evaluate_csp.py)."""
import importlib
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))
import csp_metrics as cm  # noqa: E402


@pytest.fixture(scope="module")
def evaluate_csp():
    # The module hides the GPU from its matching workers; keep that out of the test process.
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        return importlib.import_module("evaluate_csp")
    finally:
        if saved is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved


def rocksalt(a=5.64, strain=(1.0, 1.0, 1.0)):
    s = Structure.from_spacegroup(225, Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    s.lattice = Lattice.from_parameters(a * strain[0], a * strain[1], a * strain[2], 90, 90, 90)
    return s


def cscl_type(a=3.5):
    """Same composition as rocksalt, and no match to it even under the DiffCSP protocol."""
    return Structure(Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]).make_supercell([1, 2, 2])


def zincblende(a=5.64):
    """Same composition as rocksalt, different structure."""
    return Structure.from_spacegroup(216, Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.25, 0.25, 0.25]])


ROCKSALT_GENE = {"group": 225, "species": ["Na", "Cl"], "numIons": [4, 4], "sites": [["4a"], ["4b"]]}


def test_recall_at_n_is_a_prefix_over_sampling_order():
    matched = [[False, True, False], [False, False, False], [True]]
    assert cm.recall_at_n(matched, [1, 2, 3]) == {1: pytest.approx(1 / 3), 2: pytest.approx(2 / 3), 3: pytest.approx(2 / 3)}


def test_ranked_recall_keeps_the_lowest_energies_and_ranks_missing_last():
    matched = [[False, True, True]]
    energies = [[-1.0, -2.0, None]]
    assert cm.ranked_recall_at_n(matched, energies, [1]) == {1: 1.0}
    # The matched sample has no energy, so it ranks behind both others.
    assert cm.ranked_recall_at_n([[False, False, True]], energies, [1, 2, 3]) == {1: 0.0, 2: 0.0, 3: 1.0}


@pytest.mark.parametrize(
    "matched,e_min,e_gt,margin,expected",
    [
        (True, 5.0, -1.0, 0.0, "matched"),
        (False, -1.1, -1.0, 0.0, "lower"),
        (False, -1.005, -1.0, 0.01, "failure"),
        (False, -0.9, -1.0, 0.0, "failure"),
        (False, None, -1.0, 0.0, "failure"),
        (False, -2.0, None, 0.0, "failure"),
    ],
)
def test_three_way(matched, e_min, e_gt, margin, expected):
    assert cm.three_way(matched, e_min, e_gt, margin) == expected


def test_breakdown_fractions():
    assert cm.breakdown(["matched", "lower", "failure", "failure"]) == {"matched": 0.25, "lower": 0.25, "failure": 0.5}


def test_saddle_needs_more_than_the_margin():
    assert cm.is_saddle(-1.0, -1.0 - 2e-3) is True
    assert cm.is_saddle(-1.0, -1.0 - 5e-4) is False
    assert cm.is_saddle(None, -1.0) is None


def test_unique_polymorphs_clusters_in_energy_order():
    # "Same" is non-transitive here: 1~2 and 2~3 but not 1~3.
    links = {frozenset((1, 2)), frozenset((2, 3))}

    def same(a, b):
        return a == b or frozenset((a, b)) in links

    # Visiting 2 first absorbs both neighbours; visiting 1 first leaves 3 apart.
    assert cm.unique_polymorphs([1, 2, 3], [0.0, -1.0, 0.0], same) == [1]
    assert cm.unique_polymorphs([1, 2, 3], [-1.0, 0.0, 0.0], same) == [0, 2]


def test_dof_budget_default_schedule():
    assert [cm.dof_budget(d, "0:1,2:2,*:3") for d in (0, 1, 2, 3, 12)] == [1, 2, 2, 3, 3]


def test_canonical_gene_identifies_normalizer_relabellings():
    perovskite = {"group": 221, "species": ["Ba", "Ti", "O"], "sites": [["1a"], ["1b"], ["3c"]]}
    # Pm-3m's normalizer swaps a<->b and c<->d: the same cubic perovskite.
    shifted = {"group": 221, "species": ["O", "Ba", "Ti"], "sites": [["3d"], ["1b"], ["1a"]]}
    # O moved to 3d with Ba and Ti kept: O at the edge centres of Ba's cell, not the face centres.
    swapped_o = {"group": 221, "species": ["Ba", "Ti", "O"], "sites": [["1a"], ["1b"], ["3d"]]}
    assert cm.canonical_gene(perovskite) == cm.canonical_gene(shifted)
    assert cm.canonical_gene(perovskite, equivalent=False) != cm.canonical_gene(shifted, equivalent=False)
    # 1a/1b/3d is the image of 1b/1a/3c, which is not 1a/1b/3c: a different structure.
    assert cm.canonical_gene(perovskite) != cm.canonical_gene(swapped_o)


def test_letter_permutations_keep_multiplicities():
    from pyxtal.symmetry import Group

    for n in range(1, 231):
        mult = {w.letter: w.multiplicity for w in Group(n).Wyckoff_positions}
        for p in cm.letter_permutations(n):
            assert all(mult[a] == mult[b] for a, b in p.items()), n


def test_match_rms_is_the_evaluate_py_protocol():
    gt = rocksalt()
    assert cm.match_rms(gt, rocksalt(a=5.7)) == pytest.approx(0.0, abs=1e-6)
    assert cm.match_rms(gt, cscl_type()) is None
    assert cm.match_rms(gt, None) is None
    evaluate = importlib.import_module("evaluate")
    assert evaluate._score is cm.match_rms


def _record(entry, trial, structure, e_orb, ehull, e_pet_sym=-1.0, e_pet_free=-1.0, error=None):
    ok = error is None
    return {"entry": entry, "trial": trial, "raw": structure, "relaxed_sym": structure if ok else None,
            "relaxed_free": structure if ok else None, "n_atoms": 8,
            "e_pet_sym": e_pet_sym if ok else None, "e_pet_free": e_pet_free if ok else None,
            "e_orb_raw": e_orb, "e_orb_sym": e_orb if ok else None, "e_orb_free": e_orb if ok else None,
            "ehull_raw": ehull, "ehull_sym": ehull if ok else None, "ehull_free": ehull if ok else None,
            "sg_sym": 225, "sg_free": 225, "rattle_accepted": False, "rattle_delta": 0.0,
            "seconds": 1.0, "error": error}


def test_evaluate_end_to_end(evaluate_csp):
    gt = rocksalt()
    entries = [{"gt": gt, "rep": ROCKSALT_GENE}]
    other = cscl_type()
    records = [
        _record(0, 0, other, e_orb=-3.5, ehull=0.05),  # unmatched, below the ground truth
        _record(0, 1, rocksalt(a=5.66), e_orb=-3.4, ehull=0.0),  # matched
        _record(0, 2, None, e_orb=None, ehull=None, error="timeout"),
    ]
    gt_records = [_record(0, -1, gt, e_orb=-3.45, ehull=0.0)]
    res = evaluate_csp.evaluate(entries, records, gt_records, ns=[1, 3], margins=[0.0], dof_schedules=["0:1,*:3"],
                                tol=(cm.STOL, cm.ANGLE_TOL, cm.LTOL), n_jobs=1, count_saddles=False)

    assert res["classic"]["recall"] == {1: 0.0, 3: 1.0}
    assert res["mlip"]["sym"]["recall"] == {1: 0.0, 3: 1.0}
    # Ranking puts the unmatched, lower-energy sample first.
    assert res["mlip"]["sym"]["ranked_recall"] == {1: 0.0, 3: 1.0}
    assert res["three_way"]["gt_dft/margin_0/N1"] == {"matched": 0.0, "lower": 1.0, "failure": 0.0}
    assert res["three_way"]["gt_dft/margin_0/N3"]["matched"] == 1.0
    assert res["relax"]["n_errors"] == 1
    poly = res["polymorphs"]["N3"]
    assert poly["count_mean"] == 2.0 and poly["count_novel_mean"] == 1.0
    assert poly["samples_per_gene"] == 3.0
    # Rocksalt has no positional DoF, so the schedule's first bin applies.
    assert res["polymorphs"]["dof[0:1,*:3]"]["samples_per_gene"] == 1.0
    assert res["polymorphs"]["dof[0:1,*:3]"]["count_novel_mean"] == 1.0
    assert res["gt"]["relaxed_matches_dft"] == 1.0


def test_evaluate_excludes_saddles_from_polymorphs(evaluate_csp):
    gt = rocksalt()
    records = [_record(0, 0, gt, e_orb=-3.4, ehull=0.0, e_pet_sym=-1.0, e_pet_free=-1.1)]
    kw = {"ns": [1], "margins": [0.0], "dof_schedules": [], "tol": (cm.STOL, cm.ANGLE_TOL, cm.LTOL), "n_jobs": 1}
    entries = [{"gt": gt, "rep": ROCKSALT_GENE}]
    res = evaluate_csp.evaluate(entries, records, [], count_saddles=False, **kw)
    assert res["polymorphs"]["N1"]["count_mean"] == 0.0
    assert res["saddle"]["samples"]["frac_saddle"] == 1.0
    res = evaluate_csp.evaluate(entries, records, [], count_saddles=True, **kw)
    assert res["polymorphs"]["N1"]["count_mean"] == 1.0


def test_polymorphs_use_the_stricter_uniqueness_tolerances(evaluate_csp):
    gt = rocksalt()
    # The DiffCSP protocol calls these two the same structure.
    assert cm.match_rms(gt, zincblende()) is not None
    records = [_record(0, 0, gt, e_orb=-3.4, ehull=0.0), _record(0, 1, zincblende(), e_orb=-3.3, ehull=0.05)]
    kw = {"ns": [2], "margins": [0.0], "dof_schedules": [], "tol": (cm.STOL, cm.ANGLE_TOL, cm.LTOL), "n_jobs": 1,
              "count_saddles": False}
    entries = [{"gt": gt, "rep": ROCKSALT_GENE}]
    assert evaluate_csp.evaluate(entries, records, [], **kw)["polymorphs"]["N2"]["count_mean"] == 2.0
    loose = evaluate_csp.evaluate(entries, records, [], unique_tol=kw["tol"], **kw)
    assert loose["polymorphs"]["N2"]["count_mean"] == 1.0


def test_load_records_rejects_overlapping_shards(evaluate_csp, tmp_path):
    meta = {"relax_mlip": {"name": "PET-OAM-XL"}, "score_mlip": {"name": "orb_conserv_inf"}, "hull": {"revision": "x"}}
    rec = _record(0, 0, rocksalt(), e_orb=-3.4, ehull=0.0)
    for name in ("a.pkl", "b.pkl"):
        with open(tmp_path / name, "wb") as f:
            pickle.dump({"meta": meta, "records": [rec]}, f)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_csp.load_records([tmp_path / "a.pkl", tmp_path / "b.pkl"])
    _, records = evaluate_csp.load_records([tmp_path / "a.pkl"])
    assert len(records) == 1 and np.isclose(records[0]["e_orb_sym"], -3.4)
