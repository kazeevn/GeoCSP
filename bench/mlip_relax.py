"""CSP benchmark stage: relax samples with one MLIP, score them with another.

Each structure goes through WyckoffTransformer's stepwise relaxation with
PET-OAM-XL by default: fix-cell warm-up, then cell + positions under
FixSymmetry, then symmetry released, then a rattle. Two readouts are kept:

- ``relaxed_sym``: the output of the symmetry-constrained stages. It is still on
  the gene's Wyckoff orbits, so it is the primary readout of the gene track.
- ``relaxed_free``: what the free relaxation and rattle kept. If it is lower
  than ``relaxed_sym`` by more than csp_metrics.SADDLE_MARGIN, the
  fixed-symmetry minimum is a saddle point of the potential.

Energies are ORB single points (``orb_conserv_inf``) on the raw structure and
both readouts, and E_hull comes from LeMat-Bulk's published hull for the same
ORB model. That hull is self-consistent with those energies, which keeps the
numbers comparable with LeMat-Bulk-MLIP / LeMat-GenBench.

Needs the csp-bench extra (docs/platforms/zeus/environment.md):

    CUDA_VISIBLE_DEVICES=1 uv run python bench/mlip_relax.py \\
        --inits runs/bench/mp20/inits_t20.pkl --preds runs/bench/mp20/pred_none_t20.pkl \\
        --out runs/bench/mp20/relaxed_none_t20.pkl --shard 0/4

Writes ``<out>`` (or ``<out stem>.shard<i>of<n>.pkl`` when sharded) holding
``{"meta": ..., "records": [...]}``. It saves every --save_every items and
resumes from an existing file. bench/evaluate_csp.py accepts the shards directly.
"""
import argparse
import datetime
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csp_metrics import is_broken  # noqa: E402
from run_relax import _detach  # noqa: E402

warnings.filterwarnings("ignore")


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def git_commit(repo: Path) -> str | None:
    try:
        sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(repo), "diff", "--quiet", "HEAD"]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


PROVENANCE_PACKAGES = ("orb-models", "upet", "metatrain", "metatomic-torch", "metatensor-torch",
                       "nvalchemi-toolkit-ops", "torch")


def package_provenance() -> dict:
    """Version, and VCS commit where installed from git, of the packages behind the energies.

    orb-models is installed from a git commit whose metadata reports version
    0.3.2.dev0, so its version number alone identifies nothing.
    """
    import json
    from importlib.metadata import PackageNotFoundError, distribution

    out = {}
    for name in PROVENANCE_PACKAGES:
        try:
            dist = distribution(name)
        except PackageNotFoundError:
            out[name] = None
            continue
        info = {"version": dist.version}
        direct = dist.read_text("direct_url.json")
        if direct:
            vcs = json.loads(direct).get("vcs_info")
            if vcs:
                info["commit"] = vcs.get("commit_id")
        out[name] = info
    return out


def _items(blob, preds, gt: bool, max_trials: int | None):
    """(entry, trial, structure) to process, in a fixed order."""
    if gt:
        return [(i, -1, e["gt"]) for i, e in enumerate(blob["entries"])]
    items = sorted((p["entry"], p["trial"], p["pred"]) for p in preds)
    if max_trials is not None:
        items = [it for it in items if it[1] < max_trials]
    return items


def _seed(entry: int, trial: int) -> int:
    # Disjoint from gen_init.py's 1000 * t + i draw seeds, and from each other.
    return 10_000_000 + 1000 * (trial + 1) + entry


def _orb_energy(atoms, calc) -> float:
    a = atoms.copy()
    a.calc = calc
    return float(a.get_potential_energy())


def _ehull(hull, energy, structure):
    try:
        return float(hull.energy_above_hull(energy, structure.composition))
    except ValueError:
        return None


def _spacegroup(structure):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        return int(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())
    except Exception:
        return None


def process(entry, trial, structure, relax_calc, score_calc, hull, a) -> dict:
    """Relax, score and place one structure on the hull. Never raises."""
    from pymatgen.io.ase import AseAtomsAdaptor
    from wyckoff_transformer.cryspr.relaxer import stepwise_relax_stages

    rec = {"entry": entry, "trial": trial, "raw": None, "relaxed_sym": None, "relaxed_free": None,
           "n_atoms": None, "e_pet_sym": None, "e_pet_free": None,
           "e_orb_raw": None, "e_orb_sym": None, "e_orb_free": None,
           "ehull_raw": None, "ehull_sym": None, "ehull_free": None,
           "sg_sym": None, "sg_free": None,
           "rattle_accepted": None, "rattle_delta": None, "seconds": None, "error": None}
    if structure is None or is_broken(structure):
        rec["error"] = "no or broken input structure"
        return rec

    t0 = time.perf_counter()
    signal.alarm(a.timeout)
    try:
        raw = _detach(structure)
        n = raw.num_sites
        rec["raw"], rec["n_atoms"] = raw, n
        atoms = AseAtomsAdaptor.get_atoms(raw)

        e = _orb_energy(atoms, score_calc)
        rec["e_orb_raw"], rec["ehull_raw"] = e / n, _ehull(hull, e, raw)

        with tempfile.TemporaryDirectory(prefix="mlip_relax_") as wdir:
            stages = stepwise_relax_stages(
                atoms, relax_calc, rattle=not a.no_rattle, seed=_seed(entry, trial),
                fmax=a.fmax, steps_limit=a.steps, wdir=Path(wdir), warmup=True,
            )
        rec["rattle_accepted"] = stages.rattle_accepted
        rec["rattle_delta"] = stages.rattle_delta_ev_per_atom
        rec["e_pet_sym"] = stages.fixed_symmetry_energy / n
        kept = stages.kept.copy()
        kept.calc = relax_calc
        rec["e_pet_free"] = float(kept.get_potential_energy()) / n

        for key, at in (("sym", stages.fixed_symmetry), ("free", stages.kept)):
            s = _detach(AseAtomsAdaptor.get_structure(at))
            e = _orb_energy(at, score_calc)
            rec[f"relaxed_{key}"] = s
            rec[f"e_orb_{key}"], rec[f"ehull_{key}"] = e / n, _ehull(hull, e, s)
            rec[f"sg_{key}"] = _spacegroup(s)
    except _Timeout:
        rec["error"] = f"timeout after {a.timeout} s"
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        signal.alarm(0)
    rec["seconds"] = time.perf_counter() - t0
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inits", required=True, help="gen_init.py output; provides the ground truth")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--preds", help="a regime's predictions, [{entry, trial, pred}]")
    src.add_argument("--gt", action="store_true", help="relax and score the ground-truth structures")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_trials", type=int, default=None, help="only trials < this")
    ap.add_argument("--limit", type=int, default=None, help="only the first N entries")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th item starting at i")
    ap.add_argument("--relax_mlip", default="PET-OAM-XL", help="wyckoff_transformer.cryspr.mlips registry key")
    ap.add_argument("--score_mlip", default="orb_conserv_inf", help="LeMat-Bulk-MLIP-Hull split")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=500, help="optimizer step limit per relaxation stage")
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per structure")
    ap.add_argument("--no_rattle", action="store_true")
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    out = Path(a.out)
    if shard_n > 1:
        out = out.with_name(f"{out.stem}.shard{shard_i}of{shard_n}{out.suffix}")

    with open(a.inits, "rb") as f:
        blob = pickle.load(f)
    preds = None
    if a.preds:
        with open(a.preds, "rb") as f:
            preds = pickle.load(f)
    items = _items(blob, preds, a.gt, a.max_trials)
    if a.limit is not None:
        items = [it for it in items if it[0] < a.limit]
    items = items[shard_i::shard_n]

    records = []
    if out.exists():
        with open(out, "rb") as f:
            records = pickle.load(f)["records"]
    done = {(r["entry"], r["trial"]) for r in records}
    todo = [it for it in items if (it[0], it[1]) not in done]
    print(f"{len(items)} structures in shard {a.shard}, {len(done)} already done, {len(todo)} to go -> {out}")

    from wyckoff_transformer.cryspr.mlips import MLIP_REGISTRY, build_calculator
    from wyckoff_transformer.evaluation.hull_energy import HullEnergyCalculator
    from wyckoff_transformer.evaluation.hull_mlips import build_hull_calculator, resolve_hull_mlip

    relax_calc = build_calculator(a.relax_mlip, device=a.device)
    score_calc = build_hull_calculator(a.score_mlip, device=a.device)
    hull = HullEnergyCalculator(a.score_mlip)

    import wyckoff_transformer

    meta = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "diffcsp_commit": git_commit(Path(__file__).resolve().parent),
        "wyckoff_transformer_commit": git_commit(Path(wyckoff_transformer.__file__).resolve().parent),
        "args": vars(a),
        "source": "gt" if a.gt else a.preds,
        "relax_mlip": {"name": a.relax_mlip, "checkpoint": MLIP_REGISTRY[a.relax_mlip].checkpoint},
        "score_mlip": {"name": a.score_mlip, "checkpoint": resolve_hull_mlip(a.score_mlip).checkpoint},
        "hull": hull.provenance,
        "packages": package_provenance(),
    }

    def save():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"meta": meta, "records": records}, f)
        tmp.replace(out)

    signal.signal(signal.SIGALRM, _alarm)
    t0 = time.perf_counter()
    for k, (entry, trial, structure) in enumerate(tqdm(todo, desc=f"{a.relax_mlip} relax")):
        records.append(process(entry, trial, structure, relax_calc, score_calc, hull, a))
        if (k + 1) % a.save_every == 0:
            save()
    save()

    el = time.perf_counter() - t0
    n_err = sum(r["error"] is not None for r in records)
    print(f"{len(records)} records ({n_err} errors), {el / 60:.1f} min this run "
          f"({el / max(len(todo), 1):.1f} s each) -> {out}")


if __name__ == "__main__":
    main()
