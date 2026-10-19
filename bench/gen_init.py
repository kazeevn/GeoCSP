"""Stage 2: instantiate each Wyckoff representation with pyxtal.from_random.

Every regime is then run on exactly these draws, so the three are compared on
identical initializations rather than on independent random draws.

With --genes (the CSP formula track, bench/gen_csp_formula.py), trial t of entry i
instantiates the t-th gene a model sampled for that entry's formula instead of the
entry's own representation; each draw then carries its gene.
"""
import argparse, pickle, warnings
from pathlib import Path

from joblib import Parallel, delayed
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _draw(rep: dict, seed: int):
    """One from_random draw, returned as graph arrays plus the initial structure."""
    from pyxtal import pyxtal
    from diffcsp.data.graph import crystal_graph_from_pyxtal

    try:
        c = pyxtal()
        c.from_random(**rep, max_count=30, random_state=seed)
        arrays = crystal_graph_from_pyxtal(c, graph_method="crystalnn")
        return {"graph_arrays": arrays, "init": c.to_pymatgen(resort=False)}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchset", default="runs/bench/mp20/benchset.pkl")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--genes", default=None, help="gen_csp_formula.py output: one gene per (entry, trial)")
    ap.add_argument("--out", default="runs/bench/mp20/inits.pkl")
    a = ap.parse_args()

    with open(a.benchset, "rb") as f:
        entries = pickle.load(f)
    if a.limit:
        entries = entries[: a.limit]
    print(f"{len(entries)} representations x {a.trials} trials")

    genes = genes_meta = None
    if a.genes:
        with open(a.genes, "rb") as f:
            blob = pickle.load(f)
        genes, genes_meta = blob["genes"], blob["meta"]
        if len(genes) < len(entries) or any(len(g) < a.trials for g in genes[: len(entries)]):
            raise ValueError(f"{a.genes} has fewer than {a.trials} genes for some entries")

    def gene(i, t):
        return genes[i][t] if genes else {"rep": entries[i]["rep"]}

    jobs = [(i, t) for i in range(len(entries)) for t in range(a.trials)]
    # A formula whose decoding failed has no gene for that trial: a miss, like a failed draw.
    jobs = [(i, t) for i, t in jobs if gene(i, t) is not None]
    got = Parallel(n_jobs=-1)(
        delayed(_draw)(gene(i, t)["rep"], 1000 * t + i) for i, t in tqdm(jobs, desc="from_random draws")
    )

    out, n_fail = [], 0
    for (i, t), g in zip(jobs, got):
        if g is None:
            n_fail += 1
            continue
        out.append({"entry": i, "trial": t, **g, **({"gene": genes[i][t]} if genes else {})})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"entries": entries, "draws": out, "trials": a.trials,
                     **({"genes_meta": genes_meta} if genes else {})}, f)
    n_missing = len(entries) * a.trials - len(jobs)
    print(f"{len(out)} draws kept, {n_fail} failed, {n_missing} without a gene -> {a.out}")


if __name__ == "__main__":
    main()
