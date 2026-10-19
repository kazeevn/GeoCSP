"""CSP benchmark scoring, Wyckoff-gene track (docs/benchmark.md, "CSP benchmark").

Scores one regime's samples, after bench/mlip_relax.py, against the ground truth
and the hull:

- classic: Recall@N of the raw samples, as in the DiffCSP protocol;
- MLIP-assisted: Recall@N of the relaxed samples, unranked and ranked by ORB
  energy; RMSD before and after relaxation; the three-way breakdown
  matched / lower-energy unmatched / failure;
- saddle points: how often a fixed-symmetry minimum relaxes further once released;
- polymorph counting: unique non-saddle relaxed structures with
  E_hull <= 0.1 eV/atom per gene, on flat and DoF-dependent budgets.

    uv run python bench/evaluate_csp.py --label diffcsp-geo \\
        --inits runs/bench/mp20/inits_t20.pkl \\
        --relaxed runs/bench/mp20/relaxed_geo_t20.shard*.pkl \\
        --gt_relaxed runs/bench/mp20/relaxed_gt.pkl \\
        --out runs/bench/mp20/csp_geo_t20.pkl

Logs the summary, the Recall@N curves and the results file to W&B (online by default).
"""
import os

# Matching is pure CPU work spread over many worker processes.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import datetime
import pickle
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csp_metrics as cm  # noqa: E402
from dof_analysis import representation_dof  # noqa: E402
from mlip_relax import git_commit  # noqa: E402

warnings.filterwarnings("ignore")

READOUTS = ("sym", "free")
DEFAULT_DOF_SCHEDULES = ("0:1,2:2,*:3", "0:1,2:3,5:5,8:8,*:10")


def load_records(paths):
    """Concatenate mlip_relax.py outputs (e.g. shards), refusing mixed provenance."""
    metas, records = [], []
    for p in paths:
        with open(p, "rb") as f:
            blob = pickle.load(f)
        metas.append(blob["meta"])
        records.extend(blob["records"])
    keys = ("relax_mlip", "score_mlip")
    for m in metas[1:]:
        for k in keys:
            if m[k] != metas[0][k]:
                raise ValueError(f"{k} differs between inputs: {m[k]} vs {metas[0][k]}")
        if m["hull"]["revision"] != metas[0]["hull"]["revision"]:
            raise ValueError("inputs were scored against different hull revisions")
    seen = set()
    for r in records:
        key = (r["entry"], r["trial"])
        if key in seen:
            raise ValueError(f"duplicate record {key}: overlapping shards?")
        seen.add(key)
    return metas, records


def _match_record(gt, rec, tol):
    return {k: cm.match_rms(gt, rec.get(k), *tol) for k in ("raw", "relaxed_sym", "relaxed_free")}


def _cluster_entry(cands, tol):
    """Unique polymorphs among one gene's candidates: (structure, energy, matched_gt).

    Returns (all clusters, clusters whose representative does not match the ground truth).
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher

    stol, angle_tol, ltol = tol
    m = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)

    def same(a, b):
        try:
            return m.fit(a, b)
        except Exception:
            return False

    reps = cm.unique_polymorphs([c[0] for c in cands], [c[1] for c in cands], same)
    return len(reps), sum(not cands[j][2] for j in reps)


def _stats(x):
    x = np.asarray([v for v in x if v is not None], dtype=float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "frac_le_0.1": float(np.mean(x <= cm.METASTABLE_E_HULL)), "frac_le_0": float(np.mean(x <= 0))}


def evaluate(entries, records, gt_records, ns, margins, dof_schedules, tol, n_jobs, count_saddles,
             unique_tol=(cm.UNIQUE_STOL, cm.UNIQUE_ANGLE_TOL, cm.UNIQUE_LTOL)):
    """All metrics for one regime. *records* are mlip_relax.py records (or raw-only)."""
    rms = Parallel(n_jobs=n_jobs)(
        delayed(_match_record)(entries[r["entry"]]["gt"], r, tol) for r in tqdm(records, desc="matching")
    )
    for r, m in zip(records, rms):
        r["rms"] = m
        r["saddle"] = cm.is_saddle(r.get("e_pet_sym"), r.get("e_pet_free"))

    by_entry = defaultdict(list)
    for r in records:
        by_entry[r["entry"]].append(r)
    for v in by_entry.values():
        v.sort(key=lambda r: r["trial"])
    # A gene with no surviving sample still counts, as a miss.
    per_entry = [by_entry.get(i, []) for i in range(len(entries))]
    relaxed = any(r.get("relaxed_sym") is not None for r in records)

    def matched(key):
        return [[r["rms"][key] is not None for r in recs] for recs in per_entry]

    def readout(key):
        flat = [r["rms"][key] for r in records]
        hits = [x for x in flat if x is not None]
        return {"recall": cm.recall_at_n(matched(key), ns),
                "match_per_sample": float(np.mean([x is not None for x in flat])) if flat else float("nan"),
                "mean_rms_matched": float(np.mean(hits)) if hits else float("nan")}

    res = {"n_entries": len(entries), "n_samples": len(records), "ns": list(ns), "tolerances": tol,
           "unique_tolerances": unique_tol,
           "classic": readout("raw")}
    if not relaxed:
        return res

    res["relax"] = {
        "n_errors": sum(r["error"] is not None for r in records),
        "median_seconds": float(np.median([r["seconds"] for r in records if r["seconds"] is not None])),
        "frac_free_left_gene_spacegroup": float(np.mean([
            r["sg_free"] != entries[r["entry"]]["rep"]["group"] for r in records if r["sg_free"] is not None])),
        "frac_sym_left_gene_spacegroup": float(np.mean([
            r["sg_sym"] != entries[r["entry"]]["rep"]["group"] for r in records if r["sg_sym"] is not None])),
    }

    res["mlip"] = {}
    for ro in READOUTS:
        d = readout(f"relaxed_{ro}")
        d["ranked_recall"] = cm.ranked_recall_at_n(
            matched(f"relaxed_{ro}"), [[r[f"e_orb_{ro}"] for r in recs] for recs in per_entry], ns)
        pairs = [(r["rms"]["raw"], r["rms"][f"relaxed_{ro}"]) for r in records
                 if r["rms"]["raw"] is not None and r["rms"][f"relaxed_{ro}"] is not None]
        d["rms_before_after"] = {"n_pairs": len(pairs),
                                 "raw": float(np.mean([p[0] for p in pairs])) if pairs else float("nan"),
                                 "relaxed": float(np.mean([p[1] for p in pairs])) if pairs else float("nan")}
        d["ehull"] = _stats(r[f"ehull_{ro}"] for r in records)
        res["mlip"][ro] = d
    res["mlip"]["raw_ehull"] = _stats(r["ehull_raw"] for r in records)

    # Ground-truth energies: ORB on the DFT structure (LeMat-GenBench style),
    # and ORB on the same PET relaxation the samples went through.
    gt_e = {"gt_dft": {}, "gt_relaxed": {}}
    for g in gt_records:
        gt_e["gt_dft"][g["entry"]] = g["e_orb_raw"]
        gt_e["gt_relaxed"][g["entry"]] = g["e_orb_sym"]
    res["three_way"] = {}
    if gt_records:
        for ref, e_ref in gt_e.items():
            for margin in margins:
                for n in ns:
                    outcomes = []
                    for i, recs in enumerate(per_entry):
                        first = recs[:n]
                        es = [r["e_orb_sym"] for r in first if r["e_orb_sym"] is not None]
                        outcomes.append(cm.three_way(any(r["rms"]["relaxed_sym"] is not None for r in first),
                                                     min(es) if es else None, e_ref.get(i), margin))
                    res["three_way"][f"{ref}/margin_{margin:g}/N{n}"] = cm.breakdown(outcomes)
        gt_m = Parallel(n_jobs=n_jobs)(
            delayed(cm.match_rms)(entries[g["entry"]]["gt"], g["relaxed_sym"], *tol) for g in gt_records)
        res["gt"] = {
            "n": len(gt_records),
            "n_errors": sum(g["error"] is not None for g in gt_records),
            "ehull_dft_structure": _stats(g["ehull_raw"] for g in gt_records),
            "ehull_relaxed_sym": _stats(g["ehull_sym"] for g in gt_records),
            "relaxed_matches_dft": float(np.mean([m is not None for m in gt_m])),
        }

    def saddle_stats(recs):
        meta = [r for r in recs if r.get("ehull_sym") is not None and r["ehull_sym"] <= cm.METASTABLE_E_HULL
                and r["saddle"] is not None]
        flags = [r["saddle"] for r in recs if r.get("saddle") is not None]
        return {"frac_saddle": float(np.mean(flags)) if flags else float("nan"),
                "n_metastable_sym": len(meta),
                "frac_saddle_among_metastable_sym": float(np.mean([r["saddle"] for r in meta])) if meta else float("nan"),
                "frac_rattle_accepted": float(np.mean([bool(r["rattle_accepted"]) for r in recs
                                                       if r.get("rattle_accepted") is not None]))
                if any(r.get("rattle_accepted") is not None for r in recs) else float("nan")}

    for g in gt_records:
        g["saddle"] = cm.is_saddle(g.get("e_pet_sym"), g.get("e_pet_free"))
    res["saddle"] = {"samples": saddle_stats(records)}
    if gt_records:
        res["saddle"]["gt"] = saddle_stats(gt_records)

    # Polymorph counting: neither rewards nor penalises recovering the ground truth,
    # which is counted as an ordinary polymorph ("count"); "count_novel" drops it.
    dofs = [representation_dof(e["rep"])[0] for e in entries]
    budgets = {f"N{n}": [n] * len(entries) for n in ns}
    for s in dof_schedules:
        budgets[f"dof[{s}]"] = [cm.dof_budget(d, s) for d in dofs]
    res["polymorphs"] = {}
    for name, per_gene in budgets.items():
        jobs = []
        n_meta = n_seen = 0
        for recs, b in zip(per_entry, per_gene):
            first = recs[:b]
            n_seen += len(first)
            cands = [(r["relaxed_sym"], r["e_orb_sym"], r["rms"]["relaxed_sym"] is not None) for r in first
                     if r["ehull_sym"] is not None and r["ehull_sym"] <= cm.METASTABLE_E_HULL
                     and (count_saddles or r["saddle"] is False)]
            n_meta += len(cands)
            jobs.append(cands)
        counts = Parallel(n_jobs=n_jobs)(delayed(_cluster_entry)(c, unique_tol) for c in jobs if c)
        it = iter(counts)
        full = [next(it) if c else (0, 0) for c in jobs]
        c_all, c_novel = np.array([c[0] for c in full]), np.array([c[1] for c in full])
        res["polymorphs"][name] = {
            # What was used, which is less than the budget where a gene ran out of samples.
            "samples_per_gene": n_seen / len(entries),
            "budget_per_gene": float(np.mean(per_gene)),
            "frac_samples_metastable": n_meta / n_seen if n_seen else float("nan"),
            "count_mean": float(c_all.mean()), "count_median": float(np.median(c_all)),
            "frac_genes_ge1": float(np.mean(c_all >= 1)),
            "count_novel_mean": float(c_novel.mean()), "frac_genes_novel_ge1": float(np.mean(c_novel >= 1)),
        }
    res["per_entry"] = {
        i: [{k: r.get(k) for k in ("trial", "rms", "e_orb_raw", "e_orb_sym", "e_orb_free", "ehull_raw",
                                   "ehull_sym", "ehull_free", "e_pet_sym", "e_pet_free", "saddle",
                                   "rattle_accepted", "sg_sym", "sg_free", "error")} for r in recs]
        for i, recs in enumerate(per_entry)
    }
    return res


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        elif isinstance(v, int | float | np.floating | np.integer) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def print_summary(label, res):
    ns = res["ns"]
    print(f"\n{label}: {res['n_entries']} genes, {res['n_samples']} samples; "
          f"StructureMatcher(stol, angle_tol, ltol) = {res['tolerances']}")
    head = "  ".join(f"{'R@' + str(n):>7}" for n in ns)
    print(f"{'':<22}{head}   mean RMS")
    rows = [("classic (raw)", res["classic"]["recall"], res["classic"]["mean_rms_matched"])]
    for ro in READOUTS if "mlip" in res else ():
        d = res["mlip"][ro]
        rows.append((f"relaxed {ro}", d["recall"], d["mean_rms_matched"]))
        rows.append((f"relaxed {ro}, ranked", d["ranked_recall"], float("nan")))
    for name, rec, rms in rows:
        print(f"{name:<22}" + "  ".join(f"{100 * rec[n]:>6.1f}%" for n in ns) + f"   {rms:.4f}")
    if "mlip" not in res:
        return
    print("\nthree-way (matched / lower / failure), sym readout:")
    for k, v in res["three_way"].items():
        print(f"  {k:<28} {100 * v['matched']:5.1f}% / {100 * v['lower']:5.1f}% / {100 * v['failure']:5.1f}%")
    s = res["saddle"]["samples"]
    print(f"\nsaddles: {100 * s['frac_saddle']:.1f}% of samples, "
          f"{100 * s['frac_saddle_among_metastable_sym']:.1f}% of {s['n_metastable_sym']} metastable sym readouts")
    if "gt" in res["saddle"]:
        g = res["saddle"]["gt"]
        print(f"         {100 * g['frac_saddle']:.1f}% of ground truths, "
              f"{100 * g['frac_saddle_among_metastable_sym']:.1f}% of {g['n_metastable_sym']} metastable")
    print(f"\npolymorphs (unique at {res['unique_tolerances']}, E_hull <= 0.1, non-saddle) per gene:")
    for k, v in res["polymorphs"].items():
        print(f"  {k:<28} {v['samples_per_gene']:5.2f} samples/gene  count {v['count_mean']:.3f} "
              f"(>=1: {100 * v['frac_genes_ge1']:.1f}%)  novel {v['count_novel_mean']:.3f} "
              f"(>=1: {100 * v['frac_genes_novel_ge1']:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--inits", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--relaxed", nargs="+", help="mlip_relax.py outputs for this regime (shards allowed)")
    src.add_argument("--preds", help="raw predictions only: classic metrics, no MLIP")
    ap.add_argument("--gt_relaxed", nargs="*", default=[], help="mlip_relax.py --gt outputs")
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.01],
                    help="eV/atom a sample must beat the ground truth by to count as a lower polymorph")
    ap.add_argument("--dof_schedules", nargs="*", default=list(DEFAULT_DOF_SCHEDULES))
    ap.add_argument("--count_saddles", action="store_true", help="count saddle points as polymorphs")
    ap.add_argument("--stol", type=float, default=cm.STOL)
    ap.add_argument("--angle_tol", type=float, default=cm.ANGLE_TOL)
    ap.add_argument("--ltol", type=float, default=cm.LTOL)
    ap.add_argument("--unique_stol", type=float, default=cm.UNIQUE_STOL)
    ap.add_argument("--unique_angle_tol", type=float, default=cm.UNIQUE_ANGLE_TOL)
    ap.add_argument("--unique_ltol", type=float, default=cm.UNIQUE_LTOL)
    ap.add_argument("--n_jobs", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_project", default="diffcsp")
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", "symmetry-advantage"))
    ap.add_argument("--wandb_tags", nargs="*", default=[])
    a = ap.parse_args()

    with open(a.inits, "rb") as f:
        blob = pickle.load(f)
    entries, trials = blob["entries"], blob["trials"]
    ns = sorted({n for n in a.ns if n <= trials} | {trials})

    metas = []
    if a.relaxed:
        metas, records = load_records(a.relaxed)
    else:
        with open(a.preds, "rb") as f:
            records = [{"entry": p["entry"], "trial": p["trial"], "raw": p["pred"]} for p in pickle.load(f)]
    gt_metas, gt_records = load_records(a.gt_relaxed) if a.gt_relaxed else ([], [])
    if metas and gt_metas:
        for k in ("relax_mlip", "score_mlip"):
            if metas[0][k] != gt_metas[0][k]:
                raise ValueError(f"samples and ground truth disagree on {k}")

    tol = (a.stol, a.angle_tol, a.ltol)
    unique_tol = (a.unique_stol, a.unique_angle_tol, a.unique_ltol)
    res = evaluate(entries, records, gt_records, ns, a.margins, a.dof_schedules, tol, a.n_jobs, a.count_saddles,
                   unique_tol)
    res["label"] = a.label
    res["meta"] = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
                   "diffcsp_commit": git_commit(Path(__file__).resolve().parent),
                   "args": vars(a), "trials": trials, "relaxed_meta": metas[0] if metas else None,
                   "gt_meta": gt_metas[0] if gt_metas else None}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(res, f)
    print_summary(a.label, res)
    print(f"\n-> {a.out}")

    if a.wandb_mode == "disabled":
        return
    import wandb

    rm = res["meta"]["relaxed_meta"] or {}
    run = wandb.init(
        project=a.wandb_project, entity=a.wandb_entity, mode=a.wandb_mode, job_type="csp_bench",
        name=f"csp_bench-{a.label}", tags=["csp_bench", *a.wandb_tags],
        config={"label": a.label, "inits": a.inits, "relaxed": a.relaxed, "preds": a.preds,
                "gt_relaxed": a.gt_relaxed, "trials": trials, "ns": ns, "margins": a.margins,
                "dof_schedules": a.dof_schedules, "count_saddles": a.count_saddles, "tolerances": tol, "unique_tolerances": unique_tol,
                "git_commit": res["meta"]["diffcsp_commit"],
                "wyckoff_transformer_commit": rm.get("wyckoff_transformer_commit"),
                "relax_mlip": rm.get("relax_mlip"), "score_mlip": rm.get("score_mlip"),
                "hull": rm.get("hull"), "relax_args": rm.get("args")},
    )
    summary = {k: v for k, v in res.items()
               if k not in ("per_entry", "meta", "label", "tolerances", "unique_tolerances", "ns")}
    run.summary.update(flatten(summary, "csp_bench/"))
    cols = ["N", "classic"] + ([f"{ro}{s}" for ro in READOUTS for s in ("", "_ranked")] if "mlip" in res else [])
    table = wandb.Table(columns=cols)
    for n in ns:
        row = [n, res["classic"]["recall"][n]]
        if "mlip" in res:
            row += [res["mlip"][ro][k][n] for ro in READOUTS for k in ("recall", "ranked_recall")]
        table.add_data(*row)
    run.log({"csp_bench/recall_at_n": table})
    art = wandb.Artifact(f"csp_bench-{run.id}", type="csp_bench_eval")
    art.add_file(a.out)
    run.log_artifact(art)
    print(f"W&B run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
