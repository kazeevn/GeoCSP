"""Attaches benchmark results to the W&B run that produced the checkpoint.

Training logs losses; the benchmark produces match rates. Without this they live in
different places and a checkpoint ends up with no quotable metric attached, which is
how `runs/mp20_wyckoff/` ended up unreproducible. This closes the loop: summary
metrics onto the training run, prediction and result pickles as artifacts.

The run id is read from the checkpoint's own ``wandb_run_id`` field, so the metrics
land on the run that trained the weights rather than on a fresh one.

    uv run python bench/log_to_wandb.py \
        --ckpt runs/mp20_wyckoff/wyckoff_inno2_best.pt \
        --results runs/bench/mp20/results.pkl --label "asymm-inno2 (converged)" \
        --artifacts runs/bench/mp20/pred_wyckoff.pkl

See AGENTS.md, "Experiment Tracking".
"""
import argparse, pickle, warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def summary_metrics(entries, per_entry, dof, mult) -> dict:
    """Per-trial and best-of-k rates overall and on the subsets that discriminate."""
    n = len(entries)
    hit = [[r is not None for r in per_entry[i]] for i in range(n)]
    rms = [r for i in range(n) for r in per_entry[i] if r is not None]
    trial = np.array([np.mean(h) if h else 0.0 for h in hit])
    best = np.array([any(h) for h in hit])

    out = {
        "bench/match_per_trial": 100 * float(np.mean([r for h in hit for r in h])),
        "bench/best_of_k": 100 * float(best.mean()),
        "bench/mean_rms": float(np.mean(rms)) if rms else float("nan"),
        "bench/n_representations": n,
    }
    for label, m in [
        ("dof_le_5", dof <= 5),
        ("dof_ge_9", dof >= 9),
        ("mult_eq_1", mult == 1),
        ("mult_gt_1_dof_ge_9", (mult > 1) & (dof >= 9)),
    ]:
        if m.sum():
            out[f"bench/{label}/match_per_trial"] = 100 * float(trial[m].mean())
            out[f"bench/{label}/best_of_k"] = 100 * float(best[m].mean())
            out[f"bench/{label}/n"] = int(m.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint whose wandb_run_id the metrics attach to")
    ap.add_argument("--results", required=True, help="results.pkl from bench/evaluate.py")
    ap.add_argument("--label", required=True, help="regime label inside that results.pkl")
    ap.add_argument("--inits", default="runs/bench/mp20/inits.pkl")
    ap.add_argument("--artifacts", nargs="*", default=[], help="extra files to upload (pred_*.pkl etc.)")
    ap.add_argument("--run", default=None, help="override the run id from the checkpoint")
    ap.add_argument("--project", default="diffcsp")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print what would be logged and exit")
    a = ap.parse_args()

    import torch
    from multiplicity_analysis import representation_stats  # noqa: E402

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    run_id = a.run or (ck.get("wandb_run_id") if isinstance(ck, dict) else None)
    if not run_id:
        raise SystemExit(
            f"'{a.ckpt}' carries no wandb_run_id. It was trained with --no-wandb, so its "
            "metrics cannot be attached to a run -- see AGENTS.md, 'Experiment Tracking'. "
            "Pass --run explicitly only if you know which run produced these weights."
        )

    table = pickle.load(open(a.results, "rb"))
    if a.label not in table:
        raise SystemExit(f"label '{a.label}' not in {a.results}; available: {list(table)}")
    entries = pickle.load(open(a.inits, "rb"))["entries"]

    cache = {}
    stats = [representation_stats(e["rep"], cache) for e in entries]
    dof = np.array([s[0] for s in stats])
    mult = np.array([s[1] for s in stats])

    metrics = summary_metrics(entries, table[a.label]["per_entry"], dof, mult)
    for k in ("epoch", "train_loss", "val_loss", "train_time_s", "s_per_epoch"):
        if isinstance(ck, dict) and ck.get(k) is not None:
            metrics[f"bench/ckpt_{k}"] = ck[k]

    print(f"run {run_id}  <-  {a.label}")
    for k, v in metrics.items():
        print(f"  {k:<44} {v:.4f}" if isinstance(v, float) else f"  {k:<44} {v}")
    uploads = [a.results, *a.artifacts]
    print("  artifacts: " + ", ".join(uploads))
    if a.dry_run:
        return

    import wandb

    run = wandb.init(id=run_id, project=a.project, entity=a.entity, resume="must")
    run.summary.update(metrics)
    artifact = wandb.Artifact(name=f"bench-{run_id}", type="benchmark")
    for f in uploads:
        if Path(f).exists():
            artifact.add_file(str(f))
    run.log_artifact(artifact, aliases=["latest"])
    run.finish()
    print(f"attached {len(metrics)} metrics and {len(uploads)} artifacts to run {run_id}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
