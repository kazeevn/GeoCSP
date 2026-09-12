# Environment Management
- **Always use `uv`** for Python environment management and running commands.
- Run tests with `uv run pytest`.
- Execute CLI tools and Python scripts with `uv run ...` (for example, `uv run python -m diffcsp.cli.train ...` or `uv run diffcsp-train ...`).
- Manage dependencies with `uv` (`uv add`, `uv sync`, etc.). Do not use raw system python or bare `pytest`.
- **Each git worktree needs its own `uv venv`.** Symlinking the root `.venv` into a worktree makes the first `uv run` there repoint the shared editable install, so *other* checkouts silently import the wrong tree. Before any long run, confirm with `uv run python -c "import diffcsp; print(diffcsp.__file__)"`.

# Experiment Tracking
Every trained model whose numbers get quoted must be reproducible from its W&B run alone.
- **Never pass `--no-wandb` to a run you intend to cite.** Tracking is on by default; the opt-out is for smoke tests. The `runs/mp20_wyckoff/` checkpoints were trained with it off and carry `wandb_run_id: None`, so their recipe cannot be recovered — that cost an extra control arm to work around.
- Log to the run: the full command line, the resolved config, `git rev-parse HEAD`, and whether the tree was dirty.
- Log **wall-clock training time and s/epoch**, with the GPU and whether the run was co-scheduled. At these model sizes throughput is overhead-bound rather than FLOP-bound, so a time quoted without contention context is not comparable to another run's.
- Checkpoints already upload as W&B Artifacts (`model-<run-id>`, aliases `epoch-N` and `latest`) and record `wandb_run_id` for resume. Keep it that way, and load them as `wandb://[entity/][project/]model-<id>:alias` rather than by bare path.
- **Log benchmark results back to the training run**, not only into a document. `bench/log_to_wandb.py` does this: it reads `wandb_run_id` from the checkpoint so the match rates land on the run that trained the weights, and uploads `results_*.pkl` and `pred_*.pkl` as a `benchmark` artifact. A match rate that exists only in a terminal is untracked, and a checkpoint with no metrics attached cannot be compared later.
- Training logs `epoch_time_s` and cumulative `train_time_s`; checkpoints carry `train_time_s` and `s_per_epoch`. Do not strip these — they are the only record of what a result cost.

# Documenting an Experiment
- Findings go in `docs/<topic>.md`. Every number in them names the W&B run or checkpoint artifact it came from and the command that produced it.
- Ship the analysis as a script in `bench/`, not ad-hoc code. The document's "Reproducing" section must run end to end as written.
- Quote the statistic the question actually turns on — paired counts and p-values over the relevant subset. Aggregates over a benchmark that is largely saturated hide the effect and overstate the significance.
- Record null and negative results with the same care as positive ones, and state explicitly what was *not* controlled. A hypothesis that failed is only useful if the next reader can see how hard it was tested.
