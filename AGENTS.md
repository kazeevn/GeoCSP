# Environment Management
- **Always use `uv`** for Python environment management and running commands. Just running system Python won't work and can destroy the virtual environment!
- Run tests with `uv run pytest`.
- Execute CLI tools and Python scripts with `uv run ...` (for example, `uv run python -m diffcsp.cli.train ...` or `uv run diffcsp-train ...`).
- Manage dependencies with `uv` (`uv add`, `uv sync`, etc.). Do not use raw system python or bare `pytest`.

# Project conventions
This is a research project, which means that ideas will be tried and discarded. As such, it is extremely important to avoid confusion. In particular:
1. When describing the results of an experiment, note the date, git commit hash and applicable WanDB runs
2. Files in docs/archive/ are not maintained - there is no need to update them, but they might contradict the state of the code base and later findings
3. In general, files in docs/*.md are supposed to be current - correct them if you notice a contradiction
4. Ideas which haven't yet been tried should go to docs/ideas/
5. By default, run with WanDB online logging enabled - unless there is a specific reason not to, such as avoiding overwriting. Substantial results of the experiments should be logged to WanDB.
