# The Zeus Platform Environment

DiffCSP runs on **zeus** in a host Python virtual environment (`.venv`) managed via `uv`. The machine-specific characteristic is that **PyTorch is built locally** targeting this host's hardware (Xeon CPU with `-march=native`, Intel oneAPI MKL, and NVIDIA RTX 6000 Ada with CUDA 13.4), served from a local flat-directory index at `/mnt/hdd/torch_wheels/`.

---

## 1. System Overview

| Component | Version / Path | Details |
|---|---|---|
| **OS** | Linux (Ubuntu / Debian x86_64) | Host execution (no container required) |
| **Python** | 3.14.4 (`/usr/bin/python3`) | System interpreter (`home = /usr/bin` in `pyvenv.cfg`) |
| **Package Manager** | `uv` | Dependency resolution, virtualenv management, and execution |
| **CUDA Driver & Toolkit** | CUDA 13.4 (`/usr/local/cuda`) | System installation |
| **PyTorch** | `2.14.1+cu134` (local wheel) | Built for sm89, native CPU instructions, linked against MKL |
| **Wheel Cache Index** | `/mnt/hdd/torch_wheels/` | Flat index for locally-compiled and prebuilt wheels |

---

## 2. Environment Initialization

To initialize or sync the virtual environment on Zeus:

```bash
# Standard initialization (installs orb, wandb, and dev extras)
scripts/platforms/zeus/env_init.sh

# Dry-run mode to inspect resolution without modifying the environment
scripts/platforms/zeus/env_init.sh --dry-run

# Custom extras
DIFFCSP_EXTRAS="dev" scripts/platforms/zeus/env_init.sh
```

### What `env_init.sh` does:
1. Copies `scripts/platforms/zeus/uv.toml` to the project root (`uv.toml` is gitignored as machine-specific state).
2. Executes `uv sync --python "${UV_PYTHON:-/usr/bin/python3}"` with the designated extras (`orb`, `wandb`, `dev`).
3. Reinstalls the local `torch-scatter` / `torch-sparse` wheels. They are not in the lock, since they exist for this Torch only as local wheels, so the exact `uv sync` above removes them every time.
4. Checks and ensures the link to host CUDA 13 NVSHMEM (`libnvshmem_host.so.3`) within the Torch library folder if required.

### CSP benchmark: WyckoffTransformer and PET-OAM-XL

The CSP benchmark (`bench/mlip_relax.py`, [benchmark.md](../../benchmark.md#csp-benchmark--wyckoff-gene-track)) needs the `csp-bench` extra:

```bash
DIFFCSP_EXTRAS="orb wandb dev csp-bench" scripts/platforms/zeus/env_init.sh
```

- **WyckoffTransformer** comes in editable from `../WyckoffTransformer` (`[tool.uv.sources]`), for its relaxer, MLIP registry and LeMat-Bulk hulls. Its `relax` extra is not used: `mace-torch` 0.3.16 ships a top-level `tests` package that shadows this repo's `tests/`, and MACE is not needed. WyckoffTransformer's matminer holds pandas below 3 (2.3.3, was 3.0.5).
- **UPET (PET-OAM-XL)** is an ordinary locked dependency. Its Torch-ABI packages resolve from the local wheel index like Torch itself: `metatomic-torch` 0.1.19 and `metatensor-torch` 0.10.8 come from `/mnt/hdd/torch_wheels/`.
- **orb-models is pinned to a git commit** (`1794f85` on master), because upet 0.3.1 needs `nvalchemi-toolkit-ops>=0.4`, which the latest release (0.7.0) excludes. That commit has the same ORB v3 checkpoint URLs as 0.7.0. It reports version `0.3.2.dev0`, because orb-models stamps versions only at release, so `mlip_relax.py` records the commit instead. Move back to a release once one allows nvalchemi 0.4.

Verified 2026-10-18 at DiffCSPNew `525c232` (dirty): the full test suite passes (84). PET-OAM-XL relaxes the smoke set. ORB `orb_conserv_inf` single points on 111 LeMat-Bulk test structures that sit on the hull reproduce the published `LeMaterial/LeMat-Bulk-MLIP-Hull` energies to a mean of +8.7e-5 and a max of 4.3e-4 eV/atom.

**History.** The first CSP benchmark run (2026-10-16 to 18) predates this setup. PET came from a `--no-deps` PYTHONPATH overlay copied from WyckoffTransformer's notes (upet 0.3.1, metatensor-torch 0.10.6, nvalchemi-toolkit-ops 0.4.0), with orb-models 0.7.0 in the venv. That overlay was never necessary: the lock resolves the metatomic and metatensor wheels from the local index. It also paired orb-models 0.7.0 with an nvalchemi version 0.7.0 excludes. The same ORB check gives identical numbers in that environment (mean +8.8e-5, max 4.3e-4 eV/atom; within 4e-5 eV/atom of the current setup structure by structure), so the run's energies are unaffected.

---

## 3. Configuration & Wheel Index

### Local `uv.toml`
The Zeus configuration in `scripts/platforms/zeus/uv.toml` configures:
- **Local flat index**: Points to `file:///mnt/hdd/torch_wheels/` where Python 3.14 compatible wheels (`torch`, `torch-scatter`, `torch-sparse`, `spglib`, `warp-lang`, etc.) reside.
- **Dependency overrides**: Sets `dm-tree>=0.1.10` to replace transitive constraints with modern C++ / CMake 4.2 compatibility under Python 3.14.
- **Build dependencies**: Enforces build requirements for C++ extensions (`torch-scatter = ["torch"]`, `torch-sparse = ["torch"]`).

### PyTorch Wheels
Local wheels in `/mnt/hdd/torch_wheels/` for Python 3.14 include:
- `torch-2.14.1+cu134-cp314-cp314-linux_x86_64.whl`
- `torch_scatter-2.1.2-cp314-cp314-linux_x86_64.whl`
- `torch_sparse-0.6.18-cp314-cp314-linux_x86_64.whl`
- `openequivariance-0.7.0-cp314-cp314-linux_x86_64.whl`
- `spglib-2.7.0-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
- `warp_lang-1.18.0-py3-none-manylinux_2_28_x86_64.whl`

---

## 4. Verification & Testing

Always verify the environment and test suite after updates:

```bash
# Verify Python interpreter and CUDA device detection
uv run python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Verify optional scatter acceleration
uv run python -c "import torch_scatter, torch_sparse; print('scatter:', torch_scatter.__version__, 'sparse:', torch_sparse.__version__)"

# Execute full automated test suite
uv run pytest
```
