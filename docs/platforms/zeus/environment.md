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
3. Checks and ensures the link to host CUDA 13 NVSHMEM (`libnvshmem_host.so.3`) within the Torch library folder if required.

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
