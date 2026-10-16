#!/bin/bash
# Initialise the DiffCSP environment on zeus: put this host's uv configuration
# in place, then sync the venv against it.
#
# Usage:
#     scripts/platforms/zeus/env_init.sh              # orb + wandb + dev
#     scripts/platforms/zeus/env_init.sh --dry-run    # show the plan, change nothing
#     DIFFCSP_EXTRAS="dev" scripts/platforms/zeus/env_init.sh
#
# Extras are named here rather than in uv.toml because `extra = [...]` in a
# uv.toml is honoured by the `uv pip` interface and ignored by `uv sync` --
# left to itself, `uv sync` prunes the venv down to the base dependencies.
set -euo pipefail

DIFFCSP_EXTRAS=${DIFFCSP_EXTRAS:-"orb wandb dev"}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

# uv reads uv.toml from the project root only. It is gitignored: this copy is
# the tracked source of truth.
cp scripts/platforms/zeus/uv.toml uv.toml

extra_args=()
for extra in ${DIFFCSP_EXTRAS}; do
    extra_args+=(--extra "${extra}")
done

# uv sync creates .venv if it is missing, honouring requires-python from
# pyproject.toml. On zeus, default to the system Python 3.14 interpreter.
uv sync --python "${UV_PYTHON:-/usr/bin/python3}" "${extra_args[@]}" "$@"

# Link NVSHMEM library into Torch's $ORIGIN directory if needed.
# Keep --dry-run read-only and honour scratch/worktree environments.
if [[ " $* " != *" --dry-run "* ]]; then
    for torch_lib in "${UV_PROJECT_ENVIRONMENT:-.venv}"/lib/python*/site-packages/torch/lib; do
        nvshmem_lib=/usr/lib/x86_64-linux-gnu/nvshmem/13/libnvshmem_host.so.3
        if [[ -d "$torch_lib" && -f "$nvshmem_lib" && ! -e "$torch_lib/libnvshmem_host.so.3" ]]; then
            ln -s "$nvshmem_lib" "$torch_lib/libnvshmem_host.so.3"
        fi
    done
fi

cat <<MSG

Environment synced with extras: ${DIFFCSP_EXTRAS}
Activate it with:

    source ${repo_root}/.venv/bin/activate
MSG
