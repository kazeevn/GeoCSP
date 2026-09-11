"""Launcher: guarantees this worktree is first on sys.path.

The shared venv's editable .pth for `diffcsp` is rewritten by whichever checkout
last ran `uv`, so it cannot be relied on to point here. Inserting the worktree
root at sys.path[0] makes imports independent of it.
"""
import os, runpy, sys

root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root)
target = sys.argv.pop(1)
sys.argv[0] = target
runpy.run_path(target, run_name="__main__")
