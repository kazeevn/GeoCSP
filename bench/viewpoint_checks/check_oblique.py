"""Is the frame round-trip failure exactly the oblique-projector sites (doc §5.1)?"""
import os
import pickle
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "bench"))
from torch_geometric.loader import DataLoader

from run_diffusion import DrawSet

draws = pickle.load(open("/home/kna/DiffCSPNew/runs/bench/mp20/inits.pkl", "rb"))["draws"][:2048]
batch = next(iter(DataLoader(DrawSet(draws), batch_size=2048, shuffle=False)))
ops, ops_inv, anchor = batch.ops.double(), batch.ops_inv.double(), batch.anchor_index
anchors, inv = torch.unique(anchor, return_inverse=True, sorted=True)
mult = torch.bincount(inv)
order = torch.argsort(inv, stable=True)
start = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), mult[:-1]]), 0)

P0 = ops[anchors][:, :3, :3]
dof = torch.round(torch.diagonal(P0, dim1=-2, dim2=-1).sum(-1)).long()
symmetric = (P0 - P0.transpose(-1, -2)).abs().amax(dim=(-1, -2)) < 1e-9

rng = np.random.default_rng(0)
err = torch.zeros(len(anchors), dtype=torch.float64)
for _ in range(8):
    r = order[start + torch.tensor(rng.integers(0, np.maximum(mult.numpy(), 1)), dtype=torch.long)]
    v = (P0 @ torch.randn(len(anchors), 3, dtype=torch.float64).unsqueeze(-1)).squeeze(-1)
    rt = (ops_inv[r] @ (ops[r][:, :3, :3] @ v.unsqueeze(-1))).squeeze(-1)
    err = torch.maximum(err, (rt - v).abs().amax(-1))

bad = err > 1e-9
print(f"{len(anchors)} sites: {int(symmetric.sum())} symmetric P0, {int((~symmetric).sum())} oblique")
print(f"round-trip fails on {int(bad.sum())} sites")
print(f"  of the failures, oblique: {int((bad & ~symmetric).sum())} / {int(bad.sum())}")
print(f"  symmetric sites that fail: {int((bad & symmetric).sum())}")
print(f"  oblique sites that pass:   {int((~bad & ~symmetric).sum())}")
print(f"max err on symmetric sites = {err[symmetric].max():.3e}")
print()
print("failures by DoF:", {int(d): int((bad & (dof == d)).sum()) for d in dof.unique()})
print("multiplicity>1 AND symmetric (augmentable):",
      int(((mult > 1) & symmetric).sum()), "/", int((mult > 1).sum()), "sites with replicas")
