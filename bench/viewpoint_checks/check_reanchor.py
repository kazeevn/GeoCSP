"""Gate 1+2: does M'_j = M_j @ pinv(M_r) reproduce the cell and a valid projector?"""
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

INITS = "/home/kna/DiffCSPNew/runs/bench/mp20/inits.pkl"
draws = pickle.load(open(INITS, "rb"))["draws"][:2048]
batch = next(iter(DataLoader(DrawSet(draws), batch_size=2048, shuffle=False)))
ops, anchor = batch.ops.double(), batch.anchor_index
x = batch.frac_coords.double() % 1.0
print(f"{batch.num_graphs} crystals, {len(x)} atoms, {len(torch.unique(anchor))} sites")

anchors, inv = torch.unique(anchor, return_inverse=True, sorted=True)
mult = torch.bincount(inv)
order = torch.argsort(inv, stable=True)
start = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), mult[:-1]]), 0)

dof_of_site = torch.round(torch.diagonal(ops[anchors][:, :3, :3], dim1=-2, dim2=-1).sum(-1)).long()
print("sites by DoF:", {int(d): int((dof_of_site == d).sum()) for d in dof_of_site.unique()})
print("sites by multiplicity:", {int(m): int((mult == m).sum()) for m in mult.unique()[:12]})

rng = np.random.default_rng(0)
worst_cell = worst_idem = worst_trace = 0.0
n_checked = 0
for trial in range(6):
    r_in_orbit = torch.tensor(rng.integers(0, np.maximum(mult.numpy(), 1)), dtype=torch.long)
    rep_atom = order[start + r_in_orbit]  # (K,) chosen member per site
    M_r = ops[rep_atom]  # (K,4,4)
    M_r_pinv = torch.linalg.pinv(M_r)
    new_ops = ops @ M_r_pinv[inv]  # (N,4,4)

    # does the re-anchored op reproduce every atom from the NEW anchor's position?
    y = torch.cat([x[rep_atom][inv], torch.ones(len(x), 1, dtype=torch.float64)], -1)
    rebuilt = (new_ops @ y.unsqueeze(-1)).squeeze(-1)[:, :3] % 1.0
    d = (rebuilt - x + 0.5) % 1.0 - 0.5  # wrap-aware difference
    worst_cell = max(worst_cell, d.abs().max().item())

    P_new = new_ops[rep_atom][:, :3, :3]
    worst_idem = max(worst_idem, (P_new @ P_new - P_new).abs().max().item())
    tr = torch.diagonal(P_new, dim1=-2, dim2=-1).sum(-1)
    worst_trace = max(worst_trace, (torch.round(tr) - dof_of_site).abs().max().item())
    n_checked += len(x)

print(f"\nover {n_checked} atom re-anchorings across 6 random draws:")
print(f"  max |rebuilt - original| (mod 1)   = {worst_cell:.3e}")
print(f"  max |P'^2 - P'| (idempotence)      = {worst_idem:.3e}")
print(f"  max |round(tr P') - DoF|           = {worst_trace:.3e}")
print("PASS" if worst_cell < 1e-9 and worst_idem < 1e-9 and worst_trace == 0 else "FAIL")
