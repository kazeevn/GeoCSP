"""Gate 1 (revised): the frame round-trip the augmentation actually needs.

Design chosen (see notes): do NOT rewrite `ops`. Keep the canonical anchor for
noise and target, and change only (a) which orbit member's position is the edge
reference, and (b) map the head's output back from that member's frame with the
already-stored `ops_inv[r] = pinv(R_r)`.

That needs exactly one algebraic property: for a tangent vector v at the canonical
anchor, pinv(R_r) @ (R_r @ v) == v. Equivalently pinv(R_r) R_r acts as the identity
on the canonical tangent space T_0. Also checks R_r @ pinv(R_r) is a genuine
orthogonal projector onto T_r (trace == DoF), which is what makes the mapped-back
score well defined.
"""
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
ops = batch.ops.double()
ops_inv = batch.ops_inv.double()
anchor = batch.anchor_index

anchors, inv = torch.unique(anchor, return_inverse=True, sorted=True)
mult = torch.bincount(inv)
order = torch.argsort(inv, stable=True)
start = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), mult[:-1]]), 0)

P0 = ops[anchors][:, :3, :3]  # canonical projector per site
dof = torch.round(torch.diagonal(P0, dim1=-2, dim2=-1).sum(-1)).long()

rng = np.random.default_rng(0)
worst_rt = worst_proj = worst_tr = 0.0
for _ in range(8):
    r_in_orbit = torch.tensor(rng.integers(0, np.maximum(mult.numpy(), 1)), dtype=torch.long)
    rep = order[start + r_in_orbit]
    R_r = ops[rep][:, :3, :3]
    Rinv_r = ops_inv[rep]

    # (a) round-trip on the canonical tangent space: pinv(R_r) R_r v == v for v in T_0
    v = torch.randn(len(anchors), 3, dtype=torch.float64)
    v = (P0 @ v.unsqueeze(-1)).squeeze(-1)  # into T_0
    rt = (Rinv_r @ (R_r @ v.unsqueeze(-1))).squeeze(-1)
    worst_rt = max(worst_rt, (rt - v).abs().max().item())

    # (b) R_r pinv(R_r) is an orthogonal projector onto T_r, with trace == DoF
    Pr = R_r @ Rinv_r
    worst_proj = max(worst_proj, (Pr @ Pr - Pr).abs().max().item())
    tr = torch.diagonal(Pr, dim1=-2, dim2=-1).sum(-1)
    worst_tr = max(worst_tr, (torch.round(tr) - dof).abs().max().item())

print(f"sites={len(anchors)}  DoF hist={ {int(d): int((dof == d).sum()) for d in dof.unique()} }")
print(f"max |pinv(R_r) R_r v - v| for v in T_0 = {worst_rt:.3e}")
print(f"max |Pr^2 - Pr|                        = {worst_proj:.3e}")
print(f"max |round(tr Pr) - DoF|               = {worst_tr:.3e}")
print("PASS" if worst_rt < 1e-9 and worst_proj < 1e-9 and worst_tr == 0 else "FAIL")
