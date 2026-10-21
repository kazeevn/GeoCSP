"""Sanity: do the stored ops already reproduce the cell from the CURRENT anchor?

If this fails, the re-anchoring check was measuring the wrong thing.
"""
import os
import pickle
import sys
import warnings

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "bench"))
from torch_geometric.loader import DataLoader

from run_diffusion import DrawSet

draws = pickle.load(open("/home/kna/DiffCSPNew/runs/bench/mp20/inits.pkl", "rb"))["draws"][:2048]
batch = next(iter(DataLoader(DrawSet(draws), batch_size=2048, shuffle=False)))
ops = batch.ops.double()
x = batch.frac_coords.double() % 1.0
anchor = batch.anchor_index

# exactly what wyckoff_diffusion.forward does to expand anchors -> full cell
xa = torch.cat([x[anchor], torch.ones(len(x), 1, dtype=torch.float64)], -1)
rebuilt = (ops @ xa.unsqueeze(-1)).squeeze(-1)[:, :3] % 1.0
d = ((rebuilt - x + 0.5) % 1.0 - 0.5).abs()
print(f"ops[j] @ (x_anchor,1) vs x_j : max {d.max():.3e}  mean {d.mean():.3e}")
print(f"  atoms off by >1e-6: {int((d.max(-1).values > 1e-6).sum())} / {len(x)}")

# and the 3x3-only form the model actually uses
rebuilt2 = (ops[:, :3, :3] @ x[anchor].unsqueeze(-1)).squeeze(-1) + ops[:, :3, 3]
d2 = ((rebuilt2 % 1.0 - x + 0.5) % 1.0 - 0.5).abs()
print(f"R@x_anchor + t     vs x_j : max {d2.max():.3e}")

# is ops[anchor] really a projector?
anchors = torch.unique(anchor)
P = ops[anchors][:, :3, :3]
print(f"\nops[anchor] linear part: max |P^2-P| = {(P @ P - P).abs().max():.3e}")
tr = torch.diagonal(P, dim1=-2, dim2=-1).sum(-1)
print(f"  trace integral? max |tr - round(tr)| = {(tr - tr.round()).abs().max():.3e}")
print(f"  P symmetric?     max |P - P^T|       = {(P - P.transpose(-1, -2)).abs().max():.3e}")

# does the anchor sit on its own affine subspace?
fix = (P @ x[anchors].unsqueeze(-1)).squeeze(-1) + ops[anchors][:, :3, 3]
print(f"  P@x_a + x0 == x_a: max {(((fix - x[anchors] + 0.5) % 1.0) - 0.5).abs().max():.3e}")
