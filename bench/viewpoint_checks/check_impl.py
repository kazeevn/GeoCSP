"""Gate 3: unaugmented path unchanged; augmented path is a genuine viewpoint change."""
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
from diffcsp.models.wyckoff_cspnet import WyckoffCSPNet
from diffcsp.models.wyckoff_diffusion import WyckoffDiffusion

dev = torch.device("cuda")
draws = pickle.load(open("/home/kna/DiffCSPNew/runs/bench/mp20/inits.pkl", "rb"))["draws"][:256]
batch = next(iter(DataLoader(DrawSet(draws), batch_size=256, shuffle=False))).to(dev)


def make(aug):
    torch.manual_seed(0)
    m = WyckoffDiffusion(
        device=dev, decoder=WyckoffCSPNet(hidden_dim=512, num_layers=6), augment_viewpoint=aug
    ).to(dev)
    return m


plain, aug = make(False), make(True)
aug.load_state_dict(plain.state_dict())

# (1) identical weights -> identical loss when augmentation is OFF
plain.train(); aug.train()
torch.manual_seed(7); a = plain(batch)
torch.manual_seed(7); b = make(False)(batch)
print(f"unaug reproducible: loss {a['loss'].item():.8f} vs {b['loss'].item():.8f}")

# (2) augmented model in eval() must behave exactly like the unaugmented one
plain.eval(); aug.eval()
torch.manual_seed(11); c = plain(batch)
torch.manual_seed(11); d = aug(batch)
same_eval = torch.allclose(c["loss"], d["loss"], atol=0, rtol=0)
print(f"eval(): augmented == unaugmented exactly: {same_eval} "
      f"({c['loss'].item():.8f} vs {d['loss'].item():.8f})")

# (3) in train() the augmented loss must differ (it is seeing other viewpoints)
plain.train(); aug.train()
torch.manual_seed(11); e = plain(batch)
torch.manual_seed(11); f = aug(batch)
print(f"train(): differs: {not torch.allclose(e['loss'], f['loss'])} "
      f"({e['loss'].item():.8f} vs {f['loss'].item():.8f})")

# (4) lattice loss should be untouched by a viewpoint change in expectation;
#     coordinate loss is where the augmentation acts
print(f"   loss_lattice {e['loss_lattice'].item():.6f} -> {f['loss_lattice'].item():.6f}")
print(f"   loss_coord   {e['loss_coord'].item():.6f} -> {f['loss_coord'].item():.6f}")

# (5) gradients flow
f["loss"].backward()
g = [p.grad for p in aug.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
print(f"params with nonzero grad: {len(g)}")

# (6) how many sites actually get augmented
anchors, inv = torch.unique(batch.anchor_index, return_inverse=True, sorted=True)
P0 = batch.ops[anchors][:, :3, :3]
mult = torch.bincount(inv)
sym = (P0 - P0.transpose(-1, -2)).abs().amax(dim=(-1, -2)) < 1e-6
print(f"sites {len(anchors)}: eligible (symmetric & m>1) = {int((sym & (mult > 1)).sum())}, "
      f"skipped oblique = {int((~sym).sum())}, m==1 = {int((mult == 1).sum())}")
