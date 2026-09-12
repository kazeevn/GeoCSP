"""Is the asymmetric-unit model FLOP-bound, or overhead-bound?

`flop_ratio.py` says widening 512 -> 864 multiplies edge-MLP MACs by 2.46. If wall clock
does not move by anything like that, the model was never FLOP-limited at these sizes, and
"the asymmetric unit is N times cheaper" is a claim about kernel launches and memory
traffic rather than arithmetic. That matters for reading a capacity experiment: width
bought FLOPs, but if FLOPs were free, width may not have bought anything that was scarce.

Measures forward+backward wall time per batch, solo on the GPU, against the full-cell
model at the same width for reference.
"""
import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def timed(model, batch, n_warmup: int, n_iter: int) -> float:
    """Mean seconds per forward+backward."""
    for _ in range(n_warmup):
        model(batch)["loss"].backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model(batch)["loss"].backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", default="/home/kna/DiffCSPNew/runs/bench/mp20/inits.pkl")
    ap.add_argument("--widths", type=int, nargs="+", default=[512, 864])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()

    from torch_geometric.loader import DataLoader

    from run_diffusion import DrawSet
    from diffcsp.models.cspnet import CSPNet
    from diffcsp.models.diffusion import CSPDiffusion
    from diffcsp.models.wyckoff_cspnet import WyckoffCSPNet
    from diffcsp.models.wyckoff_diffusion import WyckoffDiffusion

    dev = torch.device("cuda")
    draws = pickle.load(open(a.inits, "rb"))["draws"][: a.batch_size]
    batch = next(iter(DataLoader(DrawSet(draws), batch_size=a.batch_size, shuffle=False))).to(dev)
    print(f"batch of {batch.num_graphs} crystals, {int(batch.num_atoms.sum())} atoms\n")

    print(f"{'model':>10}{'width':>7}{'params':>13}{'s/batch':>11}{'vs asym-512':>13}")
    print("-" * 54)
    ref = None
    for h in a.widths:
        m = WyckoffDiffusion(
            device=dev, decoder=WyckoffCSPNet(hidden_dim=h, num_layers=a.num_layers)
        ).to(dev).train()
        t = timed(m, batch, a.warmup, a.iters)
        ref = ref if ref is not None else t
        p = sum(x.numel() for x in m.decoder.parameters())
        print(f"{'asym':>10}{h:>7}{p:>13,}{t:>11.4f}{t / ref:>13.2f}x")
        del m
        torch.cuda.empty_cache()

    for h in a.widths:
        m = CSPDiffusion(
            device=dev, decoder=CSPNet(hidden_dim=h, num_layers=a.num_layers)
        ).to(dev).train()
        t = timed(m, batch, a.warmup, a.iters)
        p = sum(x.numel() for x in m.decoder.parameters())
        print(f"{'full-cell':>10}{h:>7}{p:>13,}{t:>11.4f}{t / ref:>13.2f}x")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
