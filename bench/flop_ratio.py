"""Does widening the asymmetric-unit model actually match vanilla's edge-MLP FLOPs?

The rationale for hidden_dim=768 was `h' = h*sqrt(N/K)`, from assuming edge-MLP cost
scales as `h^2`. It does not: the first linear consumes `2h + 6 + D` inputs where
`D = num_freqs * 2 * 3 = 768` is the sinusoidal edge featurisation, which is the same
order as `h` itself. So cost per edge is

    C(h) = h*(2h + 6 + D) + h*h = 3h^2 + h*(6 + D)

and the `h*(6+D)` term makes C grow slower than `h^2` in relative terms. This computes the
true ratio over the benchmark set's actual (K, N) distribution rather than over a mean of
ratios, and solves for the width that would actually match.
"""
import argparse
import pickle
import warnings

import numpy as np

warnings.filterwarnings("ignore")

NUM_FREQS = 128
D = NUM_FREQS * 2 * 3  # SinusoidsEmbedding.dim


def edge_mlp_macs(h: int, dis_dim: int = D) -> int:
    """MACs for one edge through CSPLayer.edge_mlp (two linears)."""
    return h * (2 * h + 6 + dis_dim) + h * h


def node_mlp_macs(h: int) -> int:
    """MACs for one node through CSPLayer.node_mlp (two linears)."""
    return h * (2 * h) + h * h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", default="/home/kna/DiffCSPNew/runs/bench/mp20/inits.pkl")
    ap.add_argument("--vanilla_hidden", type=int, default=512)
    ap.add_argument("--widths", type=int, nargs="+", default=[512, 768, 838, 1024])
    ap.add_argument("--layers", type=int, default=6)
    a = ap.parse_args()

    from pyxtal.symmetry import Group

    entries = pickle.load(open(a.inits, "rb"))["entries"]
    cache = {}
    K = np.zeros(len(entries), int)
    N = np.zeros(len(entries), int)
    for i, e in enumerate(entries):
        rep = e["rep"]
        g = cache.setdefault(rep["group"], Group(rep["group"]))
        by = {w.get_label(): w for w in g.Wyckoff_positions}
        ws = [by[label] for site_list in rep["sites"] for label in site_list]
        K[i] = len(ws)
        N[i] = sum(w.multiplicity for w in ws)

    e_vanilla = N.astype(np.float64) ** 2  # fully connected intra-cell, N^2 edges
    e_asym = K.astype(np.float64) * N  # bipartite K x N

    print(f"{len(entries)} structures: mean N={N.mean():.2f}, mean K={K.mean():.2f}")
    print(f"mean of per-structure K/N = {(K / N).mean():.4f}   "
          f"(this is what the h' = h*sqrt(N/K) rationale used)")
    print(f"edge ratio of TOTALS  sum(K*N)/sum(N*N) = {e_asym.sum() / e_vanilla.sum():.4f}   "
          f"(this is the quantity that sets compute)")
    print(f"  the two differ because K/N is correlated with N; "
          f"corr(K/N, N) = {np.corrcoef(K / N, N)[0, 1]:.3f}")

    hv = a.vanilla_hidden
    base_edge = e_vanilla.sum() * edge_mlp_macs(hv)
    base_node = N.sum() * node_mlp_macs(hv)
    print(f"\nper-edge MACs: C({hv}) = {edge_mlp_macs(hv):,}")
    print(f"\n{'width':>7}{'C(h)':>14}{'C(h)/C(hv)':>12}{'edge FLOP':>12}{'+node':>10}{'params':>14}")
    print("-" * 71)
    for h in a.widths:
        edge = e_asym.sum() * edge_mlp_macs(h)
        node = K.sum() * node_mlp_macs(h)
        # full decoder parameter count, matching WyckoffCSPNet
        p = 101 * h + (h + 256) * h + h  # embedding + atom_latent_emb(+bias)
        p += a.layers * ((2 * h + 6 + D) * h + h + h * h + h + 2 * h * h + h + h * h + h)
        p += h * 3 + h * 6
        print(f"{h:>7}{edge_mlp_macs(h):>14,}{edge_mlp_macs(h) / edge_mlp_macs(hv):>12.3f}"
              f"{edge / base_edge:>12.3f}{(edge + node) / (base_edge + base_node):>10.3f}"
              f"{p:>14,}")

    # width that matches vanilla's edge-MLP FLOPs exactly: solve C(h)*R = C(hv)
    r = e_asym.sum() / e_vanilla.sum()
    target = edge_mlp_macs(hv) / r
    # 3h^2 + (6+D)h - target = 0
    b = 6 + D
    h_star = (-b + np.sqrt(b * b + 12 * target)) / 6
    print(f"\nwidth that matches vanilla edge-MLP FLOPs exactly: h* = {h_star:.0f}")
    print(f"  (needs C(h)/C({hv}) = 1/{r:.4f} = {1 / r:.3f}x)")
    naive = hv * np.sqrt(e_vanilla.sum() / e_asym.sum())
    print(f"  the h*sqrt(N/K) rationale gives {naive:.0f}, which reaches "
          f"{e_asym.sum() * edge_mlp_macs(int(round(naive))) / base_edge:.3f}x -- "
          f"it over-credits width because it ignores the {D}-dim edge featurisation")


if __name__ == "__main__":
    main()
