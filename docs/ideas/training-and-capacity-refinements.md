# Training & Capacity Refinements

**Status:** Untried ideas (extracted from [`docs/training-stability.md`](../training-stability.md) and [`docs/diffcsp-geo-study.md`](../diffcsp-geo-study.md) §3.4).

---

## 1. Global Gradient Norm Clipping (`clip_grad_norm_`)

### Background
In [`diffcsp/cli/train.py`](../../diffcsp/cli/train.py#L699), gradient clipping is currently performed elementwise across all model types:
```python
torch.nn.utils.clip_grad_value_(params_to_train, 0.4)
```
While packed datasets now bound the total edge count per batch via `EdgeBudgetBatchSampler` (`--max_edges_per_batch 64000`, added in commit `6dba3dc`), `clip_grad_value_` bounds each parameter element independently. For a 12.28M-parameter network, every element can sit near the $0.4$ limit while the total $L_2$ gradient norm remains enormous, allowing an outlier batch (such as near-$t=0$ wrapped-normal score spikes or high-multiplicity cells) to produce a disproportionately large weight update. This manifested in:
- LeMat-Bulk CSPNet (`mqk461a6`, 2026-09-10): train loss spiked to $1.5624$ at epoch 7 while validation loss fell smoothly.
- LeMat-Bulk GeoV2 `a_ehull01` at $\text{lr}=10^{-3}$ (`t6l9qjp7`, 2026-09-29): train loss rose from $0.399$ (epoch 14) to $0.886$ (epoch 18) with no non-finite gradients.

### Proposed Change
Replace elementwise value clipping with global $L_2$ norm clipping in [`diffcsp/cli/train.py`](../../diffcsp/cli/train.py):
```python
torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=0.5)
```
so the step magnitude itself is bounded regardless of batch composition.

---

## 2. Stepping `ReduceLROnPlateau` on Validation Loss

### Background
While `geov2` and `geo_orb` now use `AdamW` with warmup + cosine annealing (`diffcsp/cli/train.py:345-372`), the default path for `cspnet`, `orb`, `geo`, and `wyckoff` without `--warmup_steps` still uses `ReduceLROnPlateau` stepped on the **training loss** (`diffcsp/cli/train.py:719`):
```python
if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    scheduler.step(avg_loss)
```
This has two opposite failure modes documented in [`docs/training-stability.md`](../training-stability.md):
1. **Noisy train spikes reset patience (LeMat `mqk461a6`):** A single high-loss batch spikes `avg_loss`, masking a real plateau.
2. **Premature LR decay traps the model in memorization (MP-20 `7jdhrv2f`):** When `DiffCSP-Geo` training loss flattened around epoch 300, `ReduceLROnPlateau` decayed the LR ($5\times 10^{-4} \to 1\times 10^{-4}$) even though validation loss had already bottomed out at epoch 320 ($0.4057$) and was rising ($0.4373$ at epoch 500).

### Proposed Change
Step `ReduceLROnPlateau` on `avg_val_loss` whenever validation runs (or migrate `cspnet`, `orb`, and `geo` to the same `AdamW` + warmup + cosine schedule + EMA used by `geov2`).

---

## 3. Compact Model Downscaling (4M–6M Parameters) on MP-20 Scale Datasets

### Background
In [`docs/diffcsp-geo-study.md`](../diffcsp-geo-study.md) §3 (2026-09-13, W&B `7jdhrv2f`), the 12.28M-parameter `DiffCSP-Geo` (`hidden_dim=512, num_layers=6`) reached peak validation performance on MP-20 (~27,136 training structures, $\approx 455$ parameters per crystal) at epoch 320 ($\sim 68,000$ steps) and overfit thereafter. Similarly, `DiffCSP-GeoV2` (`say1phjr`, 400 epochs) peaked at epoch 300 (val loss $0.3936$ vs $0.4216$ at epoch 400).

Because explicit Euclidean bond vectors, Bessel RBFs, and Lie-algebra tangent projections provide a strong physical inductive bias, the network no longer needs 12.28M parameters to implicitly reconstruct metric geometry on small datasets.

### Proposed Experiment
Benchmark a compact **4M–6M parameter** `DiffCSP-Geo` / `DiffCSP-GeoV2` configuration (`hidden_dim=384, num_layers=4`) on MP-20 for 300 epochs:
- Expected to match or exceed the 12.28M model's 88.4% best-of-3 match rate by reducing late-stage memorization, while cutting training and sampling latency by ~50%.
- Reserve the 12.28M (`512x6`) configuration for large-scale datasets such as Alex-MP-20 (608k structures) and LeMat-Bulk (1.58M–5.2M structures).
