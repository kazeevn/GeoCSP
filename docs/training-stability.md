# Training stability: diagnostics and applied mitigations

**Dates:** 2026-09-10 to 2026-09-30  
**Code:** commits `e6fcb3d` (initial LeMat CSPNet diagnosis), `4d39843` (MP-20 `DiffCSP-Geo` plateau trapping), `561bb8f` (`GeoV2`/`GeoOrb` `AdamW` + warmup/cosine + EMA), `6dba3dc` (`EdgeBudgetBatchSampler` + per-step warmup/cosine + DDP non-finite grad skip), `bd81c4c` (`a_ehull01_lr5e4` restart).  
**W&B Runs (project `symmetry-advantage/diffcsp`):** [`mqk461a6`](https://wandb.ai/symmetry-advantage/diffcsp/runs/mqk461a6), [`7jdhrv2f`](https://wandb.ai/symmetry-advantage/diffcsp/runs/7jdhrv2f), [`say1phjr`](https://wandb.ai/symmetry-advantage/diffcsp/runs/say1phjr), [`t6l9qjp7`](https://wandb.ai/symmetry-advantage/diffcsp/runs/t6l9qjp7), [`rnj30hh2`](https://wandb.ai/symmetry-advantage/diffcsp/runs/rnj30hh2).

---

## 1. Pathological Batch Spikes on LeMat-Bulk (`mqk461a6`, 2026-09-10)

Vanilla CSPNet (512/6, 12,277,248 params) on LeMat-Bulk fmax1, filtered to
`E_hull <= 0.1 eV` and `<= 128` atoms — 1,577,517 structures, fixed batch size 256, lr $10^{-3}$,
20 epochs, 6,162 steps per epoch (commit `3d198d8` / `96f73e4`, W&B [`feasible-firefly-70` / `mqk461a6`](https://wandb.ai/symmetry-advantage/diffcsp/runs/mqk461a6)):

| epoch | train | val | | epoch | train | val |
|---|---|---|---|---|---|---|
| 0 | 0.4411 | | | 10 | 0.6850 | |
| 1 | 0.3631 | 0.3346 | | 11 | 0.7326 | 0.3161 |
| 2 | 0.3440 | | | 12 | 0.4769 | |
| 3 | 0.3314 | 0.3129 | | 13 | 0.4336 | 0.3085 |
| 4 | **0.3204** | | | 14 | 0.3233 | |
| 5 | 0.4230 | 0.3637 | | 15 | 0.3692 | 0.2992 |
| 6 | 0.8755 | | | 16 | 0.3126 | |
| 7 | **1.5624** | 0.3441 | | 17 | 0.3184 | 0.2944 |
| 8 | 0.3693 | | | 18 | 0.8123 | |
| 9 | 1.1485 | 0.3551 | | 19 | 0.3347 | **0.2902** |

Train loss reached 0.3204 by epoch 4, then spiked to 1.5624 and kept oscillating to
the end; the final epoch (0.3347) is worse than epoch 4. No NaNs, no OOM, and the
learning rate never left $1.00 \times 10^{-3}$.

Validation nonetheless improved steadily through the second half and was **still
falling at the last measurement** (0.2902, epoch 20). The divergence between a chaotic train curve and a smooth val curve traced to three factors:
1. **Heavy-tailed edge counts per fixed-size batch:** Cost per batch in the full-cell graph scales with $\sum_i N_i^2$. On LeMat-Bulk (`<= 128` atoms), cells above 32 atoms are 14.8% of structures but **73.7% of all edges**. A fixed-size batch of 256 structures drawn from the upper tail produces a much larger gradient than a typical batch.
2. **Elementwise gradient value clipping (`clip_grad_value_(0.4)`):** Bounds each parameter coordinate independently rather than bounding the total step norm.
3. **Unreachable `ReduceLROnPlateau(patience=15)` schedule:** Inherited from 1,000-epoch MP-20 runs (212 steps/epoch), `patience=15` could never fire inside a 20-epoch LeMat run (6,162 steps/epoch), and stepping it on `avg_loss` fed the noisy train spikes into the plateau detector.

---

## 2. Scheduler Trapping & Capacity Overfitting on MP-20 (`7jdhrv2f`, 2026-09-13)

The complementary risk to LeMat's failure to decay the LR was observed on MP-20 during the 500-epoch `DiffCSP-Geo` training run (commit `d7b1a7b`, W&B [`expert-wood-72` / `7jdhrv2f`](https://wandb.ai/symmetry-advantage/diffcsp/runs/7jdhrv2f); see [`docs/archive/diffcsp-geo-study.md`](archive/diffcsp-geo-study.md)):

1. **Premature LR Decay from Stepping `ReduceLROnPlateau` on Train Loss:**
   Because `scheduler.step(avg_loss)` in [`diffcsp/cli/train.py`](../diffcsp/cli/train.py#L719) tracked training loss rather than validation loss, the scheduler interpreted the natural flattening of train loss around Epoch 300 as a plateau, sequentially dropping the LR: $5\times 10^{-4} \to 3\times 10^{-4} \to 1.8\times 10^{-4} \to 1.0\times 10^{-4}$.
   However, validation loss reached its global minimum at **Epoch 320 (0.4057)** and rose steadily thereafter (**0.4373** at Epoch 500). Lowering the learning rate while validation loss was climbing trapped the model in narrow minima, accelerating training-set memorization ($0.3619 \to 0.3189$) and widening the generalization gap from $+0.0438$ to $+0.1184$.
2. **Inductive Bias vs. Parameter Scale:**
   With physical Euclidean bond vectors, Bessel RBFs, and Lie-algebra stabilizer projections, `DiffCSP-Geo` has substantially stronger inductive bias than vanilla `CSPNet` and saturated its 12.28M parameter capacity on MP-20 (~27k crystals) by **Epoch ~300 (~65,000 steps)**.

---

## 3. Mitigations Applied in `diffcsp/cli/train.py`

In commits `561bb8f` (2026-09-16), `9a4aa98` / `6dba3dc` (2026-09-29), and `bd81c4c` (2026-09-30), the following mitigations were implemented for `--model geov2`, `--model geo_orb`, and packed datasets (`--data_dir`):

1. **Edge-Budget Batching (`EdgeBudgetBatchSampler`, `diffcsp/data/packed.py`):**
   When training on packed datasets (`--data_dir` with `--max_edges_per_batch 64000`), batches are formed by bounding $\sum_i N_i^2 \le 64,000$ full-cell edges per GPU rather than fixed structure count, eliminating the heavy-tailed batch edge explosion across DDP ranks.
2. **`AdamW` + Linear Warmup + Cosine Annealing + EMA (`diffcsp/cli/train.py:345-372`):**
   - `--model geov2` and `--model geo_orb` use `AdamW` ($\text{weight\_decay}=10^{-4}$) with linear warmup followed by cosine decay to $10^{-6}$ (per-step when `--warmup_steps` is passed, or per-epoch otherwise), plus `EMAModel(decay=0.9999)` evaluated at validation and inference.
   - On MP-20 (`say1phjr`, 400 epochs, 2026-09-13), this lowered peak validation loss from $0.4057$ (`DiffCSP-Geo`) to **$0.3936$** (at epoch 300).
   - On Alex-MP-20 (`ua1g6od4`, 150 epochs, 2026-09-29), validation loss decreased monotonically to **$0.3035$** with zero overfitting.
3. **DDP-Safe Non-Finite Gradient Skipping (`diffcsp/cli/train.py:684-691`):**
   After `loss.backward()`, all DDP ranks check `torch.isfinite` on the all-reduced gradient norms and synchronously zero and skip any non-finite step without hanging collective communication.
4. **LR Scaling for Smaller Effective Batches on the LeMat $E_{\text{hull}} \le 0.1$ Slice (`bd81c4c`, 2026-09-30):**
   On LeMat-Bulk, the $E_{\text{hull}} \le 0.1\text{ eV}$ slice has larger average cells (~100 structures per 64k-edge batch vs ~160 on full LeMat). At peak $\text{lr}=10^{-3}$, `geov2_lemat_a_ehull01` (W&B [`t6l9qjp7`](https://wandb.ai/symmetry-advantage/diffcsp/runs/t6l9qjp7)) diverged at epoch 15 (train loss $0.399 \to 0.886$). Lowering peak LR to $5\times 10^{-4}$ in `geov2_lemat_a_ehull01_lr5e4` (W&B [`rnj30hh2`](https://wandb.ai/symmetry-advantage/diffcsp/runs/rnj30hh2)) converged smoothly across all 100 epochs to `val_loss = 0.3088` (see [`docs/lemat-geov2-variants.md`](lemat-geov2-variants.md)).

---

## 4. Remaining Untried Refinements

Two general training-loop refinements from the initial diagnosis have **not yet been applied** to [`diffcsp/cli/train.py`](../diffcsp/cli/train.py) and are documented in [`docs/ideas/training-and-capacity-refinements.md`](ideas/training-and-capacity-refinements.md):
- Replacing elementwise `torch.nn.utils.clip_grad_value_(params_to_train, 0.4)` (`train.py:699`) with global norm clipping `torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=0.5)`.
- Stepping `ReduceLROnPlateau` on validation loss rather than `avg_loss` (`train.py:719`) for the legacy `--model cspnet`, `--model orb`, and `--model geo` paths.
