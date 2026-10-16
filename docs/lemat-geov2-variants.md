# GeoV2 on LeMat-Bulk: stability slice, property conditioning, classifier-free guidance

**Launched:** 2026-09-29, ASPIRE 2A, one 4x A100-40GB node per run (`aiq3`, chained 24 h links).
**Code:** branch `lemat-geov2-cond`; model and training code at `6dba3dc`, initial runs at `f580b0c`, restarted slice run `a_ehull01_lr5e4` at `bd81c4c` (2026-09-30), merged to `master` at `1d8b164` (2026-10-02).
**W&B:** project `symmetry-advantage/diffcsp`, run names `geov2_lemat_<variant>`, tag `lemat`.

## Variants

| variant | W&B run | commit | train data | condition | condition dropout | lr | epochs |
|---|---|---|---|---|---|---|---|
| `a_ehull01` | [`t6l9qjp7`](https://wandb.ai/symmetry-advantage/diffcsp/runs/t6l9qjp7) | `f580b0c` | E_hull <= 0.1 eV/atom only (= `lemat_bulk_fmax1_stress_ehull01`) | none | - | 1e-3 | 100 (stopped ep 19) |
| `a_ehull01_lr5e4` | [`rnj30hh2`](https://wandb.ai/symmetry-advantage/diffcsp/runs/rnj30hh2) | `bd81c4c` | E_hull <= 0.1 eV/atom only (= `lemat_bulk_fmax1_stress_ehull01`) | none | - | 5e-4 | 100 |
| `b_ehull` | [`ionfkdur`](https://wandb.ai/symmetry-advantage/diffcsp/runs/ionfkdur) | `f580b0c` | all | `energy_above_hull` | 0 | 1e-3 | 40 |
| `c_ehull_cfg` | [`m3pgbjze`](https://wandb.ai/symmetry-advantage/diffcsp/runs/m3pgbjze) | `f580b0c` | all | `energy_above_hull` | 0.2 (CFG) | 1e-3 | 40 |
| `d_eform` | [`ey2se9um`](https://wandb.ai/symmetry-advantage/diffcsp/runs/ey2se9um) | `f580b0c` | all | `formation_energy_per_atom` | 0 | 1e-3 | 40 |
| `e_eform_cfg` | [`ehoee5sa`](https://wandb.ai/symmetry-advantage/diffcsp/runs/ehoee5sa) | `f580b0c` | all | `formation_energy_per_atom` | 0.2 (CFG) | 1e-3 | 40 |

Model: DiffCSP-GeoV2, 512x6, EMA 0.9999 — the same architecture as the Alex-MP-20 run
([`ua1g6od4`](https://wandb.ai/symmetry-advantage/diffcsp/runs/ua1g6od4), commit `f9e3b3a`, 2026-09-29, 150 epochs, best val loss 0.3035 at epoch 140). On MP-20 (`561bb8f`, 2026-09-13, W&B [`say1phjr`](https://wandb.ai/symmetry-advantage/diffcsp/runs/say1phjr)), DiffCSP-GeoV2 reached a lower validation loss than DiffCSP-Geo (0.3936 vs 0.4057), 82.6% per-trial / 87.6% best-of-3 match rate single-sample (85.7% / 89.1% with $M=3$ ORB ranking), and lower clash rate (0.7% vs 1.5%) and cell-volume error (3.26% vs 3.90%); see [`docs/diffcsp-geo-study.md`](diffcsp-geo-study.md).

## Data

`WyFormer/data/lemat_bulk_fmax1_stress/{train,val,test}.csv.gz` (the CIFs behind the
WyFormer cache `lemat_bulk_fmax1_stress`), packed by `scripts/platforms/aspire2a/pack_dataset.pbs`
(job 25629806.pbs101) into `DiffCSPNew/cache/lemat_bulk_fmax1_stress_packed`. Each split's
`meta.json` records the source sha256 and the packing commit.

- val and test are identical, id for id, to WyFormer's `split_ids.json` (100,000 each), and
  their E_hull <= 0.1 rows to the `_ehull01` slice's (30,543 / 30,818). Variant a filters by
  the same rule, so it trains on exactly the WyFormer slice.
- Cells above 128 atoms are dropped (0.44% of train, 0.37% of the slice): the full-cell
  graph costs N^2 edges.
- Model selection: a fixed random 20k subset of val (seed 0), E_hull-filtered for variant a.
  The b-e runs also log `val_loss_ehull0.1` on val's stable structures, the same population
  as a's validation, and the CFG runs log `*_uncond` losses with the condition nulled.
  test is not touched.

## Training

- Batches of <= 64,000 full-cell edges (sum N^2) per GPU, ~160 structures at LeMat's mean;
  every rank runs the same number of steps. One A100 sustains ~290k edges/s above ~40k
  edges per batch, so larger budgets buy nothing.
- AdamW, lr 1e-3 (5e-4 for `a_ehull01_lr5e4`), weight decay 1e-4, per-step linear warmup (3000 steps), cosine to 1e-6;
  gradient value clipping 0.4 (unchanged from GeoV2).
- Conditioning: each property is clamped (E_hull 0..3, E_form -5..4 eV/atom), expanded on
  64 Gaussians, passed through a zero-initialised MLP and added to the time embedding; a
  learned null token stands for a dropped condition. At init a conditional model equals
  its unconditional counterpart.
- Sampling: `uv run python -m diffcsp.cli.inference genes.json.gz --model geov2 --ckpt_path <ckpt>
  --condition energy_above_hull=0 --guidance_scale <w>`; w = 1 is conditional, w = 0
  unconditional, w > 1 stronger guidance (b and d support w = 1 only).

## Pilot

W&B [`gwjo8lgm`](https://wandb.ai/symmetry-advantage/diffcsp/runs/gwjo8lgm) (`pilot_geov2_lemat_1gpu`), 2026-09-29, commit `6dba3dc`: the c configuration on one
A100, trained on packed val and validated on packed test. 711 structures/s, peak 6.4 GiB,
no skipped steps. Projected on 4 GPUs: ~32 min/epoch on full LeMat, ~14 min on the slice,
so ~21-23 h per run. The pilot ran one rank, so the 4-rank DDP path was first exercised by
the real runs.

## Restart of variant a (2026-09-30)

`a_ehull01` (W&B [`t6l9qjp7`](https://wandb.ai/symmetry-advantage/diffcsp/runs/t6l9qjp7)) diverged: train loss fell to 0.399 at epoch 14, then rose to
0.43, 0.42, 0.56 and 0.89 by epoch 18, with no non-finite steps, at lr ~9.3e-4. Best
checkpoint: epoch 12, val loss 0.3824 (kept in `runs/diffcsp/geov2_lemat_a_ehull01/`). The run
was stopped at epoch 19 and restarted from scratch as `a_ehull01_lr5e4` (W&B [`rnj30hh2`](https://wandb.ai/symmetry-advantage/diffcsp/runs/rnj30hh2)): identical except
peak lr 5e-4 (the Alex-MP-20 GeoV2 value), from branch `lemat-geov2-a-lr5e4` (`bd81c4c`). The slice's
larger cells mean ~100 structures per GPU batch against ~160 for the full data, i.e. noisier
gradients at the same lr. The b-e runs stayed at 1e-3.

## Training Results (Completed)

All five production runs completed without non-finite step skips (~6.5 GiB peak memory per A100):

| variant | W&B run | epochs completed | train loss (final) | `val_loss` (best / final) | `val_loss_uncond` | `val_loss_ehull0.1` | `val_loss_ehull0.1_uncond` | throughput (struct/s, 4x A100) |
|---|---|---|---|---|---|---|---|---|
| `a_ehull01` (diverged) | [`t6l9qjp7`](https://wandb.ai/symmetry-advantage/diffcsp/runs/t6l9qjp7) | 19 / 100 | 0.8856 | 0.3824 (ep 12) / 0.3853 | — | 0.3824 (ep 12) | — | 1,828 |
| `a_ehull01_lr5e4` | [`rnj30hh2`](https://wandb.ai/symmetry-advantage/diffcsp/runs/rnj30hh2) | 100 / 100 | **0.3098** | **0.3088** (ep 100) | — | **0.3088** (ep 100) | — | 1,788 |
| `b_ehull` | [`ionfkdur`](https://wandb.ai/symmetry-advantage/diffcsp/runs/ionfkdur) | 40 / 40 | 0.3630 | 0.3602 (ep 40) | — | 0.3400 | — | 2,777 |
| `c_ehull_cfg` | [`m3pgbjze`](https://wandb.ai/symmetry-advantage/diffcsp/runs/m3pgbjze) | 40 / 40 | 0.3652 | 0.3609 (ep 40) | 0.3632 | 0.3416 | 0.3431 | 2,760 |
| `d_eform` | [`ey2se9um`](https://wandb.ai/symmetry-advantage/diffcsp/runs/ey2se9um) | 40 / 40 | 0.3620 | 0.3589 (ep 40) | — | 0.3397 | — | 2,779 |
| `e_eform_cfg` | [`ehoee5sa`](https://wandb.ai/symmetry-advantage/diffcsp/runs/ehoee5sa) | 40 / 40 | 0.3690 | 0.3655 (ep 40) | 0.3668 | 0.3454 | 0.3461 | 2,755 |

