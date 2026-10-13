# Kaggle scripts

Notebooks that implement the roadmap in [../IMPROVEMENTS.md](../IMPROVEMENTS.md), sections 2.1, 2.2 and 2.4, plus the HLS export (2.1 "reconcile configuration" and "regenerate HIL samples"). All model, data and evaluation code lives in `emg_common.py`; the notebooks only configure and call it.

## Files

| File | Purpose | Roadmap items |
|---|---|---|
| `emg_common.py` | Shared library: config, causal preprocessing, repetition split, datasets, RoFormer variants, fixed-point QAT, training, sequential evaluation with Kalman fusion and rest anchoring, numpy reference forward pass, HLS export | all |
| `01_preprocess_split.ipynb` | Leakage-free per-subject windows, train-only normalisation constants, per-stride delta targets | 2.1 split, normalisation, sequential metric |
| `02_train_roformer.ipynb` | One configurable experiment: variant, abs/delta heads, previous-angle conditioning, integration-in-the-loop, QAT; window and sequential metrics; tracking plot | 2.2 all, 2.4 training items |
| `03_ablation_grid.ipynb` | Grid over variants, heads, training modes; CSV + plots of accuracy vs MACs | 2.2 attention ablation, token reduction |
| `04_sequential_eval_fusion.ipynb` | Deployment-condition evaluation: own-estimate feedback, Kalman q/r sweep, error vs horizon, fixed-point integrator drift | 2.4 fusion, anchoring, numerics, evaluation |
| `05_export_hls.ipynb` | One checkpoint to every HLS artifact: quantised weight headers with the existing array names, RoPE tables, config.h, golden dumps on a real window, HIL samples + manifest, normalisation constants, fixed-point expectation | 2.1 reconcile configuration, regenerate HIL samples |

The two original notebooks (`roformer.ipynb`, `baseline-models-joint-angles.ipynb`) are kept for reference.

## Setup on Kaggle

1. Upload `emg_common.py` as a **Utility Script** (Kaggle: File > Add utility script, or create a new "Script" notebook named `emg_common` containing the file) and attach it to each notebook. Alternatively upload it as a dataset; the notebooks search `/kaggle/usr/lib/emg_common`, `/kaggle/input/emg-common` and the working folder.
2. Attach the NinaPro DB2 dataset. Default path: `/kaggle/input/datasets/quddusikashaf/ninapro-db2/DB2_sX/DB2_sX/*.mat`. Change `base_path` in the config cell if different.
3. Weights & Biases: add a Kaggle Secret named `wandb_key`. Never paste the key into a cell. The old baseline notebook contains a plaintext key; revoke it in your W&B account settings.
4. Run order: 01 → 02 (or 03) → 04 → 05. Each notebook finds the previous one's outputs under `/kaggle/working` in the same session, or under `/kaggle/input/*` when you save the output as a dataset and attach it.

## Main configuration knobs (`ec.Config`)

| Field | Meaning |
|---|---|
| `variant` | `roformer` (current design), `patch` (strided embedding, 25 tokens), `noattn` (ablation), `causal` (causal attention, RMSNorm, last-token pooling) |
| `heads` | `absolute`, `delta`, `both` |
| `cond_prev` | feed the previous angle estimate into the head |
| `unroll_steps` | 0 = independent windows; K>1 = integration-in-the-loop over K consecutive windows |
| `qat_epochs`, `act_fmt`, `resid_fmt`, `weight_fmt`, `rope_fmt` | fixed-point fine-tuning with `ap_fixed<W,I>` round-to-nearest, saturate. `resid_fmt` (default 16,8) is the residual stream: FFN outputs of trained models reach magnitude 100+, far outside `data_t` |
| `train_reps`, `test_reps` | repetition split (NinaPro convention 1,3,4,6 / 2,5) |
| `causal_filter` | one-directional band-pass (deployable) instead of zero-phase |
| `loss_norm`, `select_metric`, `delta_detach` | absolute error divided by per-joint train variance so abs and delta terms share a 1-R² scale; checkpoint/early-stop metric (`val_abs` default); optionally train the delta head on detached trunk features |

## Export compatibility

For `variant=roformer`, `heads=absolute`, `cond_prev=False` the exported `weights/*.h`, `rope_tables.h` and `config.h` are drop-in replacements for the files in `hls_code/`; the existing C++ layer functions reproduce the exported golden dumps at cosine 0.999 (verified with the project's float mock). Other variants require the HLS changes named by the macros written to `config.h` (`CONV_STRIDE`, `USE_ATTENTION`, `NORM_IS_RMS`, `POOL_LAST_TOKEN`, `HAS_DELTA_HEAD`, `COND_PREV`). The delta head is exported in standardised units; the firmware multiplies by `DELTA_SCALE` from `normalization_constants.h` before integrating.

## Evaluation notes

- `evaluate_sequential(..., init='abs')` seeds filters and integrators with the absolute head's first prediction, which is what the device can do. `init='truth'` seeds with the glove reading at the start of each run and is optimistic. Report `init='abs'` numbers.
- `abs_smoothed` is the same Kalman filter with zero delta input. The delta head's real contribution is `abs_smoothed` minus `kalman`, not `abs_only` minus `kalman`.
- `kalman_sweep(preds, use_delta=..., init=...)` tunes the plain fusion filter. The session-2 notebook sweep tuned the rest-anchored filter by mistake.

## Local smoke test

The notebooks honour environment variables so they can run outside Kaggle: `EMG_BASE_PATH`, `EMG_WORK`, `EMG_EPOCHS`, `EMG_N_SUBJECTS`, `EMG_WANDB=0`, `EMG_GRID_N`, `EMG_N_HIL`, `EMG_CKPT`. All five were executed in order on a synthetic DB2-shaped dataset before being committed here.
