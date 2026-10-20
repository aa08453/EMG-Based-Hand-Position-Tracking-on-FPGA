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
| `05_export_hls.ipynb` | One checkpoint to every HLS artifact: quantised weight headers (encoder layers stacked), RoPE tables, `model_config.h`, golden float and fixed-point dumps on a real window, HIL samples + manifest, normalisation constants, fixed-point expectation | 2.1 reconcile configuration, regenerate HIL samples |
| `06_subject_ci_and_loso.ipynb` | Session 4: per-subject cluster-bootstrap confidence intervals for the session-2 model, and resumable 5-fold leave-subjects-out training and evaluation | 2.1 sequential metric, generalisation to new users |
| `07_window_200ms.ipynb` | 200 ms windows with a 50 ms step: preprocesses once, trains three resumable models (100 ms control on the same test windows, 200 ms at 25 tokens, 200 ms at 50 tokens), re-tunes the Kalman filter, reports pooled and per-subject bootstrap results | Window-length study |
| `08_model_scaling.ipynb` | Model-size study on the chosen 200 ms / 25-token setup: more layers, wider FFN, wider model, a second-seed baseline as the noise reference, with hardware cost (weights vs Zynq-7020 BRAM, latency) for each | Capacity vs deployability |
| `09_qat_export_deep.ipynb` | Deep models to hardware: schedule check (patience 12 retrains of base and L2), post-training quantisation baseline, QAT of L2 (10- and 12-bit activations) and L3 (8-bit weights), deployable evaluation of each, export of the best model that fits on chip, with the firmware fusion gain | Deployable deep model |

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

The HLS in `hls_code/` implements any number of post-norm encoder layers with attention, LayerNorm, mean pooling, an absolute head and an optional delta head, at `d_model` 128 with 8 heads; the patch length (window samples / tokens) and `d_ff` are free. `ec.hls_unsupported(cfg)` lists what a configuration violates, `export_hls` refuses such models, and `hw_cost` reports `hls_supported`.

Install an export with `python hls_code/scripts/install_export.py <export dir>` (repository root), then `cd hls_code && make run`. It replaces `model_config.h`, `rope_tables.h`, `weights/`, `verification_data/`, `hil_verification_samples/` and the firmware's `normalization_constants.h`. The C simulation checks every layer against the numpy fixed-point simulation written by the export (`fx_*` in `results.h`, `sample_*_fixed.txt`); the float-model gap is printed alongside. Exports made before the multi-layer HLS (with `config.h` and `l0_*.h`) cannot be installed; re-export the checkpoint.

The fixed-point formats travel with the export (`ACT_W/I`, `WEIGHT_W/I`, `RESID_W/I`, `ROPE_W/I` in `model_config.h`), so a model QAT-trained with 12-bit activations or 8-bit weights needs no HLS edit. The delta head is exported in standardised units; the firmware multiplies by `DELTA_SCALE` from `normalization_constants.h` before integrating, and uses `MODEL_FUSION_GAIN` when the export was given `fusion_gain=ec.kalman_gain(q, r)`.

## Evaluation notes

- `ordered_runs` used to index the compressed `.npz` inside its loops, which decompressed whole arrays once per window; sequential evaluation took about 55 minutes per model in notebook 08. It now reads each array once (identical output).

- `evaluate_sequential(..., init='abs')` seeds filters and integrators with the absolute head's first prediction, which is what the device can do. `init='truth'` seeds with the glove reading at the start of each run and is optimistic. Report `init='abs'` numbers.
- `abs_smoothed` is the same Kalman filter with zero delta input. The delta head's real contribution is `abs_smoothed` minus `kalman`, not `abs_only` minus `kalman`.
- `kalman_sweep(preds, use_delta=..., init=...)` tunes the plain fusion filter. The session-2 notebook sweep tuned the rest-anchored filter by mistake.

## Local smoke test

The notebooks honour environment variables so they can run outside Kaggle: `EMG_BASE_PATH`, `EMG_WORK`, `EMG_EPOCHS`, `EMG_N_SUBJECTS`, `EMG_WANDB=0`, `EMG_GRID_N`, `EMG_N_HIL`, `EMG_CKPT`. All five were executed in order on a synthetic DB2-shaped dataset before being committed here.
