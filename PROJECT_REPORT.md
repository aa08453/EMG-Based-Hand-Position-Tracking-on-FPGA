# sEMG-to-Finger-Angle Estimation on FPGA: Project Report

Status as of 15 October 2026, including sessions 4 and 5. This document records every approach tried so far, how data is processed, how the models are built and evaluated, what each costs in compute and memory, and what remains. It is written for someone joining the project with no prior context.

---

## 1. Summary

The project estimates 10 finger-joint angles from 12 channels of surface EMG (sEMG) using a small Transformer, with the goal of running inference on a Zynq-7020 FPGA (ZedBoard) in real time.

| Item | Current state |
|---|---|
| Chosen model | Patch-embedding RoFormer, 1 encoder layer, absolute-angle head plus angle-delta head, 147,476 parameters |
| Compute per inference | 3.75 M multiply-accumulates (MACs), 10× fewer than the original design |
| Post-processing | Per-joint constant-gain Kalman filter fusing the two heads |
| Deployable accuracy | Pooled over 10 subjects: R² 0.668 [0.650, 0.683], RMSE 0.103 (normalised units), repetition-held-out split. Per subject: R² 0.44 to 0.62, mean 0.52 |
| New users | Not supported yet: leave-subjects-out RMSE doubles to 0.21 and R² is negative for 9 of 10 subjects |
| Longer window | 200 ms windows give a small, consistent gain over 100 ms at the same 50 ms step: RMSE −0.0023, better in 9 of 10 subjects |
| Fixed-point export | Verified: existing HLS layer functions reproduce the golden model at cosine ≥ 0.997 |
| Hardware | HLS rewritten for the chosen model and verified bit-accurately in C simulation (all layers cosine ≥ 0.998). Estimated 9.3 ms per inference against a 25 ms budget. Not yet synthesised |

The single most important lesson so far: two early results that looked excellent (R² 0.71 and R² 0.99) were artefacts of the evaluation, not real skill. Section 9 explains both, so they are not repeated.

---

## 2. Problem and dataset

**Task.** Given a 100 ms window of sEMG, predict the hand's finger-joint angles at the end of the window. This is continuous regression, not gesture classification.

**Dataset.** NinaPro DB2.

| Property | Value |
|---|---|
| EMG | 12 Delsys electrodes, 2 kHz |
| Kinematics | 22-sensor CyberGlove II, sampled with the EMG |
| Exercises used | E1 and E2 (finger and hand movements). E3 (force patterns) is excluded |
| Repetitions | Each movement repeated 6 times, separated by rest |
| Labels used | 10 of the 22 glove sensors, columns 1, 2, 4, 5, 7, 8, 11, 12, 15, 16 (0-based) |
| Subjects used | 10: DB2_s1 and DB2_s10 to DB2_s18 |

The subject list is "the first 10 subjects in lexicographic order", which is why it skips s2 to s9. The joint names attached to the 10 columns in the code (`T_CMC_ABD`, `T_MCP_FLEX`, …) were carried over from an earlier notebook and have not been checked against the CyberGlove sensor map.

All accuracy figures are in **normalised angle units**: each joint is scaled to 0 to 1 using its training-set minimum and maximum for that subject. Converting to degrees requires per-subject glove calibration, which has not been done.

---

## 3. Timeline of approaches

| Phase | What was done | Outcome |
|---|---|---|
| 0. Original work | RoFormer trained in a Kaggle notebook. Baseline MLP, CNN, LSTM, ConvLSTM in a second notebook. HLS implementation of the RoFormer, exported as Vivado IP, tested over UART | Reported window R² 0.71. Hardware functional |
| 1. Review (Oct 4) | Replayed the model in numpy and the HLS code with a float mock. Simulated the fixed-point format on real windows. Audited the data pipeline | Found data leakage, a no-op loss term, stale verification dumps, mismatched configurations and an undersized number format. See [IMPROVEMENTS.md](IMPROVEMENTS.md) |
| 2. Rebuild (Oct 4–5) | Wrote [kaggle_scripts/emg_common.py](kaggle_scripts/emg_common.py) and five notebooks: leakage-free preprocessing, configurable model, ablation grid, sequential evaluation, HLS export | Tooling verified end to end on synthetic data |
| 3. Session 1 (Oct 7) | Preprocessed 10 subjects. Trained one dual-head model with QAT. Ran a 7-configuration ablation grid | Honest baseline R² 0.57. Patch embedding costs almost nothing. Previous-angle conditioning and unrolled training fail. Fixed-point overflow found in the residual stream |
| 4. Fixes (Oct 12) | Loss rebalanced between heads, checkpoint selection on the absolute head, residual stream given a 16-bit format, QAT made to match the HLS quantisation points | Verified on real checkpoints |
| 5. Session 2 (Oct 12) | Trained the chosen configuration with QAT and exported to HLS | R² 0.673 with true-angle seed. Export verified |
| 6. Session 3 (Oct 13) | Evaluation only: deployable seeding and a smoothing-only baseline | R² 0.667 deployable. 83 % of the fusion gain is smoothing, 17 % the delta head |
| 7. HLS v2 (Oct 15) | HLS rewritten for the chosen model: strided embedding, 16-bit residual type, second head, flattened and unrolled MAC loops, rounding and saturation. Bit-accurate C simulation using Xilinx's open-source `ap_fixed` headers. Firmware fuses the two heads | C model matches the numpy fixed-point reference to 0.0004. Estimated 9.3 ms per inference. Old HLS code with the new weights scored cosine −0.06 bit-accurately |
| 8. Session 4 (Oct 15) | Notebook 06: per-subject bootstrap confidence intervals, 5-fold leave-subjects-out | Delta-head gain holds in 8 of 10 subjects. The model does not generalise to unseen subjects |
| 9. Session 5 (Oct 15) | Notebook 07: 200 ms windows with a 50 ms step, against a 100 ms control scored on the same test windows | 200 ms with 25 tokens: R² 0.688 against 0.678 for the control. 200 ms with 50 tokens: no gain |

---

## 4. Data processing

### 4.1 Current pipeline

Implemented in `process_subject` in [emg_common.py](kaggle_scripts/emg_common.py), notebook section 01.

1. **Load** E1 and E2 `.mat` files per subject: `emg`, `glove`, `restimulus` (movement label, 0 = rest) and `rerepetition` (repetition index).
2. **Label rest samples with a repetition.** Rest samples carry repetition 0. They are forward-filled from the preceding movement so each rest period belongs to the repetition before it.
3. **Filter EMG causally.** 4th-order Butterworth band-pass, 5–500 Hz, applied one-directionally (`sosfilt`). Then full-wave rectification.
4. **Split by repetition.** Repetitions 1, 3, 4, 6 are training; repetitions 2 and 5 are test. This is the standard NinaPro protocol.
5. **Normalise EMG** using training repetitions only: divide each channel by its 99.9th percentile, clip to 0–1, then apply μ-law companding with μ = 255.
6. **Normalise glove** using training repetitions only: per-joint min-max to 0–1.
7. **Window.** 200 samples (100 ms) with a 50-sample (25 ms) hop. Any window that straddles a train/test boundary is dropped.
8. **Targets per window.**
   - `angles`: joint angles at the last sample of the window.
   - `angles_prev`: angles at the last sample of the previous window.
   - `delta_stride`: the difference between those two, which is the 25 ms motion.
   - Bookkeeping fields: `restim`, `rep`, `segment`, `t_index`, `split`, `has_prev`. These allow test windows to be replayed in time order.
9. **Save** `db2_processed/<subject>/data.npz` and `norm_stats.json`, holding the constants the firmware needs.

| Dataset statistic | Value |
|---|---|
| Windows per subject | about 86,400 |
| Train / test per subject | about 58,700 / 27,600 |
| Per-joint std of the 25 ms delta | 0.0056 to 0.0061 |
| Resolution of the 10-bit HLS format | 0.0078 |

The delta is smaller than one step of the 10-bit hardware format. Delta targets are therefore standardised by their per-joint standard deviation (`delta_scale`) before training and export, and rescaled in firmware.

**Training-set balancing.** Rest windows outnumber any single movement. In the training set only, rest windows are randomly capped at the median count of the movement classes. The test set is never balanced.

### 4.2 What changed from the original pipeline, and why

| Step | Original notebook | Current | Reason |
|---|---|---|---|
| Train/test split | First 80 % of each movement class's windows train, last 20 % test | Whole repetitions held out | With 87.5 % window overlap, adjacent train and test windows were near-duplicates. This inflated R² |
| Band-pass | Zero-phase (`sosfiltfilt`) | Causal (`sosfilt`) | Zero-phase filtering uses future samples, which the device cannot |
| EMG scaling | Max-abs over the whole recording | 99.9th percentile of training reps | Whole-recording statistics leak test data and are unavailable in real time |
| Glove scaling | Min-max over the whole recording | Min-max over training reps | Same leakage |
| Window length | 400 samples in the notebook, 200 in HLS | 200 everywhere | The notebook and the hardware disagreed |
| Rest balancing | Capped to the count of movement 1 | Capped to the median movement count | Minor robustness change |

---

## 5. Model architecture

### 5.1 Original model

From [notebooks/roformer.ipynb](notebooks/roformer.ipynb) and the HLS code in [hls_code/](hls_code/).

```
EMG [12 ch × T]
  → Conv1d(12→128, kernel 3, stride 1, pad 1)      T tokens × 128
  → RoPE applied to the full 128-dim token          (queries and keys only)
  → Multi-head self-attention (8 heads × 16 in HLS; 4 heads in the notebook)
  → Add + LayerNorm
  → FFN 128 → 256 → 128, ReLU
  → Add + LayerNorm
  → Mean over tokens → Linear(128→10)
```

T was 400 in the notebook and 200 in hardware. The notebook used PyTorch's built-in attention module with a `VelocityGuidedLoss`. That loss simplifies algebraically to 1.1 × MSE, so its velocity term had no effect.

### 5.2 Current model

`EMGRoFormer` in [emg_common.py](kaggle_scripts/emg_common.py).

```
EMG [12 ch × 200 samples]
  → Conv1d(12→128, kernel 8, stride 8, no pad)     25 tokens × 128   ("patch embedding")
  → RoPE on full token (rotation before Q/K projection, matches HLS)
  → MHA: 8 heads × 16 dims, explicit Q/K/V/O linear layers
  → Add + LayerNorm                (post-norm, as in HLS)
  → FFN 128 → 256 → 128, ReLU
  → Add + LayerNorm
  → Mean over 25 tokens            → feature [128]
  → Absolute head: Linear(128→10)                    angles at window end
  → Delta head:    Linear(128→10) × delta_scale      25 ms change in angles
```

| Property | Value |
|---|---|
| Parameters | 147,476 |
| MACs per inference | 3,746,560: 0.31 M embedding, 1.80 M attention, 1.64 M FFN, 2,560 heads |
| Dropout | 0.1 |

### 5.3 Variants explored

All are selectable through `Config.variant`.

| Variant | Change from the current model | Purpose |
|---|---|---|
| `roformer` | Kernel 3, stride 1, 200 tokens | The original design |
| `patch` | Kernel 8, stride 8, 25 tokens | Cut compute 10× |
| `noattn` | `patch` with attention removed | Test whether attention earns its cost |
| `causal` | `patch` with causal mask, RMSNorm, last-token pooling | Streaming-friendly, could reuse a key/value cache |

Further options:

| Option | Meaning |
|---|---|
| `heads` | `absolute`, `delta` or `both` |
| `cond_prev` | Add a projection of the previous angle estimate to the feature. Rejected, see section 9 |
| `unroll_steps=K` | Train on K consecutive windows, feeding back the model's own integrated estimate. Rejected |
| `delta_detach` | Train the delta head on detached features so it cannot affect the shared trunk. Available, not yet needed |

---

## 6. Training

| Setting | Value |
|---|---|
| Optimiser | AdamW, learning rate 3e-4, weight decay 1e-4 |
| Batch size | 64 |
| Schedule | Reduce LR ×0.5 after 3 epochs without improvement. Gradient clip 1.0 |
| Early stopping | Patience 8 epochs on validation absolute-head loss. At most 40 epochs |
| QAT | 5 extra epochs at LR 5e-5 with fake fixed-point quantisation |
| Hardware | Kaggle, one NVIDIA T4 |

**Loss for the absolute head.** Mean squared error divided by each joint's training-set variance. This puts it on a "1 − R²" scale.

**Loss for the delta head.** Mean squared error on standardised deltas, which is also on a "1 − R²" scale. Windows with no valid previous window are masked out.

**Why the normalisation matters.** Before it was added, absolute MSE was about 0.014 and standardised-delta MSE about 0.8. The total loss was therefore 98 % delta term. Early stopping, the LR schedule and checkpoint selection all followed the harder delta head, and the absolute head degraded during training.

**Quantisation-aware training (QAT).** Weights and activations pass through a fake-quantiser `ap_fixed<W, I>` with round-to-nearest and saturation. A straight-through estimator carries gradients past the rounding. Quantisation is applied at exactly the points where the HLS casts:

| Signal | Format |
|---|---|
| Inputs, linear inputs, norm outputs, pooled feature, head outputs | `ap_fixed<10,3>`: range ±4, step 1/128 |
| Branch outputs and pre-norm residual sums | `ap_fixed<16,8>`: range ±128, step 1/256 |
| Weights | `ap_fixed<10,3>` |
| RoPE tables | `ap_fixed<8,2>` |

The 16-bit residual format was added after session 1. Trained FFN outputs reach a magnitude of 127, far beyond the ±4 range of the 10-bit type. Squeezing them into it drops the cosine similarity after the second LayerNorm from 0.998 to 0.52.

---

## 7. Post-processing

The delta head predicts motion over each 25 ms hop. Integrating it alone drifts, so it is fused with the absolute head in a per-joint scalar Kalman filter:

```
predict:  x ← x + delta;            P ← P + q
correct:  K ← P / (P + r);          x ← x + K (abs − x);   P ← (1 − K) P
```

**The filter reduces to a constant gain.** The sweeps show that RMSE depends only on the ratio q/r. At steady state the filter is therefore a constant-gain blend:

```c
est[j] += delta[j] * DELTA_SCALE[j];
est[j] += 0.056f * (abs_pred[j] - est[j]);   // tuned: q = 1e-5, r = 3e-3
```

That is two additions and one multiplication per joint per 25 ms, cheap enough for the ARM core.

**Initialisation.** The filter starts from the absolute head's first prediction. Starting from a glove reading is optimistic and is used only as a comparison.

**Other estimators evaluated:**

| Estimator | Description |
|---|---|
| Absolute head only | No post-processing |
| Smoothed absolute head | The same filter with zero delta input, which is an exponential smoother. The best gain was about 0.095 |
| Plain integration of deltas | No anchoring |
| Leaky, clamped integration | Integration that decays toward a prior and is limited to valid joint range |
| Fixed-point integration | 7-bit truncation versus 12-bit rounding |
| Kalman with rest anchoring | Also pulls toward a pooled rest pose when EMG energy is low |

---

## 8. Evaluation methodology

Two evaluations are run on the held-out repetitions.

1. **Window-level.** Every test window is predicted independently and R² and RMSE are computed over all windows. Valid only for models that do not receive any past angle as input.
2. **Sequential.** Test windows are replayed in time order as contiguous runs, about 250 to 320 windows each, or 6 to 8 seconds. The model receives only what it would have on the device: EMG, plus its own previous estimate if it uses one. All post-processing estimators are scored on these runs. Errors are also reported against time since the run started, to expose drift.

**Sequential evaluation with the model's own seed is the only metric that reflects deployment.** Every headline number in this report uses it unless stated otherwise.

---

## 9. Results

### 9.1 Legacy results, leaky split

Kept for history. Do not compare these with section 9.2.

| Model | Setup | Params | MACs | Window R² |
|---|---|---|---|---|
| RoFormer (original notebook) | 3 subjects, 400-sample window, 4 heads | about 139 k | 95.2 M | 0.71 |
| MLP | 3 subjects, 200-sample window | 648,842 | 0.65 M | 0.627 |
| 1D CNN | as above | 29,898 | 3.23 M | 0.564 |
| LSTM, 2 layers | as above | 206,090 | 40.6 M | 0.634 |
| ConvLSTM | as above | 102,986 | 10.3 M | 0.627 |
| Any baseline, "diff" target | true start angle + predicted window change | | | 0.98 |

The "diff" models scored R² 0.98 because the evaluation added the predicted change to the *true* starting angle. The change over 100 ms is tiny, so the score mostly reflects that true angle. Run sequentially on subject 1 without shuffling, the original RoFormer checkpoint gave R² −0.91 and RMSE 17.3°. That evaluation may also have used inconsistent normalisation constants, so treat it as indicative only.

### 9.2 Session 1 ablation grid

10 subjects, repetition split, 15 epochs maximum, float training.

| Configuration | Params | MACs | Sequential RMSE | Sequential R² |
|---|---|---|---|---|
| roformer, absolute head | 138,506 | 37.4 M | 0.1172 | 0.573 |
| patch, absolute head | 146,186 | 3.75 M | 0.1188 | 0.561 |
| noattn, absolute head | 79,882 | 1.95 M | 0.1282 | 0.490 |
| patch, both heads, prev-conditioned | 148,884 | 3.75 M | 0.1221 (Kalman) | 0.536 |
| patch, both, prev-conditioned, unrolled K=8 | 148,884 | 3.75 M | 0.1432 (Kalman) | 0.328 |
| noattn, both, prev-conditioned, unrolled K=8 | 82,580 | 1.95 M | 0.1666 (rest-anchored) | 0.119 |
| causal, both, prev-conditioned, unrolled K=8 | 148,628 | 3.75 M | 0.1462 (Kalman) | 0.302 |

Conclusions:

- **Patch embedding is adopted.** It costs 0.012 R² for a 10× MAC reduction.
- **Attention is kept.** Removing it costs 0.07 R² on real data.
- **Previous-angle conditioning is rejected.** These models scored window R² 0.99 because the evaluation fed them the *true* angle from 25 ms earlier, which differs from the target by about 0.006, so they learned to copy it. Fed their own estimate, they fell back to R² 0.53, no better than models without the input. This is called teacher forcing; the gap is exposure bias.
- **Unrolled training is rejected.** The models learned to lean on the fed-back estimate.

### 9.3 Session 2: chosen configuration

`patch`, both heads, no previous-angle conditioning, normalised loss, QAT. Checkpoint `patch_both_prev0_K0_qat5_v2.pt`.

| Metric | Value |
|---|---|
| Best float epoch | 15. Early stop at 23, then 5 QAT epochs |
| Window absolute head | R² 0.555, RMSE 0.120 |
| Window delta head | R² 0.149, RMSE 0.0055 |
| Sequential, Kalman, true-angle seed | R² 0.673, RMSE 0.1024 |

### 9.4 Session 3: deployable evaluation

Same checkpoint, sequential, seeded from the model's own first prediction.

| Estimator | RMSE | R² |
|---|---|---|
| **Kalman fusion of both heads** | **0.1034** | **0.667** |
| Smoothed absolute head | 0.1062 | 0.648 |
| Absolute head only | 0.1200 | 0.555 |
| Kalman with rest anchoring | 0.1334 | 0.457 |
| Leaky, clamped delta integration | 0.1563 | 0.241 |
| Plain delta integration, float or 12-bit rounding | 0.1929 | −0.198 |
| Delta integration, 7-bit truncation | 0.7723 | −18.2 |

With each filter tuned separately, fusion reaches RMSE 0.1031 and smoothing alone 0.1064.

Conclusions:

- **Most of the gain is smoothing.** Of the 0.017 RMSE improvement over the raw absolute head, 0.014 comes from smoothing and 0.003 from the delta head.
- **The delta head's gain appears only in long runs.** It shows up after about 80 steps (2 s). Earlier, smoothing alone is equal or slightly better.
- **The seeding question is settled.** Seeding with the true angle changed the result by only 0.001.
- **Rest anchoring hurts.** A rest pose pooled across subjects is wrong for each individual subject.
- **Truncation makes an integrator unusable.** Any integration in hardware must round and use a wide accumulator.

**Reportable claim:** R² 0.67, RMSE 0.103 in normalised units, on 10 NinaPro DB2 subjects pooled, with repetitions 2 and 5 held out, at 3.75 M MACs per inference. Two qualifiers must always accompany it:

- The split is within-subject. The same people are in training and test, so this does not demonstrate generalisation to new users.
- RMSE is not in degrees.

### 9.5 Session 4: per subject, with confidence intervals

The session-2 model, evaluated sequentially and seeded from the model, per subject. Intervals are 95 % cluster bootstraps that resample whole test runs, 80 runs per subject.

| Subject | Absolute head R² | Smoothed R² | Fused R² [95 % CI] | Fused RMSE | Delta-head gain [95 % CI] |
|---|---|---|---|---|---|
| DB2_s1 | 0.459 | 0.570 | 0.620 [0.585, 0.647] | 0.109 | 0.0087 [0.0069, 0.0106] |
| DB2_s10 | 0.431 | 0.532 | 0.551 [0.488, 0.596] | 0.104 | 0.0027 [0.0015, 0.0038] |
| DB2_s11 | 0.257 | 0.427 | 0.442 [0.292, 0.523] | 0.084 | 0.0016 [0.0002, 0.0031] |
| DB2_s12 | 0.367 | 0.513 | 0.538 [0.472, 0.589] | 0.095 | 0.0029 [0.0015, 0.0044] |
| DB2_s13 | 0.385 | 0.542 | 0.579 [0.526, 0.616] | 0.104 | 0.0053 [0.0030, 0.0076] |
| DB2_s14 | 0.276 | 0.431 | 0.453 [0.362, 0.534] | 0.092 | 0.0020 [0.0010, 0.0032] |
| DB2_s15 | 0.325 | 0.465 | 0.459 [0.382, 0.519] | 0.112 | −0.0002 [−0.0018, 0.0013] |
| DB2_s16 | 0.315 | 0.442 | 0.466 [0.413, 0.506] | 0.109 | 0.0026 [0.0014, 0.0038] |
| DB2_s17 | 0.471 | 0.595 | 0.600 [0.548, 0.637] | 0.101 | 0.0009 [−0.0004, 0.0022] |
| DB2_s18 | 0.402 | 0.499 | 0.534 [0.493, 0.568] | 0.119 | 0.0051 [0.0039, 0.0063] |
| **Pooled** | 0.555 | 0.648 | **0.668 [0.650, 0.683]** | **0.103** | **0.0032 [0.0027, 0.0038]** |

Conclusions:

- **The delta head's gain is real but small.** It is positive in 9 of 10 subjects, and its interval excludes zero in 8. Smoothing contributes about four times as much, 0.0136 RMSE.
- **Pooled R² is higher than any single subject's R².** RMSE is about 0.10 in both views. Pooled R² divides by the variance of all subjects together, which includes differences between subjects. Per-subject R² divides by one subject's variance only. Quote per-subject R², mean 0.52 ± 0.07, beside the pooled figure.

### 9.6 Session 4: leave-subjects-out

Five folds. Each trains on 8 subjects and tests on the held-out repetitions of the other 2. The held-out subjects are normalised with their own training repetitions, as a short calibration recording would allow, but the model is not retrained.

| | Within-subject | Leave-subjects-out |
|---|---|---|
| Fused RMSE, mean over subjects | 0.103 | 0.210 |
| Fused R², mean over subjects | 0.52 | −1.21 |
| Subjects with R² > 0 | 10 of 10 | 1 of 10 |

The model does not transfer to unseen subjects. Negative R² means it is worse than predicting each subject's average pose. This is common for sEMG without per-user adaptation, because electrode placement, anatomy and glove fit differ between people. Fine-tuning on a few repetitions from the new user is the standard remedy and has not been tried yet.

### 9.7 Session 5: 200 ms windows

New preprocessing: 400-sample (200 ms) windows with a 100-sample (50 ms) step, about 42,900 windows per subject. All three models are float, with no QAT, and have the Kalman filter re-tuned for the 50 ms step. The control uses the last 100 ms of each window, so every model is scored on windows ending at the same samples.

| Model | Window | Tokens | MACs | Absolute head R² | Fused R² [95 % CI] | Fused RMSE |
|---|---|---|---|---|---|---|
| Control | 100 ms | 25 | 3.75 M | 0.568 | 0.678 [0.659, 0.692] | 0.1015 |
| **`w200ms_patch16_t25`** | **200 ms** | **25** | **4.05 M** | **0.603** | **0.688 [0.669, 0.703]** | **0.0993** |
| `w200ms_patch8_t50` | 200 ms | 50 | 7.81 M | 0.598 | 0.667 [0.649, 0.683] | 0.1030 |

Paired against the control, subject by subject:

| Model | Mean RMSE change | Subjects improved | 95 % CI of the mean change |
|---|---|---|---|
| `w200ms_patch16_t25` | −0.0023 | 9 of 10 | [−0.0035, −0.0011] |
| `w200ms_patch8_t50` | +0.0014 | 3 of 10 | [−0.0006, +0.0033] |

The interval comes from resampling subjects.

Conclusions:

- **A longer window helps the absolute head most.** Its R² rises by 0.035, but fusion already integrates over time, so the fused gain is smaller: 2.3 % RMSE.
- **The gain is consistent across subjects.** The pooled intervals overlap, but the paired comparison is decisive.
- **Finer tokens do not help.** The 50-token model doubles the compute for no benefit.
- **Hardware cost is small.** The 25-token model adds 8 % MACs and 12 k parameters. In the HLS it only changes three constants in `config.h`: `RAW_SEQ_LEN` to 400, and `KERNEL_SIZE` and `CONV_STRIDE` to 16.
- **The trade-off is the update rate.** A 50 ms step gives 20 outputs per second, against 40 with the current 25 ms step. Whether 20 Hz is fast enough depends on the application.
- **Not directly comparable with session 3.** Session 3 used a 25 ms step and different test windows.

---

## 10. Hardware

### 10.1 Original HLS design

Source: [hls_code/](hls_code/). Target part xc7z020clg484-1 at 100 MHz.

**Structure.**

- A top function `RoFormer_Top` reads 2,400 inputs over AXI master, runs conv → RoPE/attention/norm → FFN/norm → pool → linear, and writes 10 outputs.
- Activations alternate between two ping-pong buffers, each 128 × 200.
- Attention loops over heads in the outer loop. It recomputes the K and V projection for each head, then does one query at a time.
- Softmax uses float `exp` and division. LayerNorm uses float `rsqrt`.

**Number formats.**

| Type | Format | Used for |
|---|---|---|
| `data_t` | `ap_fixed<10,3>` | Inputs, weights, activations |
| `rope_t` | `ap_fixed<8,2>` | RoPE tables |
| `mha_sum_t` | `ap_fixed<28,10>` | Attention and linear accumulators |
| `conv_sum_t` | `ap_fixed<24,8>` | Convolution accumulator |
| `ln_calc_t` | `ap_fixed<24,12>` | LayerNorm arithmetic |

All of these use the default truncation and wrap-around overflow.

**Memory, read from the generated RTL.**

| Memory | Size | Notes |
|---|---|---|
| Activation and buffer RAM | 1.14 Mbit | The largest is `attn_concat` at 716 kbit, stored 28 bits wide |
| Weight and table ROM | 1.07 Mbit | Weights synthesise to 6 to 7 bits wide, because the top bits of the 10-bit type are constant |
| Total | 2.2 Mbit | Of 4.9 Mbit block RAM on the Zynq-7020 |

**Latency.** The design performs 37.4 M MACs per inference. Every MAC loop is pipelined at one MAC per cycle and none is unrolled, so the lower bound is about 374 ms at 100 MHz. The softmax and normalisation float operators add to that. The budget is 25 ms. No synthesis or timing report is in the repository, so this is an estimate from the loop structure. Earlier notes in this project quoted 28 M MACs and 280 ms, which undercounted the attention arithmetic.

**Host link.** [pc_test_script/uart_basic.py](pc_test_script/uart_basic.py) sends one 2,400-float window as 9,600 bytes at 115,200 baud. That takes about 0.8 s, so UART is suitable for hardware-in-the-loop checks only, not for real-time streaming.

### 10.2 Verification findings

| Check | Result |
|---|---|
| numpy float replay vs PyTorch golden dumps | Cosine 1.0000 at every layer |
| HLS layer code built with a float mock of `ap_fixed`, vs golden | Cosine 0.9995 after attention |
| Fixed-point simulation vs golden | Cosine 0.9996 after attention |
| Checked-in Vitis dumps (`hls_code/vitis_outputs/`) | Cosine 0.54 after attention. Stale: produced by an older build |
| Fixed point on 41 real HIL windows | MAE 0.11, worst 0.68 |

The HIL manifest outputs did not match the exported weights. The notebook, the HLS code and the HIL samples described three different models: 4 heads with 400 samples, 8 heads with 200 samples, and an unknown third.

### 10.3 Export flow (current)

`export_hls` in [emg_common.py](kaggle_scripts/emg_common.py), notebook section 05. It writes every hardware artefact from one checkpoint and one real test window:

- Weight headers with the array names the HLS already uses.
- RoPE tables and a `config.h` with new macros such as `CONV_STRIDE`, `HAS_DELTA_HEAD` and `COND_PREV`.
- A suggested `types.h` with rounding, saturation and the `resid_t` type.
- Golden intermediates for every layer.
- 40 hardware-in-the-loop samples with a manifest.
- Firmware constants: EMG scale, glove range, μ, filter coefficients and `DELTA_SCALE`.
- `fixed_point_expectation.json`, the per-layer error a correct HLS build should reproduce.

Verification of the session-2 export:

| Check | Result |
|---|---|
| Fixed-point expectation, every layer | Cosine ≥ 0.998 |
| Existing HLS layer functions, float mock, fed the golden embedding output | Every downstream dump including both heads at cosine ≥ 0.9977 |

### 10.4 HLS v2: implementation for the chosen model

The code in [hls_code/](hls_code/) now implements the session-2 checkpoint. The old design is in git history and stale artefacts are in `hls_code/legacy/`.

| Change | Where |
|---|---|
| Strided embedding, kernel 8, stride 8, 25 tokens; reads the 12 × 200 channel-major AXI input | `Conv1dLayer.h` |
| One flattened, pipelined dense kernel: `MAC_UNROLL` MACs per cycle, no per-neuron pipeline refill | `FCLayer.h` |
| Q, K, V projected once for all tokens and heads instead of once per head | `MHANorm.h` |
| Residual stream in `resid_t` = `ap_fixed<16,8>`; LayerNorm takes it as input | `MHANorm.h`, `FFNorm.h`, `LayerNorm.h` |
| Separate types: `qkv_t` 18 bits so each Q·K multiply fits one DSP; `prob_t` for softmax; narrow LayerNorm multipliers | `types.h` |
| Rounding and saturation on every narrowing type | `types.h` |
| Second output head; output is 20 values: absolute angles, then standardised deltas | `RoFormer_Top.cpp` |
| Firmware rescales deltas with `DELTA_SCALE` and runs the constant-gain fusion; returns 30 values | `vitis_workspace/uart_app/src/helloworld.c` |

**Bit-accurate verification.** `make run` in `hls_code` builds with Xilinx's open-source `ap_fixed` headers, vendored in `hls_code/third_party/ap_types` under Apache-2.0. Vitis is not needed.

| Check | Result |
|---|---|
| Old HLS code with v2 weights, bit-accurate | Cosine −0.06 after the FFN block: the 10-bit residual wraps around. The float mock had shown 0.9997 |
| New code, every layer against the float golden model | Cosine 0.9981 to 0.9999. Matches `fixed_point_expectation.json` to the fourth decimal |
| New code on 40 real test windows, against the float model | Absolute head MAE 0.024, about 3 steps of 1/128. Delta MAE 0.12 standardised |
| New code against the numpy fixed-point simulation, same 40 windows | Absolute head 0.0004, delta 0.002: the C model is correct, and the remaining error is the format itself |
| `MAC_UNROLL` = 2, 4, 8 | Identical outputs |

The 10-bit format's error is already included in the deployable R² of 0.667, because quantisation-aware training applied the same rounding during evaluation.

**Latency estimate** from the loop structure, by `hls_code/scripts/estimate_cycles.py`. These are not synthesis results.

| `MAC_UNROLL` | Cycles | Time at 100 MHz |
|---|---|---|
| 2 | 1.75 M | 17.5 ms |
| 4 (default) | 0.93 M | 9.3 ms |
| 8 | 0.52 M | 5.2 ms |
| Original design | ≥ 37.4 M | ≥ 374 ms |

At the default setting, the FFN is 44 % of the time and the Q/K/V projections 33 %. Softmax and LayerNorm still use float `exp`, division and `rsqrt`; replacing them is optional, because they are under 7 % of the time.

**Not yet done:** synthesis, timing closure and resource reports; co-simulation; re-export of the IP, Vivado block design update and board test. The IP interface changed: the output depth went from 10 to 20, so the block design and firmware must be rebuilt together.

---

## 11. Resource consumption overview

Latency is the lower bound at one MAC per cycle and 100 MHz, ignoring nonlinear operators. Weight memory is parameters × 10 bits. Training time is on one Kaggle T4 for 10 subjects.

| Model | Params | MACs | Latency bound | Weights | Train time per epoch |
|---|---|---|---|---|---|
| Original RoFormer, 200 tokens | 138,506 | 37.4 M | 374 ms | 1.39 Mbit | 133 s |
| patch, absolute head | 146,186 | 3.75 M | 37 ms | 1.46 Mbit | 62 s |
| noattn, absolute head | 79,882 | 1.95 M | 19 ms | 0.80 Mbit | 53 s |
| **patch, both heads (chosen)** | **147,476** | **3.75 M** | **37 ms** | **1.47 Mbit** | **83 s float, 135 s QAT** |
| Unrolled K=8 variants | 82 k–149 k | 1.95–3.75 M | | | 290–516 s |

**Activation buffers** shrink 8× with the patch model: 25 × 128 values per buffer instead of 200 × 128.

**Session wall-clock times:**

| Session | Time |
|---|---|
| Session 1, preprocessing plus 8 runs | 11.9 h, exceeded the Kaggle 12 h limit once |
| Session 2, one run plus evaluation and export | 3.3 h |
| Session 3, evaluation only | 2.2 h |

---

## 12. Pitfalls and lessons

1. **Overlapping windows with a within-class split leak test data.** Always hold out whole repetitions or sessions.
2. **Any input that contains the past ground-truth angle makes window metrics meaningless.** This applies to previous-angle conditioning and to "true start + predicted diff". Evaluate sequentially, with the model's own estimate fed back.
3. **When combining two loss terms, check their scales.** An unnormalised second head silently takes over early stopping and model selection.
4. **QAT must quantise where the hardware quantises.** Quantising only linear-layer inputs hid a 30× overflow in the residual stream.
5. **Separate smoothing from model skill.** Compare fusion against a smoothing-only baseline before crediting a new head.
6. **Keep one source of truth for configuration.** Regenerate golden dumps and HIL samples from the deployed checkpoint every time.
7. **Kaggle sessions are capped at 12 hours.** Keep grids to about three configurations per session. Click "Save Version", or the output folder is lost.
8. **Report per-subject metrics beside pooled ones.** Pooled R² over several subjects counts between-subject differences as explained variance. Here it read 0.67 while the per-subject mean was 0.52.
9. **Check the accelerator before a long run.** Session 4 ran on the CPU because the GPU was not switched on. It finished, but took 9.3 hours instead of about 3.
10. **Secrets.** The original baseline notebook contains a plaintext Weights & Biases API key. It should be revoked. The new scripts read the key from a Kaggle secret named `wandb_key`.

---

## 13. Open work

| Priority | Task |
|---|---|
| 1 | Synthesise the v2 HLS in Vitis (`vitis_workspace/transformer`), commit the utilisation and timing reports, run co-simulation, then rebuild the block design and measure latency on the board |
| 2 | Decide the step: keep 100 ms windows at 25 ms (40 Hz output), or adopt `w200ms_patch16_t25` at 50 ms (20 Hz, 2.3 % lower RMSE). If adopting it, fine-tune with QAT, export, and change the three `config.h` constants |
| 2b | New-user calibration: fine-tune a leave-subjects-out model on 1 to 4 training repetitions of the held-out subject, and measure how much within-subject accuracy is recovered |
| 3 | Convert normalised RMSE to degrees using glove calibration |
| 4 | Port the causal filter and μ-law normalisation to the ARM firmware using `normalization_constants.h`. The fusion update is already in the firmware |
| 5 | Replace UART with a faster host link if live streaming is required |

---

## 14. Repository map and reproduction

| Path | Contents |
|---|---|
| [kaggle_scripts/emg_common.py](kaggle_scripts/emg_common.py) | All current data, model, training, evaluation and export code |
| [kaggle_scripts/combined-roformer-ablation.ipynb](kaggle_scripts/combined-roformer-ablation.ipynb) | Notebook configured as the session-3, evaluation-only run |
| [kaggle_scripts/06_subject_ci_and_loso.ipynb](kaggle_scripts/06_subject_ci_and_loso.ipynb) | Session 4: per-subject confidence intervals and leave-subjects-out |
| [kaggle_scripts/07_window_200ms.ipynb](kaggle_scripts/07_window_200ms.ipynb) | Window-length study: 200 ms windows, 50 ms step |
| `kaggle_scripts/01_` … `05_*.ipynb` | Stand-alone notebooks for each stage |
| [kaggle_scripts/README.md](kaggle_scripts/README.md) | Kaggle setup and configuration options |
| [roformer_output/](roformer_output/) | Session 1: grid checkpoints, `ablation_results_merged.csv`, logs |
| [roformer_output_v2/](roformer_output_v2/) | Session 2: chosen checkpoint, report, HLS export |
| [roformer_output_v3/](roformer_output_v3/) | Session 3: deployable evaluation report and plots |
| [hls_code/](hls_code/) | HLS v2 implementation, bit-accurate tests (`make run`), latency estimator, golden data from the session-2 export. Old artefacts in `hls_code/legacy/` |
| [ip_repo/](ip_repo/), `roformer.tcl`, `transformer_wrapper.xsa`, [vitis_workspace/](vitis_workspace/) | Vivado IP, project script, platform, firmware |
| [notebooks/](notebooks/), [kaggle_scripts/roformer.ipynb](kaggle_scripts/roformer.ipynb), [kaggle_scripts/baseline-models-joint-angles.ipynb](kaggle_scripts/baseline-models-joint-angles.ipynb) | Original notebooks, legacy pipeline |
| [IMPROVEMENTS.md](IMPROVEMENTS.md) | Review findings and the original roadmap |

**To reproduce on Kaggle:**

1. Attach NinaPro DB2.
2. Add `emg_common.py` as a utility script.
3. Add a Kaggle secret named `wandb_key`, or set `use_wandb=False`.
4. Run notebook 01 and save its output as a dataset.
5. Run notebook 02 with the configuration in section 9.3, then 04 and 05.

**To verify the HLS locally,** run `make run` in `hls_code`. It builds with the vendored open-source `ap_fixed` headers, compares every layer with the golden dumps in `verification_data/`, and runs the 40 windows in `hil_verification_samples/`. Do not trust the float mock `mock_ap_fixed.h`: it hid a complete numerical failure (section 10.4).

---

## 15. Glossary

| Term | Meaning |
|---|---|
| sEMG | Surface electromyography: electrical muscle activity measured on the skin |
| RoPE | Rotary position embedding: encodes position by rotating query and key vectors |
| Patch embedding | A strided convolution that turns a sequence of samples into fewer, wider tokens |
| MAC | Multiply-accumulate, the unit of compute for neural-network layers |
| QAT | Quantisation-aware training: training with simulated fixed-point rounding |
| `ap_fixed<W,I>` | Xilinx fixed-point type with W total bits, of which I are integer bits including sign |
| Teacher forcing | Feeding the true previous output as input; scores computed this way overstate skill |
| Sequential evaluation | Replaying test data in time order with the model's own outputs fed back |
| HIL | Hardware in the loop: running real data through the board and comparing outputs |
| Golden dump | Reference intermediate outputs from the float model, used to check the hardware |
