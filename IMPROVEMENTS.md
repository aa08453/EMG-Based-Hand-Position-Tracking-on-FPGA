# RoFormer sEMG-to-Angle Pipeline: Review Findings and Roadmap

For the full history and current results, see [PROJECT_REPORT.md](PROJECT_REPORT.md).

This document records the results of a review of the PyTorch model, the data pipeline, and the HLS implementation, followed by the recommended changes. No source files were modified during the review; the "verified" items below are measurements made against the artifacts already in the repository.

## 1. What was verified

### 1.1 Method

- The forward pass was re-implemented in numpy from the float weights in `hls_code/verification_data/` and compared layer by layer against the PyTorch golden dumps.
- The HLS layer functions (`Conv1dLayer.h`, `RotaryEmbedding.h`, `MHANorm.h`, `FFNorm.h`, `Pooling.h`, `FCLayer.h`) were compiled with the project's float mock of `ap_fixed` and run on the golden input.
- The fixed-point rounding of `ap_fixed<10,3>` (truncation, 7 fractional bits) was simulated in numpy on both the golden input and the 41 real sEMG windows in `hil_verification_samples/`.
- Memory sizes were read from the RAM/ROM module parameters in `ip_repo/RoFormer_Top/hdl/verilog/`.

### 1.2 Results

| Check | Result |
|---|---|
| numpy float replay vs PyTorch golden (8 heads) | cosine 1.0000 at every layer |
| current C++ layer code (float mock) vs golden, attention block | cosine 0.9995 |
| fixed-point simulation vs golden, attention block | cosine 0.9996 |
| checked-in `hls_code/vitis_outputs/` vs golden, attention block | cosine 0.54 |
| fixed-point error on 41 real HIL windows (targets span 0 to 1) | MAE 0.11, worst 0.68 |
| attention entropy on real windows (uniform = 5.30) | 5.28 to 5.30 |
| uniform-attention ablation vs real attention, output MAE | 0.14 |
| notebook sequential evaluation, subject 1 | R² −0.91, RMSE 17.3° |
| estimated MACs per inference | about 37.4 million: 26.2 million in per-token linear layers, 10.2 million in attention scores and weighting (corrected from an earlier 28 million) |
| estimated latency at 1 MAC/cycle, 100 MHz | about 374 ms versus a 25 ms window stride (corrected from 280 ms) |
| activation RAM / weight ROM instantiated | about 1.14 Mbit / 1.07 Mbit of 4.9 Mbit on the Zynq-7020 (RAM corrected from 1.7 Mbit) |
| synthesized weight ROM width | 6 to 7 bits of the 10-bit weight type |

### 1.3 Conclusions drawn from the results

1. **The attention block in `MHANorm.h` is logically correct.** The float-mock build reproduces the golden dumps and no accumulator can overflow with these weights (largest pre-scale dot product is 149 against a range of ±512). The checked-in Vitis dumps that show the block failing come from an earlier build and should be regenerated.
2. **Three artifacts describe three different models.** The notebook trains 4 heads on 400-sample windows; the HLS uses 8 heads and 200 samples; the HIL manifest outputs match neither the exported weights nor the normalized target range (79 percent of values fall outside 0 to 1).
3. **The reported R² of 0.71 is inflated by leakage.** Windows overlap by 350 of 400 samples and the split takes the first 80 percent of each repetition, so test windows are near-duplicates of training windows. The sequential, unshuffled evaluation gives negative R² on a training subject.
4. **`VelocityGuidedLoss` is a no-op.** Predicted velocity minus true velocity simplifies to prediction minus target, so the loss equals 1.1 × MSE.
5. **Normalization is non-causal.** Per-recording max-abs scaling of EMG and min-max scaling of glove angles use the whole recording, which cannot be reproduced in real time and leaks test statistics into training.
6. **The fixed-point format is sized for the random test input, not real data.** Real inputs lie in 0 to 0.68 with standard deviation 0.10, so a 1/128 step gives about 13 levels per standard deviation. Weights peak at 0.45, so three of the ten weight bits carry nothing. Default `ap_fixed` casts wrap on overflow, and the golden FFN output already contains values beyond ±4.
7. **The design misses real time by roughly 15×.** Every MAC loop runs one multiply per cycle. The cyclic-8 array partitions buy nothing because no MAC loop is unrolled.

## 2. Recommended changes

Ordered by expected value. Items in section 2.1 should come first because HLS tuning of a model that does not work on real data is wasted effort.

### 2.1 Data pipeline and evaluation

- [ ] Split by repetition (NinaPro convention, e.g. repetitions 2 and 5 held out) or by session, never within a repetition.
- [ ] Replace per-recording normalization with train-set statistics or a causal running normalizer, and store the constants in the firmware.
- [ ] Make the sequential, unshuffled evaluation the primary metric; report error as a function of time.
- [ ] Reconcile the configuration: one head count, one window length, one checkpoint shared by the notebook, the weight export, the HLS `config.h`, and the HIL samples.
- [ ] Regenerate `hil_verification_samples/` from the same checkpoint that produced `hls_code/weights/`.
- [ ] Re-run Vitis C-simulation and replace the stale `hls_code/vitis_outputs/`.

### 2.2 Model architecture

- [ ] Reduce the token count with a strided conv or patch embedding (200 samples to about 25 tokens). Cuts compute about 8×.
- [ ] Train an ablation with mean pooling plus the FFN and no attention; keep attention only if it measurably helps on the sequential metric.
- [ ] Switch to causal attention and exploit RoPE's relative encoding for a streaming K/V cache so each stride processes only the new samples.
- [ ] Replace LayerNorm with RMSNorm (removes the mean pass in hardware).
- [ ] Run quantization-aware training with fake-quantization that matches the HLS formats exactly, including the attention path.
- [ ] Add a second output head predicting angle deltas alongside absolute angles.
- [ ] Replace `VelocityGuidedLoss` with supervision of the delta head against the difference between consecutive window labels.

### 2.3 HLS design

- [ ] Unroll MAC loops by 8 or 16 and partition the weight ROMs to match the existing cyclic activation partitions.
- [ ] Introduce `hls::stream` and `DATAFLOW` so FFN, LayerNorm 2 and pooling stream behind attention; fuse pooling into the FFN loop.
- [ ] Give inputs, weights and activations separate fixed-point types, or use integer weights with per-tensor shifts. Use `AP_RND` and `AP_SAT` on every narrowing cast.
- [ ] Store K and V in 10 or 16 bits instead of 28, freeing the 716 kbit `attn_concat` buffer and allowing one QKV pass instead of eight per-head passes.
- [ ] Track the softmax maximum inside the dot-product loop and replace float `exp` and divide with a fixed-point table and one reciprocal.
- [ ] Compute LayerNorm sum and sum of squares in one pass.
- [ ] Check in the synthesis report so latency and utilization claims can be verified.

### 2.4 Angle-delta prediction and drift control

Integrating predicted deltas is a random walk: independent errors grow with the square root of the step count, bias grows linearly. sEMG tracks activation and velocity better than absolute pose, so deltas are a good target, but they must be anchored.

- [ ] Fuse the delta head (process model) with the absolute head (measurement) in a per-joint Kalman filter; the scalar smoother already in the notebook becomes this filter once the delta drives the state update.
- [ ] Train with integration in the loop: feed back the previous estimate with added noise, unroll several steps, and apply the loss to the integrated trajectory.
- [ ] Re-anchor at rest, detected from EMG energy; clamp to joint limits; use leaky integration so old errors decay.
- [ ] Standardize delta targets before quantization (per-stride deltas are only a few LSBs at 1/128 resolution).
- [ ] Keep the integrator and filter in float on the ARM core, or use `AP_RND` with a wide accumulator; truncation bias integrates into steady drift.
- [ ] Evaluate on sequential streams and report error versus time since the last anchor.

## 3. Reproducing the measurements

All measurements used the Python environment at `~/Music/wheels/.venv` and the files already in the repository. The float-mock build of the layer functions needs two one-line shim headers, `ap_fixed.h` including `mock_ap_fixed.h` and `hls_math.h` including `mock_hls_math.h`, placed on the include path ahead of the HLS tool headers.

## 4. Implementation status (Kaggle scripts)

The machine-learning side of section 2 is implemented in `kaggle_scripts/` (see its README):

| Roadmap item | Where |
|---|---|
| 2.1 repetition split, train-only normalisation, causal filter, sequential metric | `01_preprocess_split.ipynb`, `emg_common.process_subject`, `ordered_runs` |
| 2.1 reconcile configuration, regenerate HIL samples and golden dumps from one checkpoint | `05_export_hls.ipynb`, `emg_common.export_hls` |
| 2.2 token reduction, attention ablation, causal attention, RMSNorm, fixed-point QAT, delta head | `02_train_roformer.ipynb`, `03_ablation_grid.ipynb`, `emg_common.EMGRoFormer`, `convert_to_qat` |
| 2.4 Kalman fusion, integration-in-the-loop, rest anchoring, delta standardisation, drift evaluation | `04_sequential_eval_fusion.ipynb`, `emg_common.kalman_fuse`, `_loss_unrolled`, `fixed_point_integrate` |
| 2.3 HLS changes | implemented and verified bit-accurately in C simulation (Oct 15); synthesis pending. See PROJECT_REPORT.md section 10.4 |
| Multi-layer encoder in HLS, formats from the export, install script | Oct 20: `hls_code/RoFormer_Top.cpp` layer loop, `model_config.h`, `scripts/install_export.py`; 1-, 2- and 3-layer models verified against the fixed-point reference. PROJECT_REPORT.md section 10.5 |
| QAT and export of the deep model | `09_qat_export_deep.ipynb` (prepared, not yet run on Kaggle) |
