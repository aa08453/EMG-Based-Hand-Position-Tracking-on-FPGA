# Synthesis and board test: L3_w8

How to take the installed model from C code to the ZedBoard. The model is L3_w8:

- 3 encoder layers, 200 ms windows, 50 ms step
- 8-bit weights, 12-bit activations
- fused R² 0.775 in notebook 09

The HLS code in `hls_code/` already holds this model and passes the bit-accurate C simulation (`make run`). See PROJECT_REPORT.md section 10.5.

## 0. Requirements

- **Vitis and Vivado 2025.2**, the version `roformer.tcl` was generated with, with Zynq-7000 device support.
- **The repository on the `dev` branch** (`git pull`). It must include:
  - `hls_code/`, with `third_party/`
  - `ip_repo/`, `roformer.tcl`, `constraints/`
  - `vitis_workspace/`
  - `pc_test_script/`
- Python with `pyserial` and `numpy`, for the UART test only.

Python, the Kaggle outputs and `hil_verification_samples/` are not needed for synthesis.

Check that the installed model is L3_w8:

```bash
grep -E "generated|NUM_LAYERS|ACT_W|WEIGHT_W" hls_code/model_config.h
# expect: run "L3_w8", NUM_LAYERS 3, ACT_W 12, WEIGHT_W 8
```

## 1. HLS synthesis

Run from a folder outside the repository, so the build files stay out of it:

```bash
mkdir -p ~/hls_runs && cd ~/hls_runs
vitis-run --mode hls --tcl <repo>/hls_code/scripts/run_hls.tcl
# older Vitis HLS:  vitis_hls -f <repo>/hls_code/scripts/run_hls.tcl
```

The script `hls_code/scripts/run_hls.tcl` runs three steps, in roughly 15 to 40 minutes:

1. **C simulation**, which must end with `PASS`.
2. **C synthesis.**
3. **IP export** to `~/hls_runs/RoFormer_Top_ip.zip`.

Optional settings, set as environment variables before the command:

| Variable | Default | Effect |
|---|---|---|
| `MAC_UNROLL` | `4` | Multiply-accumulates per cycle in each dense layer. `8` cuts the estimated latency from 27 ms to 15 ms at the cost of more DSPs. Either fits the 50 ms step |
| `HLS_STEPS` | `"csim csynth export"` | Add `cosim` for C/RTL co-simulation, which is slow (30 minutes or more) |
| `LUTRAM_ATTN_BUFFERS` | `1` | Keeps the Q/K/V and attention-output buffers in LUT-RAM so the block RAM is left for the weights. Set to `0` if LUTs run short instead |
| `HLS_CLOCK_NS` | `10` | Clock period. 100 MHz matches FCLK0 in the block design |

Example: `MAC_UNROLL=8 vitis-run --mode hls --tcl <repo>/hls_code/scripts/run_hls.tcl`

**IDE alternative:** the component `vitis_workspace/transformer` has an updated file list in `hls_config.cfg`. If the IDE cannot find the sources, correct the `../../../hls_code` prefix in that file. The Tcl script avoids that problem.

### Check the synthesis report

The report is `~/hls_runs/roformer_hls/sol1/syn/report/RoFormer_Top_csynth.rpt`.

| Item | Must be | Expected |
|---|---|---|
| BRAM_18K | ≤ 280 | about 210 to 230 (weights 3.4 Mbit) |
| DSP | ≤ 220 | well below the limit at `MAC_UNROLL=4` |
| LUT | ≤ 53,200 | about 4.6 k of it is LUT-RAM for the attention buffers |
| Estimated clock period | ≤ 8.75 ns | 10 ns minus 12.5 % uncertainty |
| Latency | < 5,000,000 cycles (50 ms) | about 2.7 M cycles (27 ms) at `MAC_UNROLL=4` |
| `DENSE_MAC` loops | II = 1 | |

Warning signs in the log (`vitis_hls.log` or the console output):

- An II violation on `DENSE_MAC`, `MHA_DOT` or `CONV_MAC`.
- An `ARRAY_PARTITION` pragma reported as ignored.
- BRAM above 100 %.

If you see any of these, save the log and the report and stop here.

## 2. Vivado: bitstream

1. **Replace the IP.** Delete the contents of `<repo>/ip_repo/RoFormer_Top/`, then unzip `RoFormer_Top_ip.zip` into that folder.
2. **Open the project.** On a new machine, recreate it from the repository root in the Vivado Tcl console:
   ```tcl
   cd <repo>
   source roformer.tcl
   ```
3. **Upgrade the IP.** Tools → Report IP Status → select `RoFormer_Top_0` → Upgrade Selected. The interface is unchanged:
   - `s_axi_control` at 0x4000_0000
   - `m_axi_gmem0` and `m_axi_gmem1` to the HP ports

   No wiring changes are needed. Accept the regeneration of the block design output products.
4. **Generate Bitstream.**
5. **Check:**
   - Utilization: Block RAM within 140 tiles, and LUT and DSP within limits.
   - Timing summary: WNS ≥ 0 and no failing endpoints.
6. **Export hardware.** File → Export → Export Hardware → Include bitstream. This writes the new `.xsa`.

## 3. Firmware

1. **Update the platform.** In Vitis, update the platform component `model1` with the new `.xsa`, then rebuild it. This regenerates the `XRoformer_top` driver for the new IP.
2. **Build `uart_app`.** Its `vitis_workspace/uart_app/src/normalization_constants.h` already holds the L3_w8 values:
   - `MODEL_IN_WORDS 4800` (12 channels × 400 samples)
   - `ACT_TOTAL_BITS 12` and `ACT_INT_BITS 4` (data format `ap_fixed<12,4>`)
   - `MODEL_FUSION_GAIN` (0.095), the constant gain used for fusion

   `helloworld.c` reads all of these, so no code edits are needed.
3. **Program the board** with the bitstream and run `uart_app`.

## 4. Board test

On the PC, from the repository root, after setting `COM_PORT` in the script:

```bash
python pc_test_script/uart_basic.py
```

The script sends `hls_code/verification_data/full_input.txt`, which is 4,800 floats taking about 1.7 s over UART. It is the same window the C simulation uses.

The script prints three columns per joint. Compare them with the "abs hw" and "dlt hw" columns printed by `cd hls_code && make run` (`./test_top`):

| Column printed by the script | Expected value |
|---|---|
| `abs` | identical to "abs hw", e.g. joint 0: 0.4453, joint 1: 0.3125, joint 2: 0.3594 |
| `delta` | "dlt hw" × `DELTA_SCALE[j]` from `normalization_constants.h`. The firmware converts the standardised delta (joint 0: 0.2383) to angle units |
| `fused` | equal to `abs` on the first window after reset, because the filter is seeded from the first absolute prediction; it diverges from `abs` on later windows |

`abs` and the unscaled delta must match exactly. The board runs the same fixed-point arithmetic as the C simulation, so any difference means the hardware, platform or firmware is out of date (see the last table).

## 5. What to send back

1. `RoFormer_Top_csynth.rpt`, plus the HLS log if it has warnings.
2. The Vivado utilization report and timing summary.
3. The UART test output from the board.

## If it does not fit

| Problem | First thing to try |
|---|---|
| BRAM over 100 % | Confirm `LUTRAM_ATTN_BUFFERS=1` was used. Otherwise fall back to **L2_q10** (2 layers, 2.9 Mbit, fused R² 0.766). Re-run notebook 09 with its own output attached and `EMG_EXPORT=L2_q10`, then `python hls_code/scripts/install_export.py <unzipped export>` and `make run` |
| LUT over 100 % | `LUTRAM_ATTN_BUFFERS=0`, if BRAM has room |
| Timing fails (WNS < 0) | Re-run HLS with `HLS_CLOCK_NS=12` (83 MHz) and set FCLK0 to match in the block design |
| Latency above 50 ms | `MAC_UNROLL=8` |
| Board outputs differ from the C simulation | Check that the platform was rebuilt from the new `.xsa` and that `uart_app` was rebuilt afterwards |
