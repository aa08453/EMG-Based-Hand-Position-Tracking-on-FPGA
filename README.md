# EMG-Based Hand Position Tracking on FPGA

This repository contains a research-oriented FPGA implementation for estimating hand position from surface EMG (sEMG) signals using a RoFormer-inspired transformer pipeline. The design targets a Zynq-based platform and combines high-level synthesis (HLS), Vivado project infrastructure, Vitis software support, and a host-side UART verification flow.

## Project Overview

The goal of this project is to map sEMG input windows to continuous hand joint predictions using a compact transformer-style model accelerated on FPGA hardware. The workflow includes:

- preprocessing and preparing EMG/gesture data,
- implementing the model inference pipeline in C++ for HLS,
- exporting the IP core for integration in Vivado,
- verifying the design on hardware through a PC-side serial interface.

## Current Status

The deployed model is **L3_w8**: a 3-layer RoFormer on 200 ms windows every 50 ms, trained with 8-bit weights and 12-bit activations, fused R² 0.775 on held-out repetitions of 10 NinaPro DB2 subjects. Its HLS implementation (layer-loop version) fits the Zynq-7020 in Vitis HLS synthesis at 23.95 ms per window. Vivado implementation and the board test are pending.

- [PROJECT_REPORT.md](PROJECT_REPORT.md): everything tried, results, hardware, open work.
- [SYNTHESIS_GUIDE.md](SYNTHESIS_GUIDE.md): from HLS to the board test.
- [kaggle_scripts/README.md](kaggle_scripts/README.md): training and export notebooks.
- [unity_bridge/README.md](unity_bridge/README.md): run the model on a PC and drive a Unity hand.

## Repository Structure

- hls_code/: HLS implementation of the RoFormer-style inference pipeline, including headers, weights, tests, and verification data.
- ip_repo/RoFormer_Top/: exported IP repository content for use in Vivado.
- transformer_wrapper.xsa: the exported Xilinx support archive (XSA) platform file that can be used directly in Vitis as the embedded platform.
- vivado_project/ and roformer.tcl: Vivado project files and a Tcl script to recreate the project.
- vitis_workspace/: Vitis workspace for embedded software integration.
- constraints/: board constraint files, including the ZedBoard pin mapping.
- kaggle_scripts/: current training, evaluation and HLS export code (`emg_common.py`) and notebooks 01 to 09.
- notebook_outputs/: results of every Kaggle session, including the exported model.
- report/, report_layer_loop/: Vitis HLS synthesis reports.
- unity_bridge/: PC runtime and Unity scripts for a live hand demo.
- pc_test_script/: Python script for UART-based communication between a PC and the FPGA board.
- hil_verification_samples/: 40 real test windows with float and fixed-point reference outputs, used by the HLS tests.

## Hardware and Software Requirements

- Xilinx Vivado 2025.2 (or a compatible Vivado/Vitis installation)
- Xilinx Vitis HLS / Vitis tools for HLS synthesis and software integration
- Python 3 with numpy and pyserial
- A Zynq-based development board such as ZedBoard

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd EMG-Based-Hand-Position-Tracking-on-FPGA
```

### 2. Recreate the Vivado Project

Open Vivado and source the project script:

```tcl
source roformer.tcl
```

This will recreate the Vivado project structure and import the relevant constraint and IP files.

### 3. Build the HLS Testbench

From the HLS source directory:

```bash
cd hls_code
make
```

This builds the host-side test executables used for functional verification of the model pipeline, using the vendored open-source `ap_fixed` headers (no Vitis needed).

### 4. Run the HLS Verification Tests

```bash
cd hls_code
make run
```

### 5. Run the PC-to-FPGA UART Test

The Python test script sends input data to the FPGA board and reads the predicted output values over UART.

1. Update the serial port in pc_test_script/uart_basic.py if necessary.
2. Run:

```bash
python3 pc_test_script/uart_basic.py
```

## Data and Modeling Notes

The current pipeline is in kaggle_scripts/: causal filtering, train-only normalisation, a repetition-held-out split, quantisation-aware training and the HLS export. The original notebook (kaggle_scripts/roformer.ipynb) used a split that leaked test data; PROJECT_REPORT.md section 9.1 explains why its R² of 0.71 is not comparable.

## Notes

- This repository is intended as a prototype and research implementation.
- Hardware-specific paths, board settings, and serial-port configuration may need adjustment depending on the local environment.
- The project report and graphical abstract provide additional context and implementation details.

