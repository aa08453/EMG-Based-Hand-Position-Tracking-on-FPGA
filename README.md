# EMG-Based Hand Position Tracking on FPGA

This repository contains a research-oriented FPGA implementation for estimating hand position from surface EMG (sEMG) signals using a RoFormer-inspired transformer pipeline. The design targets a Zynq-based platform and combines high-level synthesis (HLS), Vivado project infrastructure, Vitis software support, and a host-side UART verification flow.

## Project Overview

The goal of this project is to map sEMG input windows to continuous hand joint predictions using a compact transformer-style model accelerated on FPGA hardware. The workflow includes:

- preprocessing and preparing EMG/gesture data,
- implementing the model inference pipeline in C++ for HLS,
- exporting the IP core for integration in Vivado,
- verifying the design on hardware through a PC-side serial interface.

## Repository Structure

- hls_code/: HLS implementation of the RoFormer-style inference pipeline, including headers, weights, tests, and verification data.
- ip_repo/RoFormer_Top/: exported IP repository content for use in Vivado.
- transformer_wrapper.xsa: the exported Xilinx support archive (XSA) platform file that can be used directly in Vitis as the embedded platform.
- vivado_project/ and roformer.tcl: Vivado project files and a Tcl script to recreate the project.
- vitis_workspace/: Vitis workspace for embedded software integration.
- constraints/: board constraint files, including the ZedBoard pin mapping.
- notebooks/: contains the notebook with the PyTorch-based model experimentation and preprocessing workflow.
- pc_test_script/: Python script for UART-based communication between a PC and the FPGA board.
- hil_verification_samples/: sample input/output verification files used for hardware-in-loop testing.

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

This builds the host-side test executables used for functional verification of the model pipeline.

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

The notebook in notebooks/roformer.ipynb contains preprocessing and modeling experiments for EMG data. It references the NinaPro DB2 dataset and includes steps for filtering, normalization, segmentation, and preparation of model input windows.

## Notes

- This repository is intended as a prototype and research implementation.
- Hardware-specific paths, board settings, and serial-port configuration may need adjustment depending on the local environment.
- The project report and graphical abstract provide additional context and implementation details.

