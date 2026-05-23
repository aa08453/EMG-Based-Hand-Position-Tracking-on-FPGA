// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xroformer_top.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XRoformer_top_CfgInitialize(XRoformer_top *InstancePtr, XRoformer_top_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XRoformer_top_Start(XRoformer_top *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_AP_CTRL) & 0x80;
    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XRoformer_top_IsDone(XRoformer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XRoformer_top_IsIdle(XRoformer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XRoformer_top_IsReady(XRoformer_top *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XRoformer_top_EnableAutoRestart(XRoformer_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XRoformer_top_DisableAutoRestart(XRoformer_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_AP_CTRL, 0);
}

void XRoformer_top_Set_input_axi(XRoformer_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_INPUT_AXI_DATA, (u32)(Data));
    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_INPUT_AXI_DATA + 4, (u32)(Data >> 32));
}

u64 XRoformer_top_Get_input_axi(XRoformer_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_INPUT_AXI_DATA);
    Data += (u64)XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_INPUT_AXI_DATA + 4) << 32;
    return Data;
}

void XRoformer_top_Set_output_axi(XRoformer_top *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_OUTPUT_AXI_DATA, (u32)(Data));
    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_OUTPUT_AXI_DATA + 4, (u32)(Data >> 32));
}

u64 XRoformer_top_Get_output_axi(XRoformer_top *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_OUTPUT_AXI_DATA);
    Data += (u64)XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_OUTPUT_AXI_DATA + 4) << 32;
    return Data;
}

void XRoformer_top_InterruptGlobalEnable(XRoformer_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_GIE, 1);
}

void XRoformer_top_InterruptGlobalDisable(XRoformer_top *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_GIE, 0);
}

void XRoformer_top_InterruptEnable(XRoformer_top *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_IER);
    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_IER, Register | Mask);
}

void XRoformer_top_InterruptDisable(XRoformer_top *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_IER);
    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_IER, Register & (~Mask));
}

void XRoformer_top_InterruptClear(XRoformer_top *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XRoformer_top_WriteReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_ISR, Mask);
}

u32 XRoformer_top_InterruptGetEnabled(XRoformer_top *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_IER);
}

u32 XRoformer_top_InterruptGetStatus(XRoformer_top *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XRoformer_top_ReadReg(InstancePtr->Control_BaseAddress, XROFORMER_TOP_CONTROL_ADDR_ISR);
}

