// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2025.2 (64-bit)
// Tool Version Limit: 2025.11
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XROFORMER_TOP_H
#define XROFORMER_TOP_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xroformer_top_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XRoformer_top_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XRoformer_top;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XRoformer_top_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XRoformer_top_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XRoformer_top_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XRoformer_top_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XRoformer_top_Initialize(XRoformer_top *InstancePtr, UINTPTR BaseAddress);
XRoformer_top_Config* XRoformer_top_LookupConfig(UINTPTR BaseAddress);
#else
int XRoformer_top_Initialize(XRoformer_top *InstancePtr, u16 DeviceId);
XRoformer_top_Config* XRoformer_top_LookupConfig(u16 DeviceId);
#endif
int XRoformer_top_CfgInitialize(XRoformer_top *InstancePtr, XRoformer_top_Config *ConfigPtr);
#else
int XRoformer_top_Initialize(XRoformer_top *InstancePtr, const char* InstanceName);
int XRoformer_top_Release(XRoformer_top *InstancePtr);
#endif

void XRoformer_top_Start(XRoformer_top *InstancePtr);
u32 XRoformer_top_IsDone(XRoformer_top *InstancePtr);
u32 XRoformer_top_IsIdle(XRoformer_top *InstancePtr);
u32 XRoformer_top_IsReady(XRoformer_top *InstancePtr);
void XRoformer_top_EnableAutoRestart(XRoformer_top *InstancePtr);
void XRoformer_top_DisableAutoRestart(XRoformer_top *InstancePtr);

void XRoformer_top_Set_input_axi(XRoformer_top *InstancePtr, u64 Data);
u64 XRoformer_top_Get_input_axi(XRoformer_top *InstancePtr);
void XRoformer_top_Set_output_axi(XRoformer_top *InstancePtr, u64 Data);
u64 XRoformer_top_Get_output_axi(XRoformer_top *InstancePtr);

void XRoformer_top_InterruptGlobalEnable(XRoformer_top *InstancePtr);
void XRoformer_top_InterruptGlobalDisable(XRoformer_top *InstancePtr);
void XRoformer_top_InterruptEnable(XRoformer_top *InstancePtr, u32 Mask);
void XRoformer_top_InterruptDisable(XRoformer_top *InstancePtr, u32 Mask);
void XRoformer_top_InterruptClear(XRoformer_top *InstancePtr, u32 Mask);
u32 XRoformer_top_InterruptGetEnabled(XRoformer_top *InstancePtr);
u32 XRoformer_top_InterruptGetStatus(XRoformer_top *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
