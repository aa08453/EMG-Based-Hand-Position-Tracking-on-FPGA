# Additional clean files
cmake_minimum_required(VERSION 3.16)

if("${CONFIG}" STREQUAL "" OR "${CONFIG}" STREQUAL "")
  file(REMOVE_RECURSE
  "/home/abdullah/Desktop/FYP/EMG-Based-Hand-Position-Tracking-on-FPGA/vitis_workspace/model1/ps7_cortexa9_0/standalone_ps7_cortexa9_0/bsp/include/sleep.h"
  "/home/abdullah/Desktop/FYP/EMG-Based-Hand-Position-Tracking-on-FPGA/vitis_workspace/model1/ps7_cortexa9_0/standalone_ps7_cortexa9_0/bsp/include/xiltimer.h"
  "/home/abdullah/Desktop/FYP/EMG-Based-Hand-Position-Tracking-on-FPGA/vitis_workspace/model1/ps7_cortexa9_0/standalone_ps7_cortexa9_0/bsp/include/xtimer_config.h"
  "/home/abdullah/Desktop/FYP/EMG-Based-Hand-Position-Tracking-on-FPGA/vitis_workspace/model1/ps7_cortexa9_0/standalone_ps7_cortexa9_0/bsp/lib/libxiltimer.a"
  )
endif()
