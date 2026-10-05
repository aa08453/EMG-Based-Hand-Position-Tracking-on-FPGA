#include "xparameters.h"
#include "xil_printf.h"
#include "xstatus.h"
#include "xil_cache.h"
#include "xroformer_top.h"
#include "xgpio.h"        
#include <stddef.h>
#include <stdlib.h>
#include "stdint.h"
#include <stdio.h>
#include "platform.h"
#include "xuartps.h"
#include "xscugic.h"
#include <math.h>


int init_uart();

#define CACHE_SAFE_OUTPUT_BYTES 64


#define GPIO_ID XPAR_AXI_GPIO_0_BASEADDR
#define ROFORMER_ID XPAR_ROFORMER_TOP_0_BASEADDR
#define UART_ID XPAR_XUARTPS_0_BASEADDR


// ==========================================
// 1. HARDWARE MEMORY MAP
// ==========================================
#define DDR_UART_RX_ADDR  0x0A000000 // Where UART writes 32-bit floats from PC
#define DDR_FPGA_IN_ADDR  0x0A010000 // Where ARM writes 16-bit fixed-point for FPGA
#define DDR_FPGA_OUT_ADDR 0x0C000000 // Where FPGA writes 16-bit fixed-point results
#define DDR_UART_TX_ADDR  0x0C010000 // Where ARM writes 32-bit floats to send to PC

// --- Conversion helpers (example for ap_fixed<16,8>) ---
// CHANGE THESE TO MATCH YOUR data_t:
#define TOTAL_BITS   10
#define INT_BITS     3
#define FRAC_BITS    (TOTAL_BITS - INT_BITS)  // = 7

int16_t float_to_fixed(float val) {
    float scaled = val * (1 << FRAC_BITS);  // multiply by 128 for <10,3>
    
    // Clamp to representable range for a 10-bit signed number
    // Max = 511, Min = -512
    float max_val = 511.0f;
    float min_val = -512.0f;
    
    if (scaled > max_val) scaled = max_val;
    if (scaled < min_val) scaled = min_val;
    
    // FIX: Round to nearest integer before casting to prevent truncation bias
    return (int16_t)roundf(scaled); 
}

float fixed_to_float(int16_t val) {
    if (val & 0x0200) { 
        // 0xFC00 in binary is 1111 1100 0000 0000
        // This forces the top 6 padding bits to become 1s, restoring the negative sign.
        val |= 0xFC00;  
    }
    
    return (float)val / (float)(1 << FRAC_BITS);
}



// LED Bitmasks (Assuming LED0 is bit 0, LED1 is bit 1)
#define LED_SENDING 0x01  // 0001 in binary
#define LED_DONE    0x02  // 0010 in binary
#define LED_OFF     0x00


XRoformer_top Model;    // Roformer instance 

// --- Hardware IDs ---
#define UART_DEVICE_ID  XPAR_UART1_BASEADDR
#define INTC_DEVICE_ID  XPAR_XSCUGIC_0_BASEADDR
#define UART_INT_IRQ_ID 82

// --- System Sizing ---
#define INPUT_FLOATS    2400
#define OUTPUT_FLOATS   10
#define EXPECTED_BYTES  (INPUT_FLOATS * sizeof(float))
#define OUTPUT_BYTES    (OUTPUT_FLOATS * sizeof(float))

// --- Global Instances ---
XUartPs Uart_Ps;
XScuGic InterruptController;

// --- State Machine Variables ---
volatile int packet_ready_flag = 0;
volatile u32 actual_bytes_received = 0;



// ==========================================
// 1. The Interrupt Service Routine (ISR)
// ==========================================
void UartInterruptHandler(void *CallBackRef, u32 Event, unsigned int EventData) {
    if (Event == XUARTPS_EVENT_RECV_DATA) {
        actual_bytes_received = EventData; // Reached exactly 19,200
        packet_ready_flag = 1;
    }
    else if (Event == XUARTPS_EVENT_RECV_TOUT) {
        // SHIELD: Only trigger the main loop if we ACTUALLY caught stray bytes.
        // If EventData is 0, it was a phantom interrupt. Ignore it.
        if (EventData > 0) {
            actual_bytes_received = EventData;
            packet_ready_flag = 1; 
        }
    }
}

int SetupInterruptSystem() {
    int Status;
    XScuGic_Config *IntcConfig = XScuGic_LookupConfig(INTC_DEVICE_ID); 
    if (NULL == IntcConfig) return XST_FAILURE;

    Status = XScuGic_CfgInitialize(&InterruptController, IntcConfig, IntcConfig->CpuBaseAddress);
    if (Status != XST_SUCCESS) return XST_FAILURE;

    Xil_ExceptionInit();
    Xil_ExceptionRegisterHandler(XIL_EXCEPTION_ID_INT,
                                 (Xil_ExceptionHandler)XScuGic_InterruptHandler,
                                 &InterruptController);

    Status = XScuGic_Connect(&InterruptController, UART_INT_IRQ_ID,
                             (Xil_ExceptionHandler)XUartPs_InterruptHandler,
                             (void *)&Uart_Ps);
    if (Status != XST_SUCCESS) return XST_FAILURE;

    // Connect our custom callback to the smart driver
    XUartPs_SetHandler(&Uart_Ps, (XUartPs_Handler)UartInterruptHandler, &Uart_Ps);

    // FIXED: Removed the toxic RXEMPTY mask. 
    // RXOVR = Trigger when FIFO hits 32 bytes. TOUT = Trigger if line goes quiet.
    XUartPs_SetInterruptMask(&Uart_Ps, XUARTPS_IXR_RXOVR | XUARTPS_IXR_TOUT);
    
    XScuGic_Enable(&InterruptController, UART_INT_IRQ_ID);
    Xil_ExceptionEnable();

    return XST_SUCCESS;
}

int main() 
{
    init_platform();
    
    xil_printf("\n\n=================================\r\n");
    xil_printf("       BOOTING SYSTEM\r\n");
    xil_printf("=================================\r\n");

    XGpio_Config *cfg_ptr;
    XGpio Gpio;            

    cfg_ptr = XGpio_LookupConfig(GPIO_ID);
    XGpio_CfgInitialize(&Gpio, cfg_ptr, cfg_ptr->BaseAddress);
    XGpio_SetDataDirection(&Gpio, 1, 0x00); 
    XGpio_DiscreteWrite(&Gpio, 1, LED_OFF); 

    //============== UART Configuration =======================//
    XUartPs_Config *Config = XUartPs_LookupConfig(UART_DEVICE_ID);
    XUartPs_CfgInitialize(&Uart_Ps, Config, Config->BaseAddress);
    XUartPs_SetBaudRate(&Uart_Ps, 115200);
    XUartPs_SetFifoThreshold(&Uart_Ps, 32); 
    XUartPs_SetOperMode(&Uart_Ps, XUARTPS_OPER_MODE_NORMAL);

    if (SetupInterruptSystem() != XST_SUCCESS) {
        xil_printf("[FATAL] Interrupt Setup Failed!\r\n");
        return XST_FAILURE;
    }
    XUartPs_SetRecvTimeout(&Uart_Ps, 8);

    
    // ======================= Roformer Setup ==============================
xil_printf("Initializing RoFormer IP...\r\n");
    XRoformer_top_Config *CfgPtr = XRoformer_top_LookupConfig(ROFORMER_ID);
    XRoformer_top_CfgInitialize(&Model, CfgPtr);

    // FIX 1: Point FPGA to the translated 16-bit arrays!
    XRoformer_top_Set_input_axi(&Model, DDR_FPGA_IN_ADDR);
    XRoformer_top_Set_output_axi(&Model, DDR_FPGA_OUT_ADDR); 

    xil_printf("\n--- System Ready for Real-Time Inference ---\r\n");
    xil_printf("Waiting for %d bytes from PC...\r\n", EXPECTED_BYTES);
    
    // FIX 2: Point UART receiver to the raw float array!
    XUartPs_Recv(&Uart_Ps, (u8*)DDR_UART_RX_ADDR, EXPECTED_BYTES);

    while(1) 
    {
        if (packet_ready_flag == 1) 
        {
            if (actual_bytes_received == EXPECTED_BYTES) 
            {
                XGpio_DiscreteWrite(&Gpio, 1, LED_SENDING);

                float *uart_rx_buffer = (float *)DDR_UART_RX_ADDR;
                short *fpga_in_buffer = (short *)DDR_FPGA_IN_ADDR;

                // FIX 3: Use the float_to_fixed function to prevent truncation
                for(int i = 0; i < INPUT_FLOATS; i++) {
                    fpga_in_buffer[i] = float_to_fixed(uart_rx_buffer[i]);
                }

                // FLUSH cache so FPGA sees the new 16-bit data
                Xil_DCacheFlushRange(DDR_FPGA_IN_ADDR, INPUT_FLOATS * sizeof(short));

                // FIRE FPGA
                XRoformer_top_Start(&Model);
                while (!XRoformer_top_IsDone(&Model));

                // INVALIDATE cache so CPU sees new data from FPGA
                Xil_DCacheInvalidateRange(DDR_FPGA_OUT_ADDR, OUTPUT_FLOATS * sizeof(short));
                
                XGpio_DiscreteWrite(&Gpio, 1, LED_DONE);

                short *fpga_out_buffer = (short *)DDR_FPGA_OUT_ADDR;
                float *uart_tx_buffer  = (float *)DDR_UART_TX_ADDR;

                // printf("\r\n[ZYNQ] Math Complete. Reading RAM:\r\n");
                for (int i = 0; i < OUTPUT_FLOATS; i++) {
                    // Use your fixed_to_float helper here too!
                    uart_tx_buffer[i] = fixed_to_float(fpga_out_buffer[i]);
                    // printf("  RAM[%d]: %f\r\n", i, uart_tx_buffer[i]);
                }
                // printf("==========================================\r\n");
                
                // SEND REAL DATA BACK TO PC
                XUartPs_Send(&Uart_Ps, (u8*)uart_tx_buffer, OUTPUT_BYTES);
            }
            
            packet_ready_flag = 0;
            
            // FIX 4: Re-prime the receiver to the correct UART buffer
            XUartPs_Recv(&Uart_Ps, (u8*)DDR_UART_RX_ADDR, EXPECTED_BYTES);
        }
    }

    // Note: You will never reach here unless you add a break condition to the while loop
    // SD_Eject();
    return 0;
}