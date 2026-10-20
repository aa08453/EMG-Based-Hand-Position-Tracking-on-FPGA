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
#include "normalization_constants.h"   // DELTA_SCALE, EMG_SCALE, ... (from the HLS export)


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

// --- Conversion helpers for the IP's data_t = ap_fixed<TOTAL_BITS, INT_BITS> (one 16-bit word per value) ---
// The format comes from normalization_constants.h (written by the same HLS export as the weights);
// the defaults are the v2 model's ap_fixed<10,3>.
#ifdef ACT_TOTAL_BITS
#define TOTAL_BITS   ACT_TOTAL_BITS
#define INT_BITS     ACT_INT_BITS
#else
#define TOTAL_BITS   10
#define INT_BITS     3
#endif
#define FRAC_BITS    (TOTAL_BITS - INT_BITS)  // 7 for <10,3>

int16_t float_to_fixed(float val) {
    float scaled = val * (1 << FRAC_BITS);

    // Clamp to the representable range of a TOTAL_BITS signed number (511 / -512 for 10 bits)
    float max_val = (float)((1 << (TOTAL_BITS - 1)) - 1);
    float min_val = -(float)(1 << (TOTAL_BITS - 1));

    if (scaled > max_val) scaled = max_val;
    if (scaled < min_val) scaled = min_val;

    // Round to nearest integer before casting to prevent truncation bias
    return (int16_t)roundf(scaled);
}

float fixed_to_float(int16_t val) {
    if (val & (1 << (TOTAL_BITS - 1))) {
        // Sign-extend: set the padding bits above TOTAL_BITS (0xFC00 for 10 bits)
        val |= (int16_t)~((1 << TOTAL_BITS) - 1);
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
// Sizes and the fusion gain come from normalization_constants.h (HLS export of the installed model);
// the fallbacks are the v2 model (100 ms windows, 25 ms step).
#ifdef MODEL_IN_WORDS
#define INPUT_FLOATS    MODEL_IN_WORDS  // 12 channels x MODEL_SEQ_LEN samples, channel-major
#define IP_OUTPUTS      MODEL_OUT_WORDS // RoFormer_Top: [abs angle (10) | delta / DELTA_SCALE (10)]
#else
#define INPUT_FLOATS    2400
#define IP_OUTPUTS      20
#endif
#define NUM_JOINTS      10
#define OUTPUT_FLOATS   30              // sent to PC: [abs (10) | delta (10) | fused estimate (10)]
#ifdef MODEL_FUSION_GAIN
#define FUSION_GAIN     MODEL_FUSION_GAIN   // steady-state gain of the Kalman filter tuned for this model
#else
#define FUSION_GAIN     0.056f          // v2 model at 25 ms steps, PROJECT_REPORT.md section 7
#endif
#define EXPECTED_BYTES  (INPUT_FLOATS * sizeof(float))
#define OUTPUT_BYTES    (OUTPUT_FLOATS * sizeof(float))

// Fusion state, carried across consecutive windows (one window per step: MODEL_STRIDE samples).
static float fused_est[NUM_JOINTS];
static int   fused_valid = 0;

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
                Xil_DCacheInvalidateRange(DDR_FPGA_OUT_ADDR, IP_OUTPUTS * sizeof(short));
                
                XGpio_DiscreteWrite(&Gpio, 1, LED_DONE);

                short *fpga_out_buffer = (short *)DDR_FPGA_OUT_ADDR;
                float *uart_tx_buffer  = (float *)DDR_UART_TX_ADDR;

                // abs head, delta head (rescaled to angle units), then fusion:
                //   est += delta;  est += K * (abs - est);   seeded from the first abs prediction
                for (int j = 0; j < NUM_JOINTS; j++) {
                    const float abs_pred = fixed_to_float(fpga_out_buffer[j]);
                    const float delta    = fixed_to_float(fpga_out_buffer[NUM_JOINTS + j]) * DELTA_SCALE[j];
                    if (!fused_valid) fused_est[j] = abs_pred;
                    else {
                        fused_est[j] += delta;
                        fused_est[j] += FUSION_GAIN * (abs_pred - fused_est[j]);
                    }
                    uart_tx_buffer[j]                  = abs_pred;
                    uart_tx_buffer[NUM_JOINTS + j]     = delta;
                    uart_tx_buffer[2 * NUM_JOINTS + j] = fused_est[j];
                }
                fused_valid = 1;
                
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