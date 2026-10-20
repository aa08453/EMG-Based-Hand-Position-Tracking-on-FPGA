import serial
import numpy as np
import sys
import time

import json, os

# --- Config ---
COM_PORT = '/dev/ttyACM0'  # Update this
BAUD_RATE = 115200
# Input size of the installed model (hls_code/scripts/install_export.py writes this file): 12 x seq_len
_NC = 'hls_code/verification_data/normalization_constants.json'
INPUT_FLOATS = json.load(open(_NC)).get('in_words', 2400) if os.path.exists(_NC) else 2400
OUTPUT_FLOATS = 30 # [abs angle (10) | delta per step (10) | fused estimate (10)], normalised units
INPUT_BYTES = INPUT_FLOATS * 4
OUTPUT_BYTES = OUTPUT_FLOATS * 4

def loopback_test():
    try:
        # 2,400 floats = 9,600 bytes take about 0.85 s at 115200 baud (4,800 floats: 1.7 s): UART is for verification, not real time
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=3)
        print(f"[PC] Connected to {COM_PORT}")
    except Exception as e:
        print(f"Error opening port: {e}")
        sys.exit(1)

    # 1. Create dummy input data (e.g., an array of 0.5s)
    print(f"\n[PC] Sending {INPUT_FLOATS}  floats...")
    with open("hls_code/verification_data/full_input.txt", 'r') as file:
        content = list(map(float,file.read().split()))
        # print(content)
        content = np.array(content, dtype=np.float32)
        print(content)
    # content = np.arange(1, INPUT_FLOATS + 1, dtype=np.float32)
    raw_tx_bytes = content.tobytes()

    time.sleep(2)

    # 2. Clear buffers before sending
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # 3. Transmit
    print(f"[PC] Sending {len(raw_tx_bytes)} bytes to Zynq...")
    ser.write(raw_tx_bytes)
    ser.flush()  # Block until all bytes leave the PC

    # 4. Receive
    print(f"[PC] Waiting for {OUTPUT_BYTES} bytes of joint data...")
    raw_rx_bytes = ser.read(OUTPUT_BYTES)

    if len(raw_rx_bytes) == OUTPUT_BYTES:
        # Convert the received bytes back into floats
        joints = np.frombuffer(raw_rx_bytes, dtype=np.float32)
        print("\n[SUCCESS] Received outputs:")
        print(f"  {'joint':>5} {'abs':>8} {'delta':>9} {'fused':>8}")
        for i in range(10):
            print(f"  {i:>5} {joints[i]:8.4f} {joints[10 + i]:9.5f} {joints[20 + i]:8.4f}")
        # Reference for this window (float model): hls_code/weights/input_output.h, arrays `output` and `delta_output`
    else:
        print(f"\n[ERROR] Timeout! Received only {len(raw_rx_bytes)} / {OUTPUT_BYTES} bytes.")
        
    ser.close()

if __name__ == "__main__":
    loopback_test()