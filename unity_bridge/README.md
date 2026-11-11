# EMG hand model to Unity

Runs the deployed sEMG hand model (`L3_w8`, ONNX) on a PC, fuses its two outputs with a Kalman filter, and streams 10 joint angles to Unity over UDP at 20 updates per second.

This folder is self-contained. Copy it anywhere. It needs Python 3.10 or newer and does not use PyTorch or any other file from this repository.

```
raw EMG (12 ch, 2 kHz) -> band-pass, rectify, scale, mu-law -> 200 ms window every 50 ms
    -> ONNX model -> abs + delta -> Kalman fusion -> UDP JSON -> Unity
```

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
.venv/bin/python verify.py                    # every line should say PASS
.venv/bin/python run.py --loop                # streams the bundled sample windows
```

In Unity (2021.3 or newer):

1. Copy `Unity/EmgUdpReceiver.cs` and `Unity/EmgProceduralHand.cs` into your project's `Assets` folder.
2. Create an empty GameObject at position (0, 0, 0) and add the `EmgProceduralHand` component. `EmgUdpReceiver` is added with it.
3. Press Play. The script builds a hand from capsules, so no model or rig is needed.

Start `run.py` before or after pressing Play. A label in the Game view shows the frame number, or a waiting message if nothing is arriving. If Unity is on another machine, pass `--host <its IP>` and allow UDP port 5005 through its firewall.

## Input sources

| Command | Input |
|---|---|
| `run.py --loop` | 40 bundled real windows. They are spread across a recording, so the hand jumps between unrelated poses. Use this to check the connection. |
| `run.py --source raw --file S1_E1_A1.mat --self-scale` | A continuous recording, replayed in real time through the full pipeline. Accepts a NinaPro DB2 `.mat`, or a `.npy` / `.csv` shaped `[samples, 12]`. |
| `run.py --source udp` | Live EMG. Send datagrams to port 5006 containing little-endian float32 samples, `[n x 12]` row-major, at 2 kHz, in the same units as DB2 (volts). |

For realistic motion, use a `.mat` file from the Kaggle dataset. A `.mat` also carries the glove recording, so setting **Signal** to `GroundTruth` on the Unity component shows the real hand for comparison. `--self-scale` computes the normalisation from that file. Use it for any subject other than the one in `model/runtime_config.json`.

## Packet format

Each UDP datagram to port 5005 is one JSON object:

| Field | Meaning |
|---|---|
| `seq`, `time`, `latency_ms` | Frame number, seconds since start, inference and fusion time |
| `fused`, `abs`, `truth` | Joint positions, normalised 0 to 1. `truth` is empty if the source has no glove data. |
| `delta` | Predicted change per 50 ms step, normalised units |
| `fused_angles`, `abs_angles`, `truth_angles` | The same positions in glove units, close to degrees |

To drive your own rigged hand, read `GetComponent<EmgUdpReceiver>().Latest.fused_angles` in `Update` and apply each value to the matching bone.

## Joint order

| Index | Joint | Index | Joint |
|---|---|---|---|
| 0 | Thumb MCP | 5 | Middle PIP |
| 1 | Thumb IP | 6 | Ring MCP |
| 2 | Index MCP | 7 | Ring PIP |
| 3 | Index PIP | 8 | Little MCP |
| 4 | Middle MCP | 9 | Little PIP |

This order is inferred from the standard CyberGlove sensor layout applied to the 10 glove columns used in training (0-based 1, 2, 4, 5, 7, 8, 11, 12, 15, 16). It has not been confirmed against the NinaPro sensor map. If a finger moves when it should not, compare against `GroundTruth` and reorder **Joint Order** on the Unity component. The joint names in `kaggle_scripts/emg_common.py` differ from this list, and the project report notes that they were never checked.

## Limits

- **New users:** the model was trained on 10 DB2 subjects (s1, s10 to s18) and does not transfer to unseen people. Leave-subjects-out R² was below zero for 9 of 10 subjects in the 1-layer study, so live EMG from a new person needs per-user fine-tuning. Rescaling alone will not fix it.
- **Replay accuracy:** repetitions 2 and 5 are the held-out test data. A whole `.mat` file also contains repetitions the model trained on, so replay looks better than held-out performance.
- **Normalisation:** `emg_scale`, `glove_min` and `glove_max` in `model/runtime_config.json` belong to one subject. Live input needs that person's values.
- **Angles:** glove units are not calibrated degrees. Adjust **Angle Gain** on the Unity component to taste. The outermost finger joints are not predicted, so the hand derives them from the joint before.

## Files

| Path | Purpose |
|---|---|
| `run.py` | Command-line runner and UDP sender |
| `emg_runtime.py` | Streaming preprocessing, ONNX inference, Kalman fusion |
| `verify.py` | Self-test against the FPGA fixed-point reference |
| `model/L3_w8.onnx` | Model. Input `emg` `[batch, 12, 400]`, outputs `abs` and `delta` `[batch, 10]` |
| `model/runtime_config.json` | Filter coefficients, normalisation, Kalman settings |
| `sample_data/hil_windows.npz` | 40 preprocessed real EMG windows and their reference outputs |
| `Unity/` | C# scripts for Unity |
