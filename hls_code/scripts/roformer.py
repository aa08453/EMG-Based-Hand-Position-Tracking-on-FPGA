import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

# Config
class Config:
    exp_name = 'roformer_sEMG'
    data_path = '/content/processed_ninapro_db2.npz'  # Processed NPZ file (sEMG and labels)
    save_dir = 'outputs'
    max_seq_len = 200  
    num_channels = 12  # sEMG channels
    num_angles = 10  # Output joint angles
    batch_size = 32
    lr = 1e-4
    epochs = 50
    seed = 42
    num_heads = 8
    num_layers = 1
    d_model = 128
    d_ff = 256
    dropout = 0.1
    patience = 6



# Step 2: RoPE (Rotary Position Embedding)
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, seq_len, d_model]
    # cos, sin: [seq_len, d_model]
    cos = cos.unsqueeze(0)  # [1, seq_len, d_model]
    sin = sin.unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RoFormerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.activation = nn.GELU()
        self.activation = nn.ReLU()
    def forward(self, src, cos=None, sin=None, src_mask=None, src_key_padding_mask=None):
        q = k = src
        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)  # no unsqueeze
        src2 = self.self_attn(q, k, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
class RoFormer(nn.Module):
    def __init__(self, num_channels=12, time_steps=100, num_layers=4, num_heads=8, d_model=128, d_ff=512, num_angles=10, dropout=0.1):
        super().__init__()
        self.conv1d = nn.Conv1d(num_channels, d_model, kernel_size=3, padding=1)  # Conv1D on sEMG channels
        self.rope = RotaryEmbedding(d_model)
        self.layers = nn.ModuleList([RoFormerLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_angles)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, channels, time_steps]
        x = self.conv1d(x)                # [batch, d_model, time_steps]
        x = x.transpose(1, 2)             # [batch, time_steps, d_model]
        x = self.dropout(x)
    
        seq_len = x.shape[1]
        cos, sin = self.rope(x, seq_len=seq_len)
    
        for layer in self.layers:
            x = layer(x, cos, sin)
    
        x = x.mean(dim=1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Verification helpers (golden vs HLS dump)
# ---------------------------------------------------------------------------

DEFAULT_VERIFICATION_DIR = "../verification_data"


@dataclass(frozen=True)
class LayerSpec:
    # Label used by your C++ debug checkpoints.
    label: str
    # Golden filename under verification_data/.
    golden_filename: str
    # Pass thresholds (MAE < threshold) and (cosine similarity > threshold).
    mae_threshold: float
    cosine_threshold: float


LAYER_SPECS: List[LayerSpec] = [
    LayerSpec("conv1d", "conv1d_out.txt", 0.02, 0.98),
    LayerSpec("rope", "l0_rope_out.txt", 0.05, 0.95),
    LayerSpec("mha", "l0_add_norm1_out.txt", 0.10, 0.90),
    LayerSpec("ffn", "l0_add_norm2_out.txt", 0.15, 0.85),
    LayerSpec("pool", "pool_out.txt", 0.15, 0.85),
    LayerSpec("full", "full_output.txt", 0.20, 0.80),
]


def get_layer_specs() -> List[LayerSpec]:
    return LAYER_SPECS


def load_txt_floats(path: str) -> List[float]:
    """
    Loads a text file containing space/newline separated floats.
    (Your generator produces single-line space-separated values.)
    """
    with open(path, "r", encoding="utf-8") as f:
        tokens = f.read().split()
    return [float(t) for t in tokens]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def verify_layer_outputs(
    verification_dir: str,
    hw_dir: str,
    *,
    fail_fast: bool = False,
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}

    for spec in get_layer_specs():
        golden_path = os.path.join(verification_dir, spec.golden_filename)
        hw_path = os.path.join(hw_dir, spec.golden_filename)

        if not os.path.exists(golden_path):
            raise FileNotFoundError(f"Missing golden file: {golden_path}")
        if not os.path.exists(hw_path):
            raise FileNotFoundError(f"Missing HW dump file: {hw_path}")

        golden = load_txt_floats(golden_path)
        hw = load_txt_floats(hw_path)

        if len(hw) != len(golden):
            # Don't silently continue on shape mismatch; report clearly.
            raise ValueError(
                f"Length mismatch for layer '{spec.label}': "
                f"golden={len(golden)} hw={len(hw)}"
            )

        n = len(golden)
        abs_diffs = [abs(h - r) for h, r in zip(hw, golden)]
        mae = sum(abs_diffs) / n
        rmse = math.sqrt(sum((h - r) ** 2 for h, r in zip(hw, golden)) / n)
        max_err = max(abs_diffs) if abs_diffs else 0.0
        cos = cosine_similarity(hw, golden)

        mae_pass = mae < spec.mae_threshold
        cos_pass = cos > spec.cosine_threshold
        overall = mae_pass and cos_pass

        results[spec.label] = {
            "layer": spec.label,
            "golden_file": spec.golden_filename,
            "n": n,
            "mae": mae,
            "rmse": rmse,
            "max_abs_err": max_err,
            "cosine_similarity": cos,
            "mae_threshold": spec.mae_threshold,
            "cosine_threshold": spec.cosine_threshold,
            "pass": overall,
        }

        status = "PASSED" if overall else "FAILED"
        print(
            f"[{status}] {spec.label:6s} "
            f"MAE={mae:.6f} (thr={spec.mae_threshold:.4f}) "
            f"Cos={cos:.6f} (thr={spec.cosine_threshold:.4f}) "
            f"RMSE={rmse:.6f} MaxErr={max_err:.6f}"
        )

        if fail_fast and not overall:
            break

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare RoFormer HLS dumped layer outputs vs verification_data golden tensors."
    )
    parser.add_argument(
        "--verification_dir",
        default=DEFAULT_VERIFICATION_DIR,
        help="Directory containing golden verification_data/*.txt",
    )
    parser.add_argument(
        "--hw_dir",
        default="vitis_outputs",
        help="Directory containing HW dump files written by the C++ verifier.",
    )
    parser.add_argument("--fail_fast", action="store_true", help="Stop at first failure.")
    args = parser.parse_args()

    verify_layer_outputs(args.verification_dir, args.hw_dir, fail_fast=args.fail_fast)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())