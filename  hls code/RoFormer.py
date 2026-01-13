import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import random
import os
from transformers import RoFormerModel, RoFormerConfig

# Config
class Config:
    exp_name = 'roformer_sEMG'
    data_path = '/content/processed_ninapro_db2.npz'  # Processed NPZ file (sEMG and labels)
    save_dir = 'outputs'
    max_seq_len = 400  # Time steps (50ms at 2kHz)
    num_channels = 8  # sEMG channels
    num_angles = 10  # Output joint angles
    batch_size = 32
    lr = 1e-4
    epochs = 50
    seed = 42
    num_heads = 4
    num_layers = 1
    d_model = 128
    d_ff = 256
    dropout = 0.1
    patience = 4

cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)



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


# Step 4: Training Loop
model = RoFormer(cfg.num_channels, cfg.max_seq_len, cfg.num_layers, cfg.num_heads, cfg.d_model, cfg.d_ff, cfg.num_angles, cfg.dropout).to(device)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model = model.to(device)


optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
# criterion = nn.MSELoss()
criterion = VelocityGuidedLoss(lambda_v = 0.1).to(device)
