
import torch
import torch.nn as nn
import numpy as np
import os
from RoFormer import RotaryEmbedding, RoFormerBlock, Config, RoFormer  # Assuming RoFormer.py is in the same dir

# Configuration matching C++ Defines
SEQ_LEN = 100
D_MODEL = 128
NUM_HEADS = 4
D_FF = 256
IN_CHANNELS = 8
OUT_CHANNELS = 10 # num_angles

OUT_DIR = "verification_data"
os.makedirs(OUT_DIR, exist_ok=True)

def save_tensor(name, tensor):
    """Saves a tensor to a text file as flattened space-separated values."""
    path = os.path.join(OUT_DIR, f"{name}.txt")
    data = tensor.detach().cpu().numpy().flatten()
    np.savetxt(path, data, fmt='%.8f', newline=' ')
    print(f"Saved {name}: Shape {tensor.shape} -> {path}")

def generate_conv1d():
    print("\n--- Generating Conv1d Data ---")
    # C++: Conv1dLayer<float, IN_CHANNELS, D_MODEL, SEQ_LEN>
    # PyTorch: nn.Conv1d(in, out, kernel=3, padding=1)
    
    layer = nn.Conv1d(IN_CHANNELS, D_MODEL, kernel_size=3, padding=1)
    input_data = torch.randn(1, IN_CHANNELS, SEQ_LEN)
    
    output = layer(input_data)
    # Output shape: [1, D_MODEL, SEQ_LEN]
    # C++ expects Transposed output [SEQ_LEN][D_MODEL] 
    output_transposed = output.squeeze(0).transpose(0, 1)

    save_tensor("conv1d_input", input_data)
    save_tensor("conv1d_weights", layer.weight)
    save_tensor("conv1d_bias", layer.bias)
    save_tensor("conv1d_output", output_transposed)

def generate_rope():
    print("\n--- Generating RoPE Data ---")
    # C++: RotaryEmbedding<float, D_MODEL, SEQ_LEN>
    
    rope = RotaryEmbedding(D_MODEL)
    input_data = torch.randn(1, SEQ_LEN, D_MODEL) # Batch size 1
    
    # RoPE in PyTorch returns cos, sin
    cos, sin = rope(input_data, seq_len=SEQ_LEN)
    
    # But C++ RoPE class applies rotation internally. 
    # Let's apply it manually here to get expected output
    # q, k = apply_rotary_pos_emb(q, k, cos, sin)
    from RoFormer import apply_rotary_pos_emb
    
    # We will treat input as 'q' for verification
    # Input needs to be reshaped to keys/heads if we used the generic function,
    # but let's replicate the simpler logic: x * cos + rotate_half(x) * sin
    # Our C++ RoPE applies it to the full D_MODEL vector.
    
    # The PyTorch apply_rotary_pos_emb expects [batch, seq, heads, head_dim]?
    # No, check RoFormer.py:
    # def apply_rotary_pos_emb(q, k, cos, sin):
    #   q, k: [batch, seq_len, d_model] NO, usually split heads. 
    #   RoFormer.py: lines 83: # q, k: [batch, seq_len, d_model] <-- It says this!
    
    # So we can pass [1, SEQ, D_MODEL] directly.
    q_embed, _ = apply_rotary_pos_emb(input_data, input_data, cos, sin)
    
    save_tensor("rope_input", input_data)
    save_tensor("rope_output", q_embed)

def generate_layer_norm():
    print("\n--- Generating LayerNorm Data ---")
    ln = nn.LayerNorm(D_MODEL)
    input_data = torch.randn(1, SEQ_LEN, D_MODEL)
    output = ln(input_data)
    
    save_tensor("ln_input", input_data)
    save_tensor("ln_gamma", ln.weight)
    save_tensor("ln_beta", ln.bias)
    save_tensor("ln_output", output)

def generate_feed_forward():
    print("\n--- Generating FeedForward Data ---")
    # Linear -> ReLU -> Linear
    # Match RoFormerLayer structure
    linear1 = nn.Linear(D_MODEL, D_FF)
    relu = nn.ReLU()
    linear2 = nn.Linear(D_FF, D_MODEL)
    
    input_data = torch.randn(1, SEQ_LEN, D_MODEL) # Processed position-wise
    
    # Check if RoFormer.py uses dropout? Yes, but we disable for verification
    mid = relu(linear1(input_data))
    output = linear2(mid)
    
    save_tensor("ff_input", input_data)
    save_tensor("ff_w1", linear1.weight)
    save_tensor("ff_b1", linear1.bias)
    save_tensor("ff_w2", linear2.weight)
    save_tensor("ff_b2", linear2.bias)
    save_tensor("ff_output", output)

def generate_roformer_block():
    print("\n--- Generating RoFormerBlock Data ---")
    # We need to export ALL weights from a block
    block = RoFormerBlock(D_MODEL, NUM_HEADS, D_FF, dropout=0.0) # Disable dropout
    block.eval() 
    
    input_data = torch.randn(1, SEQ_LEN, D_MODEL)
    
    # RoPE needs to be computed externally in the PyTorch model loop
    rope = RotaryEmbedding(D_MODEL)
    cos, sin = rope(input_data, seq_len=SEQ_LEN)
    
    output = block(input_data, cos, sin)
    
    # Export Weights
    # MHA
    save_tensor("blk_mha_q_w", block.self_attn.in_proj_weight[:D_MODEL]) # PyTorch packs QKV usually?
    # Wait, nn.MultiheadAttention packs weights if batch_first=True?
    # We need to be careful here. 
    # RoFormer.py uses nn.MultiheadAttention(d_model, nhead...)
    # In PyTorch, in_proj_weight is [3*D_MODEL, D_MODEL].
    # It stores [Q_weight, K_weight, V_weight].
    
    qkv_w = block.self_attn.in_proj_weight
    qkv_b = block.self_attn.in_proj_bias
    
    save_tensor("blk_mha_q_w", qkv_w[:D_MODEL, :])
    save_tensor("blk_mha_q_b", qkv_b[:D_MODEL])
    
    save_tensor("blk_mha_k_w", qkv_w[D_MODEL:2*D_MODEL, :])
    save_tensor("blk_mha_k_b", qkv_b[D_MODEL:2*D_MODEL])
    
    save_tensor("blk_mha_v_w", qkv_w[2*D_MODEL:, :])
    save_tensor("blk_mha_v_b", qkv_b[2*D_MODEL:])
    
    save_tensor("blk_mha_o_w", block.self_attn.out_proj.weight)
    save_tensor("blk_mha_o_b", block.self_attn.out_proj.bias)
    
    # LN1
    save_tensor("blk_ln1_g", block.norm1.weight)
    save_tensor("blk_ln1_b", block.norm1.bias)
    
    # FF
    save_tensor("blk_ff_w1", block.linear1.weight)
    save_tensor("blk_ff_b1", block.linear1.bias)
    save_tensor("blk_ff_w2", block.linear2.weight)
    save_tensor("blk_ff_b2", block.linear2.bias)
    
    # LN2
    save_tensor("blk_ln2_g", block.norm2.weight)
    save_tensor("blk_ln2_b", block.norm2.bias)
    
    save_tensor("blk_input", input_data)
    save_tensor("blk_output", output)

if __name__ == "__main__":
    torch.manual_seed(42)
    generate_conv1d()
    generate_rope()
    generate_layer_norm()
    generate_feed_forward()
    generate_roformer_block()
    print("\nGeneration Complete. Zip 'verification_data/' and upload to C++ environment.")
