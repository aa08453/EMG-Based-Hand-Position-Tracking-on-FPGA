import torch
import numpy as np
import os
import RoFormer from roformer.py

# Assuming the RoFormer classes (RotaryEmbedding, RoFormer, etc.) 
# are in the same scope or imported.

def generate_golden_references(model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize model and load weights
    # Note: Ensure these parameters match your Config exactly
    model = RoFormer(
        num_channels=12, 
        time_steps=200, 
        num_layers=1, 
        num_heads=8, 
        d_model=128, 
        d_ff=256, 
        num_angles=10
    )
    
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    # Handle state_dict wrapping if necessary
    sd = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
    model.load_state_dict(sd)
    model.eval()

    # 2. Create a stable dummy input (SEQ_LEN=200, CHANNELS=12)
    # We use a fixed seed to ensure your HLS and PyTorch start at the same point
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 12, 200)
    
    # Save the input for HLS to read
    np.savetxt(os.path.join(output_dir, "full_input.txt"), 
               dummy_input.numpy().flatten(), fmt='%.10f')

    # 3. Functional Hook to extract intermediate outputs
    # This simulates your HLS "DEBUG_CHECKPOINTS"
    with torch.no_grad():
        # --- Conv1D ---
        x_conv = model.conv1d(dummy_input)
        # HLS expects [SEQ_LEN, D_MODEL] for most layers, 
        # so we handle the transpose carefully
        conv_out_hls_style = x_conv.squeeze(0).transpose(0, 1) 
        np.savetxt(os.path.join(output_dir, "conv1d_out.txt"), 
                   conv_out_hls_style.numpy().flatten(), fmt='%.10f')

        # --- RoPE ---
        x = x_conv.transpose(1, 2) # [1, 200, 128]
        seq_len = x.shape[1]
        cos, sin = model.rope(x, seq_len=seq_len)
        
        # We simulate the first layer's RoPE application
        # (Assuming HLS applies RoPE immediately before/inside MHA)
        q, k = apply_rotary_pos_emb(x, x, cos, sin)
        np.savetxt(os.path.join(output_dir, "l0_rope_out.txt"), 
                   q.squeeze(0).numpy().flatten(), fmt='%.10f')

        # --- MHA + AddNorm 1 ---
        # We manually step through the first layer's forward pass
        layer = model.layers[0]
        q_rope, k_rope = apply_rotary_pos_emb(x, x, cos, sin)
        
        # PyTorch MHA call
        attn_out = layer.self_attn(q_rope, k_rope, x)[0]
        x_mha = x + attn_out
        x_mha = layer.norm1(x_mha)
        np.savetxt(os.path.join(output_dir, "l0_add_norm1_out.txt"), 
                   x_mha.squeeze(0).numpy().flatten(), fmt='%.10f')

        # --- FFN + AddNorm 2 ---
        x_ffn = layer.linear2(layer.dropout(layer.activation(layer.linear1(x_mha))))
        x_ffn = x_mha + x_ffn
        x_ffn = layer.norm2(x_ffn)
        np.savetxt(os.path.join(output_dir, "l0_add_norm2_out.txt"), 
                   x_ffn.squeeze(0).numpy().flatten(), fmt='%.10f')

        # --- Pooling ---
        x_pool = x_ffn.mean(dim=1)
        np.savetxt(os.path.join(output_dir, "pool_out.txt"), 
                   x_pool.squeeze(0).numpy().flatten(), fmt='%.10f')

        # --- Final FC ---
        logits = model.fc(x_pool)
        np.savetxt(os.path.join(output_dir, "full_output.txt"), 
                   logits.squeeze(0).numpy().flatten(), fmt='%.10f')

    print(f"Golden references generated in: {output_dir}")
    print(f"Final Logits: {logits.numpy().flatten()[:5]}...")

# Run the extraction
generate_golden_references('roformer_qat_single_encoder_200.pt.zip', '../verification_data')