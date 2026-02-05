import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import sys

# No need to import as it exists already in Kaggle environment
# try:
#     from RoFormer import RoFormer, RoFormerBlock, RoFormerLayer, RotaryEmbedding, apply_rotary_pos_emb
# except ImportError:
#     print("Error: RoFormer.py not found or could not import classes.")
#     print("Make sure RoFormer.py is in the same directory as this script.")
#     sys.exit(1)
# except Exception as e:
#     print(f"Warning during import: {e}")

# Configuration matching C++ Defines
SEQ_LEN = 400
D_MODEL = 128
NUM_HEADS = 4
D_FF = 256
IN_CHANNELS = 12
OUT_CHANNELS = 10 # num_angles
NUM_LAYERS = 1

OUT_DIR = "verification_data"
os.makedirs(OUT_DIR, exist_ok=True)

def save_tensor(name, tensor):
    """Saves a tensor to a text file as flattened space-separated values."""
    path = os.path.join(OUT_DIR, f"{name}.txt")
    if isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy().flatten()
    else:
        data = np.array(tensor).flatten()
    
    np.savetxt(path, data, fmt='%.8f', newline=' ')
    print(f"Saved {name}: Shape {tensor.shape if hasattr(tensor, 'shape') else 'N/A'} -> {path}")

def generate_roformer_full(weights_path=None):
    print("\n--- Generating Full RoFormer Data ---")
    
    # Model already instantiated in Kaggle environment
    # model = RoFormer(
    #     num_channels=IN_CHANNELS,
    #     time_steps=SEQ_LEN,
    #     num_layers=NUM_LAYERS,
    #     num_heads=NUM_HEADS,
    #     d_model=D_MODEL,
    #     d_ff=D_FF,
    #     num_angles=OUT_CHANNELS,
    #     dropout=0.0
    # )
    
    if weights_path:
        print(f"Loading weights from: {weights_path}")
        try:
            obj = torch.load(weights_path, map_location='cpu')
            if isinstance(obj, dict) and "state_dict" in obj:
                state_dict = obj["state_dict"]
            elif isinstance(obj, dict):
                state_dict = obj
            else:
                state_dict = obj.state_dict()

            
            # Handle DataParallel wrapping if keys start with 'module.'
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
                    
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

            print("Weights loaded successfully!")
        except Exception as e:
            print(f"ERROR: Failed to load weights: {e}")
            sys.exit(1)
    else:
        print("Using Random Weights")

    model.eval()
    
    # Input: [1, IN_CH, SEQ_LEN]
    # Use random input for verification trace
    input_data = torch.randn(1, IN_CHANNELS, SEQ_LEN)
    
    with torch.no_grad():
        output = model(input_data)
    
    save_tensor("full_input", input_data)
    save_tensor("full_output", output)
    
    # Full Weights extraction
    save_tensor("full_conv_w", model.conv1d.weight)
    save_tensor("full_conv_b", model.conv1d.bias)
    
    for i in range(NUM_LAYERS):
        layer = model.layers[i]
        
        # Multi-Head Attention Weights (in_proj is combined q,k,v)
        # Check if in_proj_weight exists (PyTorch MHA)
        if hasattr(layer.self_attn, 'in_proj_weight') and layer.self_attn.in_proj_weight is not None:
             qkv_w = layer.self_attn.in_proj_weight
             qkv_b = layer.self_attn.in_proj_bias
             
             save_tensor(f"full_l{i}_mha_q_w", qkv_w[:D_MODEL, :])
             save_tensor(f"full_l{i}_mha_q_b", qkv_b[:D_MODEL])
             save_tensor(f"full_l{i}_mha_k_w", qkv_w[D_MODEL:2*D_MODEL, :])
             save_tensor(f"full_l{i}_mha_k_b", qkv_b[D_MODEL:2*D_MODEL])
             save_tensor(f"full_l{i}_mha_v_w", qkv_w[2*D_MODEL:, :])
             save_tensor(f"full_l{i}_mha_v_b", qkv_b[2*D_MODEL:])
        else:
             # If separately defined (q_proj_weight etc), handle here
             # RoFormer.py uses nn.MultiheadAttention, so it should be in_proj unless qkv_bias=False separate
             pass

        save_tensor(f"full_l{i}_mha_o_w", layer.self_attn.out_proj.weight)
        save_tensor(f"full_l{i}_mha_o_b", layer.self_attn.out_proj.bias)
        
        save_tensor(f"full_l{i}_ln1_g", layer.norm1.weight)
        save_tensor(f"full_l{i}_ln1_b", layer.norm1.bias)
        
        save_tensor(f"full_l{i}_ff_w1", layer.linear1.weight)
        save_tensor(f"full_l{i}_ff_b1", layer.linear1.bias)
        save_tensor(f"full_l{i}_ff_w2", layer.linear2.weight)
        save_tensor(f"full_l{i}_ff_b2", layer.linear2.bias)
        
        save_tensor(f"full_l{i}_ln2_g", layer.norm2.weight)
        save_tensor(f"full_l{i}_ln2_b", layer.norm2.bias)

    save_tensor("full_fc_w", model.fc.weight)
    save_tensor("full_fc_b", model.fc.bias)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HLS verification data")
    parser.add_argument("--weights", type=str, default="/kaggle/input/roformer/pytorch/default/2/best_roformer.pt", help="Path to model weights (.pth)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    generate_roformer_full(args.weights)
    print("\nGeneration Complete. Zip 'verification_data/' and upload to C++ environment.")
