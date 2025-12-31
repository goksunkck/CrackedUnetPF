import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_attention import AttentionUNet
from src.utils.inference import extract_tip_coordinates
from src.dataset import CrackDataset

def estimate_noise_unlabeled(dataset_name):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(PROJECT_ROOT, "data", dataset_name, "interim")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Model
    model_path = os.path.join(PROJECT_ROOT, 'checkpoints', "attention_unet_best.pth")
    model = AttentionUNet(n_channels=2, n_classes=3).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Model not found")
        return
    model.eval()
    
    # Load Data (Only inputs needed)
    input_path = os.path.join(data_dir, "lInputData_left.pt")
    if not os.path.exists(input_path):
        print(f"Data not found: {input_path}")
        return
        
    print(f"Loading {dataset_name}...")
    inputs_list = torch.load(input_path)
    # Squeeze to list of tensors
    inputs_list = [t.squeeze(0).float() for t in inputs_list]
    
    # Physical Scale
    GRID_X_MIN, GRID_X_MAX = -82.0, 82.0
    GRID_RES = 256
    def pixel_to_physical(x_pixel):
        scale = (GRID_X_MAX - GRID_X_MIN) / GRID_RES
        return GRID_X_MIN + x_pixel * scale

    raw_lengths = []
    
    print("Running Inference to extract raw signals...")
    for i, inp in enumerate(inputs_list):
        inp_dev = inp.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(inp_dev)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu()
            
        tip = extract_tip_coordinates(pred_mask)
        if tip:
            val = pixel_to_physical(tip[1])
            raw_lengths.append(val)
        else:
            raw_lengths.append(np.nan)
            
    raw_lengths = np.array(raw_lengths)
    
    # Filter NaNs
    valid_mask = ~np.isnan(raw_lengths)
    valid_lengths = raw_lengths[valid_mask]
    
    if len(valid_lengths) < 2:
        print("Not enough valid data points.")
        return

    # 1. Calculate Differences
    diffs = np.diff(valid_lengths)
    
    # 2. Physics check: Crack growth per step should be small and positive.
    # Large negative diffs are definitely noise.
    # Large positive diffs might be growth or noise.
    # But high-freq jitter dominates the std dev.
    
    std_diff = np.std(diffs)
    estimated_sigma = std_diff / np.sqrt(2)
    
    # count negative jumps
    neg_jumps = diffs[diffs < 0]
    large_neg_jumps = diffs[diffs < -2.0]
    
    print(f"\n--- Noise Analysis for {dataset_name} ---")
    print(f"Valid Samples: {len(valid_lengths)}/{len(raw_lengths)}")
    print(f"Est. Noise Sigma (Allen Variance): {estimated_sigma:.4f} mm")
    print(f"Negative Jumps: {len(neg_jumps)} ({len(neg_jumps)/len(diffs)*100:.1f}%)")
    print(f"Large Negative Jumps (<-2.0mm): {len(large_neg_jumps)}")
    if len(large_neg_jumps) > 0:
        print(f"  Max Drop: {np.min(large_neg_jumps):.4f} mm")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(raw_lengths, '.-')
    plt.title(f"Raw Measured Length: {dataset_name}")
    plt.ylabel("Length [mm]")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(diffs, 'r.-')
    plt.title("Step-to-Step Differences")
    plt.ylabel("Delta [mm]")
    plt.axhline(0, color='k')
    plt.axhline(-2.0, color='b', linestyle='--', label='-2mm Threshold')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    out_path = os.path.join(PROJECT_ROOT, "outputs", f"noise_analysis_{dataset_name}.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    estimate_noise_unlabeled("S_950_1.6")
