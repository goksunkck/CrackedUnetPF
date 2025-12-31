import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_attention_group import AttentionUNetGN
from src.utils.inference import extract_tip_coordinates, load_model as base_load_model

def make_gif(model, data_dir, dataset_name, device='cuda', output_path='outputs/video.gif'):
    print(f"Creating GIF for {dataset_name} from {data_dir}...")
    
    input_path = os.path.join(data_dir, "lInputData_left.pt")
    if not os.path.exists(input_path):
        print("Input file not found.")
        return

    inputs_list = torch.load(input_path)
    N = len(inputs_list)
    print(f"Dataset size: {N} frames")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    def update(frame_idx):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        inp_tensor = inputs_list[frame_idx]
        inp_dev = inp_tensor.to(device)
        
        with torch.no_grad():
            output = model(inp_dev)
            probs = F.softmax(output, dim=1)
            pred_mask = torch.argmax(probs, dim=1).squeeze(0)
            
        tip = extract_tip_coordinates(pred_mask)
        
        img_cpu = inp_tensor.squeeze(0).cpu()
        pred_cpu = pred_mask.cpu()
        
        # 1. Input
        ax1.imshow(img_cpu[1], cmap='jet')
        ax1.set_title(f"Strain Y (Frame {frame_idx})")
        ax1.axis('off')
        
        # 2. Prediction
        ax2.imshow(pred_cpu, cmap='gray')
        ax2.set_title("Prediction")
        ax2.axis('off')
        
        # 3. Overlay
        ax3.imshow(img_cpu[1], cmap='jet', alpha=0.6)
        ax3.imshow(pred_cpu, cmap='gray', alpha=0.4)
        if tip:
            ax3.plot(tip[1], tip[0], 'rx', markersize=10, markeredgewidth=2)
            ax3.set_title(f"Tip: ({tip[1]:.1f}, {tip[0]:.1f})")
        else:
            ax3.set_title("Tip: None")
        ax3.axis('off')
        
        return ax1, ax2, ax3

    anim = FuncAnimation(fig, update, frames=range(0, N, 2), interval=100) # Skip every 2nd frame for speed
    
    print(f"Saving GIF to {output_path}...")
    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer)
    print("Done!")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GIF for a specific experiment dataset.")
    parser.add_argument("--experiment", type=str, required=True, help="Name of the dataset folder (e.g., S_160_4.7, S_950_1.6)")
    args = parser.parse_args()
    
    dataset_name = args.experiment

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(PROJECT_ROOT, 'checkpoints', "attention_unet_groupnorm_best.pth")
    data_dir = os.path.join(PROJECT_ROOT, "data", dataset_name, "interim")
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # Check data exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading model from {model_path}...")
    model = AttentionUNetGN(n_channels=2, n_classes=3).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Model file not found!")
        sys.exit(1)
    model.eval()
    
    gif_path = os.path.join(output_dir, f"{dataset_name}_behavior.gif")
    make_gif(model, data_dir, dataset_name, device, gif_path)
