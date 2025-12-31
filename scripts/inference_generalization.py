import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_attention import AttentionUNet
from src.utils.inference import extract_tip_coordinates, load_model as base_load_model

def load_model(model_path, device='cuda'):
    print(f"Loading Attention U-Net from {model_path}...")
    model = AttentionUNet(n_channels=2, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def visualize_prediction(model, data_dir, dataset_name, idx=100, device='cuda', output_dir='outputs'):
    # Load single sample manually
    input_path = os.path.join(data_dir, "lInputData_left.pt")
    gt_path = os.path.join(data_dir, "lGroundTruthData_left.pt")
    
    if not os.path.exists(input_path):
        print(f"Skipping {dataset_name}: Input file not found.")
        return

    print(f"Loading data from {input_path}...")
    inputs_list = torch.load(input_path)
    
    gts_list = None
    if os.path.exists(gt_path):
        gts_list = torch.load(gt_path)
    else:
        print("No Ground Truth found. Inference only.")
    
    # Handle index out of bounds
    if idx >= len(inputs_list):
        idx = len(inputs_list) // 2
    
    print(f"Visualizing Index: {idx}")
    
    # Get one sample
    inp_tensor = inputs_list[idx] # (1, 2, 256, 256)
    gt_tensor = gts_list[idx] if gts_list is not None else None
    
    inp_dev = inp_tensor.to(device)
    
    with torch.no_grad():
        output = model(inp_dev) # (1, 3, 256, 256)
        probs = F.softmax(output, dim=1)
        pred_mask = torch.argmax(probs, dim=1).squeeze(0) # (256, 256)
        
    # Extract tip
    tip_coords = extract_tip_coordinates(pred_mask)
    
    # Plot
    img_in_cpu = inp_tensor.squeeze(0).cpu()
    gt_cpu = gt_tensor.squeeze(0).cpu() if gt_tensor is not None else None
    pred_cpu = pred_mask.cpu()
    
    plt.figure(figsize=(15, 5))
    
    # Input Strain Y
    plt.subplot(1, 4, 1)
    # Using channel 1 (epsy) as primary
    plt.imshow(img_in_cpu[1], cmap='jet')
    plt.title(f"{dataset_name}\nInput Strain Y")
    plt.axis('off')

    # GT
    plt.subplot(1, 4, 2)
    if gt_cpu is not None:
        plt.imshow(gt_cpu, cmap='gray')
        plt.title("Ground Truth")
    else:
        plt.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=20)
        plt.title("Ground Truth\n(Missing)")
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 4, 3)
    plt.imshow(pred_cpu, cmap='gray')
    plt.title("Prediction")
    
    if tip_coords:
        plt.plot(tip_coords[1], tip_coords[0], 'rx', markersize=10, markeredgewidth=2)
        plt.title(f"Pred Tip:\n({tip_coords[1]:.1f}, {tip_coords[0]:.1f})")
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 4, 4)
    plt.imshow(img_in_cpu[1], cmap='jet', alpha=0.6)
    plt.imshow(pred_cpu, cmap='gray', alpha=0.4)
    if tip_coords:
        plt.plot(tip_coords[1], tip_coords[0], 'rx', markersize=10, markeredgewidth=2)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    output_filename = os.path.join(output_dir, f"inference_{dataset_name}.png")
    plt.savefig(output_filename)
    print(f"Saved inference result to {output_filename}. Tip Coords: {tip_coords}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(PROJECT_ROOT, 'checkpoints', "attention_unet_best.pth") # Use best model
    datasets_dir = os.path.join(PROJECT_ROOT, "data")
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    
    datasets_to_test = ["S_160_4.7", "S_160_2.0", "S_950_1.6"]
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Wait for training to finish.")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(model_path, device)
        
        for ds_name in datasets_to_test:
            data_dir = os.path.join(datasets_dir, ds_name, "interim")
            if os.path.exists(data_dir):
                # Pick a frame index (try 200, or auto-adjust)
                visualize_prediction(model, data_dir, ds_name, idx=200, device=device, output_dir=output_dir)
            else:
                print(f"Skipping {ds_name}, interim folder not found: {data_dir}")
