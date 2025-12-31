import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.model import UNet

def load_model(model_path, device='cuda'):
    model = UNet(n_channels=2, n_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def extract_tip_coordinates(mask_pred):
    """
    Extracts crack tip coordinates from the predicted mask.
    Assumes Class 2 is the crack tip.
    Args:
        mask_pred: (H, W) tensor with class indices (0, 1, 2)
    Returns:
        (y, x) centroid of the tip class, or None if not found.
    """
    tip_indices = torch.argwhere(mask_pred == 2)
    if tip_indices.size(0) == 0:
        return None
    
    # Simple centroid
    centroid = tip_indices.float().mean(dim=0)
    return centroid[0].item(), centroid[1].item() # y, x

def visualize_prediction(model, data_dir, idx=100, device='cuda'):
    # Load single sample manually for control
    input_path = os.path.join(data_dir, "lInputData_left.pt")
    gt_path = os.path.join(data_dir, "lGroundTruthData_left.pt")
    
    inputs_list = torch.load(input_path)
    gts_list = torch.load(gt_path)
    
    # Get one sample
    inp_tensor = inputs_list[idx] # (1, 2, 256, 256)
    gt_tensor = gts_list[idx]     # (1, 256, 256)
    
    inp_dev = inp_tensor.to(device)
    
    with torch.no_grad():
        output = model(inp_dev) # (1, 3, 256, 256)
        probs = F.softmax(output, dim=1)
        pred_mask = torch.argmax(probs, dim=1).squeeze(0) # (256, 256)
        
    # Extract tip
    tip_coords = extract_tip_coordinates(pred_mask)
    
    # Plot
    img_in_cpu = inp_tensor.squeeze(0).cpu()
    gt_cpu = gt_tensor.squeeze(0).cpu()
    pred_cpu = pred_mask.cpu()
    
    plt.figure(figsize=(15, 5))
    
    # Input Strain Y
    plt.subplot(1, 4, 1)
    plt.imshow(img_in_cpu[1], cmap='jet')
    plt.title("Input Strain Y")
    plt.axis('off')

    # GT
    plt.subplot(1, 4, 2)
    plt.imshow(gt_cpu, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 4, 3)
    plt.imshow(pred_cpu, cmap='gray')
    plt.title("Prediction")
    
    if tip_coords:
        plt.plot(tip_coords[1], tip_coords[0], 'rx', markersize=10, markeredgewidth=2)
        plt.title(f"Pred Tip: ({tip_coords[1]:.1f}, {tip_coords[0]:.1f})")
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
    plt.savefig("inference_result.png")
    print(f"Saved inference result to inference_result.png. Tip Coords: {tip_coords}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    model_path = os.path.join(PROJECT_ROOT, "checkpoints", "unet_crack_segmentation_100epochs.pth")
    data_dir = os.path.join(PROJECT_ROOT, "data", "S_160_4.7", "interim")
    
    if not os.path.exists(model_path):
        print("Model file not found. Wait for training to finish.")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(model_path, device)
        visualize_prediction(model, data_dir, idx=150, device=device)
