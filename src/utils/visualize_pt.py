import torch
import matplotlib.pyplot as plt
import os

def visualize_sample(base_dir, idx=0):
    input_path = os.path.join(base_dir, "lInputData_left.pt")
    gt_path = os.path.join(base_dir, "lGroundTruthData_left.pt")
    
    print("Loading data...")
    inputs = torch.load(input_path)
    gts = torch.load(gt_path)
    
    sample_in = inputs[idx]  # Shape (1, 2, 256, 256)
    sample_gt = gts[idx]     # Shape (1, 256, 256)
    
    # Remove batch dim
    img_in = sample_in[0] # (2, 256, 256)
    img_gt = sample_gt[0] # (256, 256)
    
    print(f"Sample {idx} Input Min/Max: {img_in.min()}/{img_in.max()}")
    print(f"Sample {idx} GT Min/Max: {img_gt.min()}/{img_gt.max()}")
    print(f"Sample {idx} GT Unique Values: {torch.unique(sample_gt)}")

    plt.figure(figsize=(15, 5))
    
    # Channel 0 (epsx?)
    plt.subplot(1, 3, 1)
    plt.imshow(img_in[0], cmap='jet')
    plt.colorbar()
    plt.title(f"Input Ch0 (epsx?)")
    
    # Channel 1 (epsy?)
    plt.subplot(1, 3, 2)
    plt.imshow(img_in[1], cmap='jet')
    plt.colorbar()
    plt.title(f"Input Ch1 (epsy?)")
    
    # GT
    plt.subplot(1, 3, 3)
    plt.imshow(img_gt, cmap='gray')
    plt.colorbar()
    plt.title(f"Ground Truth")
    
    plt.savefig("tensor_viz.png")
    print("Saved tensor_viz.png")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    base_dir = os.path.join(PROJECT_ROOT, "data", "S_160_4.7", "interim")
    visualize_sample(base_dir, idx=100) # Pick a middle sample
