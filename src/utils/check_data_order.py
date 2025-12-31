import torch
import matplotlib.pyplot as plt
import os
import sys

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.dataset import CrackDataset
from src.utils.inference import extract_tip_coordinates

def check_order():
    data_dir = os.path.join(PROJECT_ROOT, "data", "S_160_4.7", "interim")
    input_paths = [os.path.join(data_dir, "lInputData_left.pt")]
    gt_paths = [os.path.join(data_dir, "lGroundTruthData_left.pt")]
    
    dataset = CrackDataset(input_paths, gt_paths)
    
    print(f"Total samples: {len(dataset)}")
    
    x_coords = []
    
    # Check first 200 samples
    N = min(len(dataset), 200)
    print(f"Checking first {N} samples...")
    
    for i in range(N):
        _, target = dataset[i]
        coords = extract_tip_coordinates(target)
        if coords:
            x_coords.append(coords[1]) # x coordinate
        else:
            x_coords.append(0) # Missing
            
    plt.figure(figsize=(10, 5))
    plt.plot(x_coords, '.-')
    plt.title("Ground Truth Tip X-Coordinate Sequence")
    plt.xlabel("Index")
    plt.ylabel("Tip X (pixels)")
    plt.grid(True)
    plt.savefig("data_sequence_check.png")
    print("Saved data_sequence_check.png")

if __name__ == "__main__":
    check_order()
