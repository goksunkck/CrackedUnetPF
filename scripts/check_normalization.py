import torch
import os

def check_normalization():
    path = r"c:/Users/Goksun/Desktop/üniversite/Year 5/Semester 9/EE571/cnn_pf_crack_length/data/S_950_1.6/interim/lInputData_left.pt"
    if not os.path.exists(path):
        print("File not found")
        return

    data = torch.load(path)
    # data is list of tensors
    # Concatenate specific channel to check stats
    
    # Assume shape (1, 2, H, W) or similar
    tensors = [t.squeeze(0) for t in data] 
    stack = torch.stack(tensors) # (N, 2, H, W)
    
    print(f"Shape: {stack.shape}")
    
    # Check Channel 0
    c0 = stack[:, 0, :, :]
    print(f"Channel 0: Min={c0.min():.4f}, Max={c0.max():.4f}, Mean={c0.mean():.4f}, Std={c0.std():.4f}")
    
    # Check Channel 1
    c1 = stack[:, 1, :, :]
    print(f"Channel 1: Min={c1.min():.4f}, Max={c1.max():.4f}, Mean={c1.mean():.4f}, Std={c1.std():.4f}")

if __name__ == "__main__":
    check_normalization()
