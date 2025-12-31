import torch
import os

def inspect_pt(filepath):
    print(f"Inspecting {filepath}...")
    try:
        data = torch.load(filepath)
        if isinstance(data, torch.Tensor):
            print(f"  Type: Tensor")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Min: {data.min()}, Max: {data.max()}")
            print(f"  Mean: {data.mean()}, Std: {data.std()}")
        elif isinstance(data, list):
             print(f"  Type: List of length {len(data)}")
             if len(data) > 0 and isinstance(data[0], torch.Tensor):
                 print(f"  Element 0 Shape: {data[0].shape}")
        else:
            print(f"  Type: {type(data)}")
    except Exception as e:
        print(f"  Error loading: {e}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    base_dir = os.path.join(PROJECT_ROOT, "data", "S_160_4.7", "interim")
    files = [
        "lInputData_left.pt",
        "lInputData_right.pt", 
        "lGroundTruthData_left.pt",
        "lGroundTruthData_right.pt"
    ]
    
    for f in files:
        inspect_pt(os.path.join(base_dir, f))
