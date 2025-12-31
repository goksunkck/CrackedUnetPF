import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.inference import extract_tip_coordinates

def estimate_noise():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(PROJECT_ROOT, "data", "S_160_4.7", "interim")
    
    # Load CNN Predictions (We need to run inference or assume we have them, 
    # but we don't have saved predictions. We have the model.)
    # Actually, simpler: Use the main_cnn_pf logic but just collect errors.
    
    # Let's import the model + load data logic
    from src.model_attention import AttentionUNet
    from src.dataset import CrackDataset
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Model
    model_path = os.path.join(PROJECT_ROOT, 'checkpoints', "attention_unet_best.pth")
    model = AttentionUNet(n_channels=2, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Data
    input_paths = [os.path.join(data_dir, "lInputData_left.pt")]
    gt_paths = [os.path.join(data_dir, "lGroundTruthData_left.pt")]
    dataset = CrackDataset(input_paths, gt_paths)
    
    print(f"Loaded {len(dataset)} samples.")
    
    errors = []
    
    # Physical Scale
    GRID_X_MIN, GRID_X_MAX = -82.0, 82.0
    GRID_RES = 256
    def pixel_to_physical(x_pixel):
        scale = (GRID_X_MAX - GRID_X_MIN) / GRID_RES
        return GRID_X_MIN + x_pixel * scale

    for i in range(len(dataset)):
        inputs, target_mask = dataset[i]
        
        # Prediction
        inputs_dev = inputs.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(inputs_dev)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu()
            
        tip = extract_tip_coordinates(pred_mask)
        gt_tip = extract_tip_coordinates(target_mask)
        
        if tip and gt_tip:
             err_pixel = tip[1] - gt_tip[1]
             meas_mm = pixel_to_physical(tip[1])
             gt_mm = pixel_to_physical(gt_tip[1])
             err_mm = meas_mm - gt_mm
             errors.append(err_mm)
             
        if i % 100 == 0:
            print(f"Processed {i}/{len(dataset)}...")
            
    errors = np.array(errors)
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    
    print("\n--- Measurement Noise Statistics ---")
    print(f"Mean Error: {mean_err:.4f} mm")
    print(f"Std Dev (sigma): {std_err:.4f} mm")
    print(f"3-Sigma (Recommended Threshold): {3*std_err:.4f} mm")
    
    plt.figure()
    plt.hist(errors, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Measurement Error Distribution (std={std_err:.2f}mm)")
    plt.xlabel("Error (Meas - GT) [mm]")
    plt.ylabel("Count")
    plt.axvline(mean_err, color='r', linestyle='dashed', linewidth=1)
    
    # Show recommended threshold
    plt.axvline(mean_err - 3*std_err, color='g', linestyle='dotted', label='-3 Sigma')
    
    out_path = os.path.join(PROJECT_ROOT, "outputs", "noise_estimation.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    estimate_noise()
