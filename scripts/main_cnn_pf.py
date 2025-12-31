import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader

from src.model_attention import AttentionUNet
from src.utils.inference import extract_tip_coordinates
from src.particle_filter import CrackParticleFilter
from src.dataset import CrackDataset

def load_data(data_root, dataset_name):
    # Construct path for specific dataset
    dataset_dir = os.path.join(data_root, dataset_name, "interim")
    input_path = os.path.join(dataset_dir, "lInputData_left.pt")
    gt_path = os.path.join(dataset_dir, "lGroundTruthData_left.pt")
    
    if not os.path.exists(input_path):
        print(f"Warning: Input file not found for {dataset_name} at {input_path}")
        return None

    input_paths = [input_path]
    gt_paths = []
    if os.path.exists(gt_path):
        gt_paths = [gt_path]
    
    dataset = CrackDataset(input_paths, gt_paths)
    return dataset

# Physical Grid Parameters (approximate based on analysis)
GRID_X_MIN, GRID_X_MAX = -82.0, 82.0
GRID_RES = 256

def pixel_to_physical(x_pixel):
    """Converts pixel x-coordinate (0-255) to physical mm."""
    if x_pixel is None: return None
    scale = (GRID_X_MAX - GRID_X_MIN) / GRID_RES
    return GRID_X_MIN + x_pixel * scale

def run_tracking(dataset, dataset_name, model, device, output_dir):
    print(f"\n--- Running Tracking on {dataset_name} ---")
    sequence_length = len(dataset)
    print(f"Sequence Length: {sequence_length}")
    
    # Initialize PF
    # Get first measurement for initialization
    first_input, first_gt = dataset[0] # first_gt might be None
    
    # If GT is None, we MUST use Model Prediction on first frame to initialize
    # Or start from a default if model misses.
    start_a = -81.0 # Default fallback
    
    # Try model on first frame
    with torch.no_grad():
        first_in_dev = first_input.unsqueeze(0).to(device)
        first_out = model(first_in_dev)
        first_mask = torch.argmax(first_out, dim=1).squeeze(0).cpu()
        first_tip = extract_tip_coordinates(first_mask)
        if first_tip:
           start_a = pixel_to_physical(first_tip[1])
           print(f"Initialized 'a' from First Frame Prediction: {start_a:.2f}")
        elif first_gt is not None:
             # Fallback to GT if model fails but GT exists
             gt_tip = extract_tip_coordinates(first_gt)
             if gt_tip:
                 start_a = pixel_to_physical(gt_tip[1])
                 print(f"Initialized 'a' from Ground Truth: {start_a:.2f}")

    print(f"PF Initial State a0 = {start_a}")

    initial_state = {
        'a_mean': start_a, 'a_std': 0.5,
        'logC_mean': -11.0, 'logC_std': 0.5,
        'm_mean': 3.0, 'm_std': 0.1
    }
    
    # Adaptive Measurement Noise
    DATASET_NOISE_PARAMS = {
        "S_160_4.7": 0.5,
        "S_160_2.0": 2.0,
        "S_950_1.6": 15.0 
    }
    meas_R = DATASET_NOISE_PARAMS.get(dataset_name, 3.0)
    print(f"Using Measurement Noise Sigma (R) = {meas_R:.2f} mm")

    pf = CrackParticleFilter(num_particles=5000, initial_state=initial_state, 
                             process_noise_std=[0.5, 0.01, 0.01],
                             measurement_noise_std=meas_R)
                             
    def stress_intensity(a_coords):
        L = np.maximum(a_coords - (-82.0), 0.1)
        return 5.0 * np.sqrt(L)

    true_lengths = []
    cnn_lengths = []
    pf_estimates = []
    pf_std = []
    pf_dadn = []
    pf_logC = []
    pf_m = []
    
    has_gt = False
    
    # Monotonicity Constraint State
    last_valid_meas = start_a
    REJECTION_THRESHOLD = 10.0 # mm
    
    for i in range(sequence_length):
        if i % 50 == 0:
            print(f"[{dataset_name}] Step {i}/{sequence_length}...")
            
        inputs, target_mask = dataset[i]
        
        # --- CNN Step ---
        inputs_dev = inputs.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(inputs_dev)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu()
            
        tip_coords = extract_tip_coordinates(pred_mask)
        
        # GT Step
        gt_tip_coords = None
        if target_mask is not None:
            has_gt = True
            gt_tip_coords = extract_tip_coordinates(target_mask)
            
        # Convert
        meas_a = tensor_gt_a = None
        if tip_coords: meas_a = pixel_to_physical(tip_coords[1])
        if gt_tip_coords: tensor_gt_a = pixel_to_physical(gt_tip_coords[1])
        
        # --- Monotonicity Check ---
        if meas_a is not None:
            if meas_a < (last_valid_meas - REJECTION_THRESHOLD):
                # Reject measurement
                # print(f"  Step {i}: Rejected drop {meas_a:.2f} < {last_valid_meas:.2f}")
                meas_a = None
            else:
                # Accept and Update (even if it's slightly lower but within noise threshold, or higher)
                # We update last_valid_meas ONLY if it's an increase? 
                # Or just valid measurement? 
                # Standard is: Crack never shrinks. So valid measurement should ideally be >= last_valid
                # But allow small noise. 
                # Let's say: Update last_valid if it's > last_valid. 
                # If it's slightly less (noise), we use it for update but don't lower our "floor".
                if meas_a > last_valid_meas:
                    last_valid_meas = meas_a

        # Fallback GT for Plotting
        if tensor_gt_a is None and has_gt:
             tensor_gt_a = true_lengths[-1] if true_lengths else start_a
             
        # --- PF Step ---
        pf.predict(dK_func=stress_intensity, cycles=100)
        if meas_a is not None:
            pf.update(meas_a)
        pf.resample()
        est_state, est_var = pf.estimate()
        
        # Store
        START_X = -82.0
        
        len_gt = (tensor_gt_a - START_X) if tensor_gt_a is not None else np.nan
        len_cnn = (meas_a - START_X) if meas_a is not None else np.nan
        len_est = est_state[0] - START_X
        
        true_lengths.append(len_gt)
        cnn_lengths.append(len_cnn)
        pf_estimates.append(len_est)
        pf_std.append(np.sqrt(est_var[0]))
        pf_logC.append(est_state[1])
        pf_m.append(est_state[2])
        
        # da/dN
        est_dK = stress_intensity(np.array([est_state[0]]))
        est_dadn = (10**est_state[1]) * (est_dK.item() ** est_state[2])
        pf_dadn.append(est_dadn)

    # Plotting
    cycles = np.arange(sequence_length)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1
    if has_gt:
        ax1.plot(cycles, true_lengths, 'g-', label='Ground Truth', linewidth=2)
        
    ax1.plot(cycles, cnn_lengths, 'r.', label='CNN Meas', alpha=0.5)
    est = np.array(pf_estimates)
    std = np.array(pf_std)
    ax1.plot(cycles, est, 'b-', label='PF Est', linewidth=2)
    ax1.fill_between(cycles, est - 2*std, est + 2*std, color='b', alpha=0.2)
    ax1.set_ylabel('Crack Length [mm]')
    ax1.set_title(f'Tracking: {dataset_name}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2
    ax2.plot(cycles, pf_dadn, 'm-', label='Est da/dN')
    ax2.set_ylabel('Growth Rate')
    ax2.set_yscale('log')
    ax2.set_title("Growth Rate")
    ax2.grid(True)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"pf_tracking_{dataset_name}.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    
    # Stats
    last_n = min(50, len(pf_logC))
    avg_C = np.mean(pf_logC[-last_n:])
    avg_m = np.mean(pf_m[-last_n:])
    print(f"[{dataset_name}] Final Est: logC={avg_C:.2f}, m={avg_m:.2f}")


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_root = os.path.join(PROJECT_ROOT, "data")
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    model_path = os.path.join(PROJECT_ROOT, "checkpoints", "attention_unet_best_flip_1.pth")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Model Once
    print(f"Loading Model from {model_path}...")
    model = AttentionUNet(n_channels=2, n_classes=3).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("Model not found!")
        return
    model.eval()
    
    datasets_to_run = ["S_160_4.7", "S_160_2.0", "S_950_1.6"]
    
    for ds_name in datasets_to_run:
        dataset = load_data(data_root, ds_name)
        if dataset and len(dataset) > 0:
            run_tracking(dataset, ds_name, model, device, output_dir)
        else:
            print(f"Skipping {ds_name} (No data loaded)")

if __name__ == "__main__":
    main()
