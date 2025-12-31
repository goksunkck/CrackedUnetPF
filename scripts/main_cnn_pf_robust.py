import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_attention import AttentionUNet
from src.utils.inference import extract_tip_coordinates
from src.particle_filter import CrackParticleFilter
from src.dataset import CrackDataset

# --- Configuration ---
# Plate Geometry
PLATE_X_MIN = -82.0
PLATE_X_MAX = 82.0
GRID_RES = 256

def pixel_to_physical(x_pix, y_pix):
    """
    Converts pixel (x,y) to physical (x_mm, y_mm).
    Assumes square grid mapped to [PLATE_X_MIN, PLATE_X_MAX].
    """
    width_mm = PLATE_X_MAX - PLATE_X_MIN
    scale = width_mm / GRID_RES
    
    x_mm = PLATE_X_MIN + x_pix * scale
    # Assuming Y is centered around 0? 
    # Usually DIC sets origin. Let's assume centered for Y too (common).
    # Height is usually same as Width in these datasets (256x256 crop).
    y_mm = (y_pix - (GRID_RES / 2)) * scale 
    return x_mm, y_mm

def get_crack_length_2d(tip_x_mm, tip_y_mm, origin_point):
    """
    Calculates crack length 'a' as Euclidean distance from the Crack Mouth (origin).
    origin_point: (x, y) tuple of the start of the crack (e.g. edge of plate).
    """
    dist = np.sqrt((tip_x_mm - origin_point[0])**2 + (tip_y_mm - origin_point[1])**2)
    return dist

def run_robust_tracking(dataset, dataset_name, model, device, output_dir):
    print(f"\n--- Running Robust Tracking on {dataset_name} ---")
    
    # 1. Initialize & Auto-Detect Origin
    # We scan the first few frames to find the tip.
    # If tip is closer to Left Edge -> Origin is Left.
    # If tip is closer to Right Edge -> Origin is Right.
    
    valid_tip = None
    first_frames_to_check = min(len(dataset), 20)
    
    print("Auto-detecting Reference Origin...")
    for i in range(first_frames_to_check):
        inp, _ = dataset[i]
        with torch.no_grad():
            out = model(inp.unsqueeze(0).to(device))
            mask = torch.argmax(out, dim=1).squeeze(0).cpu()
        tip = extract_tip_coordinates(mask)
        if tip:
            # tip is (y, x) pixels
            vx, vy = pixel_to_physical(tip[1], tip[0])
            valid_tip = (vx, vy)
            print(f"  Found start tip at (x={vx:.1f}, y={vy:.1f}) mm")
            break
            
    if valid_tip is None:
        print("  Could not detect valid tip in first frames. Defaulting to Left Edge.")
        valid_tip = (-50.0, 0.0) # Fallback
        
    # Determine Origin (Crack Mouth)
    # Distance to Left (-82) vs Right (+82)
    dist_left = abs(valid_tip[0] - PLATE_X_MIN)
    dist_right = abs(valid_tip[0] - PLATE_X_MAX)
    
    if dist_left < dist_right:
        origin = (PLATE_X_MIN, valid_tip[1]) # Assume straight crack from left edge at same Y
        print(f"  -> Origin set to LEFT EDGE: {origin}")
    else:
        origin = (PLATE_X_MAX, valid_tip[1]) # Assume straight crack from right edge at same Y
        print(f"  -> Origin set to RIGHT EDGE: {origin}")
        
    start_a = get_crack_length_2d(valid_tip[0], valid_tip[1], origin)
    print(f"  -> Initial Crack Length a0 = {start_a:.2f} mm")

    # 2. Setup Particle Filter
    # Adaptive Meas Noise (keeping what we learned)
    noise_map = {
        "S_160_4.7": 0.5,
        "S_160_2.0": 2.0,
        "S_950_1.6": 15.0
    }
    R = noise_map.get(dataset_name, 3.0)
    
    initial_state = {
        'a_mean': start_a, 'a_std': 1.0,
        'logC_mean': -11.0, 'logC_std': 1.0,
        'm_mean': 3.0, 'm_std': 0.2
    }
    
    pf = CrackParticleFilter(num_particles=5000, 
                             initial_state=initial_state,
                             process_noise_std=[0.1, 0.01, 0.01],
                             measurement_noise_std=R)
                             
    # Physics Model: dK matches geometrical length
    def stress_intensity(a_val):
        # Simplest dK model: K ~ sqrt(a)
        # a_val is now strictly "Distance from Mouth"
        L = np.maximum(a_val, 0.1) 
        return 5.0 * np.sqrt(L)

    # 3. Tracking Loop
    results_a = []
    rec_m = []
    
    last_valid_a = start_a
    # Threshold for monotonicity: 2mm (or slightly higher if 2D is noisier)
    MONO_THRESH = 2.0
    
    print(f"  > Using Measurement Noise Sigma (R) = {R:.2f} mm")
    print(f"  > Monotonicity Threshold = {MONO_THRESH:.2f} mm")

    rejection_count = 0
    
    for i in range(len(dataset)):
        if i % 100 == 0: print(f"  Step {i}/{len(dataset)}...")
        
        inputs, target = dataset[i]
        
        # Inference
        inputs_dev = inputs.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(inputs_dev)
            pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu()
            
        tip_pix = extract_tip_coordinates(pred_mask)
        
        meas_a = None
        if tip_pix:
            vx, vy = pixel_to_physical(tip_pix[1], tip_pix[0])
            dist = get_crack_length_2d(vx, vy, origin)
            
            # Monotonicity Check (Distance must grow)
            if dist < (last_valid_a - MONO_THRESH):
                meas_a = None # Reject back-jump
                rejection_count += 1
            else:
                meas_a = dist
                if meas_a > last_valid_a:
                    last_valid_a = meas_a
                    
        # PF Update
        pf.predict(dK_func=stress_intensity, cycles=100)
        if meas_a is not None:
            pf.update(meas_a)
        pf.resample()
        
        est, _ = pf.estimate()
        results_a.append(est[0])
        rec_m.append(est[2])
    
    print(f"  > Total Rejections: {rejection_count}/{len(dataset)} ({rejection_count/len(dataset)*100:.1f}%)")
        
    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(results_a, 'b-', label='PF Estimated Length (Robust)')
    plt.title(f"Robust Tracking: {dataset_name} (Origin: {origin})")
    plt.xlabel("Frame")
    plt.ylabel("Crack Length (from Origin) [mm]")
    plt.grid(True)
    plt.legend()
    
    out_path = os.path.join(output_dir, f"robust_track_{dataset_name}.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    print(f"Final Estimated m: {np.mean(rec_m[-50:]):.2f}")


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(PROJECT_ROOT, 'checkpoints', "attention_unet_best.pth")
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading Model...")
    model = AttentionUNet(n_channels=2, n_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    datasets_to_run = ["S_160_4.7", "S_160_2.0", "S_950_1.6"]
    
    for ds_name in datasets_to_run:
        data_dir = os.path.join(PROJECT_ROOT, "data", ds_name, "interim")
        input_path = os.path.join(data_dir, "lInputData_left.pt")
        
        if not os.path.exists(input_path):
            print(f"Skipping {ds_name}: Input not found at {input_path}")
            continue
            
        # load GT only if it exists
        gt_path = os.path.join(data_dir, "lGroundTruthData_left.pt")
        gt_paths_list = [gt_path] if os.path.exists(gt_path) else None
        
        dataset = CrackDataset([input_path], gt_paths_list)
        
        try:
            run_robust_tracking(dataset, ds_name, model, device, output_dir)
        except Exception as e:
            print(f"Failed to run tracking on {ds_name}: {e}")
            import traceback
            traceback.print_exc()
