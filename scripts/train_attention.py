import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import CrackDataset
from src.model_attention import AttentionUNet
import matplotlib.pyplot as plt

# --- Custom Loss Functions ---

class FocalTverskyLoss(nn.Module):
    def __init__(self, delta=0.7, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Calculate Tversky Index for each class
        # TP, FP, FN calculated over (B, H, W) for each channel C
        tp = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        fp = (probs * (1 - targets_one_hot)).sum(dim=(0, 2, 3))
        fn = ((1 - probs) * targets_one_hot).sum(dim=(0, 2, 3))
        
        tversky_index = (tp + self.smooth) / (tp + self.delta * fn + (1 - self.delta) * fp + self.smooth)
        
        # Focal Tversky: (1 - TI)^gamma
        focal_tversky_loss = (1 - tversky_index) ** self.gamma
        
        return focal_tversky_loss.mean()

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss with gamma=2.0.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super(AsymmetricFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        # Cross Entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Focal weights
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
             focal_loss = self.alpha * focal_loss
             
        return focal_loss.mean()

class OptimizedHybridLoss(nn.Module):
    """
    Combines Focal Tversky Loss and Asymmetric Focal Loss.
    """
    def __init__(self):
        super(OptimizedHybridLoss, self).__init__()
        self.ftl = FocalTverskyLoss(delta=0.7, gamma=0.75)
        self.afl = AsymmetricFocalLoss(gamma=2.0)

    def forward(self, inputs, targets):
        loss_ftl = self.ftl(inputs, targets)
        loss_afl = self.afl(inputs, targets)
        return loss_ftl + loss_afl

# --- Training Loop ---

def train_attention_model(data_dir, num_epochs=100, batch_size=16, learning_rate=1e-3, device='cuda'):
    # Paths
    print("Initializing dataset for Attention U-Net...")
    input_paths = [
        os.path.join(data_dir, "lInputData_left.pt"),
        os.path.join(data_dir, "lInputData_right.pt")
    ]
    gt_paths = [
        os.path.join(data_dir, "lGroundTruthData_left.pt"),
        os.path.join(data_dir, "lGroundTruthData_right.pt")
    ]
    
    # Dataset
    dataset = CrackDataset(input_paths, gt_paths)
    
    # Split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # Optimizations: num_workers, pin_memory
    # persistent_workers=True helps if you have enough RAM
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Model: Attention U-Net
    print("Building Attention U-Net...")
    model = AttentionUNet(n_channels=2, n_classes=3).to(device)
    
    # Loss: Optimized Hybrid (Focal Tversky + Asymmetric Focal)
    print("Using Optimized Hybrid Loss (Focal Tversky + Asymmetric Focal)...")
    criterion = OptimizedHybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2, betas=(0.9, 0.999))
    
    # Optimization: Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"Starting training on {device} for {num_epochs} epochs...")
    print(f"Batch Size: {batch_size}, Workers: 4, AMP: Enabled")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Optimization: Mixed Precision Training
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_ds)
        train_losses.append(epoch_loss)
        
        # QA / Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                # AMP for validation too (saves memory/time)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = val_running_loss / len(val_ds)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
    # Project Root
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    
    # Save Best Model
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "attention_unet_best.pth"))
        print(f"New best validation loss: {best_val_loss:.4f}. Saved model.")
            
    # Save final model
    save_path = os.path.join(checkpoint_dir, "attention_unet_last.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
    
    # Plot loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Training Curves (Attention U-Net + Hybrid Loss)")
    plt.savefig(os.path.join(output_dir, "attention_training_loss.png"))
    print("Saved attention_training_loss.png")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(PROJECT_ROOT, "data", "S_160_4.7", "interim")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_attention_model(data_dir, device=device)
