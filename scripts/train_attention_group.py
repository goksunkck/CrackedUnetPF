"""
Training script for AttentionUNet with GroupNorm.
Uses GroupNorm instead of BatchNorm for better domain shift handling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import CrackDataset
from src.model_attention_group import AttentionUNetGN  # GroupNorm model
import matplotlib.pyplot as plt

# --- Custom Loss Functions ---

class FocalTverskyLoss(nn.Module):
    def __init__(self, delta=0.8, gamma=0.75, smooth=1e-6, class_weights=None):
        super(FocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.smooth = smooth
        self.class_weights = class_weights if class_weights is not None else [1.0, 10.0, 20.0]

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        tp = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        fp = (probs * (1 - targets_one_hot)).sum(dim=(0, 2, 3))
        fn = ((1 - probs) * targets_one_hot).sum(dim=(0, 2, 3))
        
        tversky_index = (tp + self.smooth) / (tp + self.delta * fn + (1 - self.delta) * fp + self.smooth)
        focal_tversky_loss = (1 - tversky_index) ** self.gamma
        
        weights = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
        weighted_loss = focal_tversky_loss * weights
        
        return weighted_loss.sum() / weights.sum()


class WeightedFocalCELoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None):
        super(WeightedFocalCELoss, self).__init__()
        self.gamma = gamma
        self.class_weights = class_weights if class_weights is not None else [1.0, 10.0, 20.0]

    def forward(self, logits, targets):
        weight = torch.tensor(self.class_weights, device=logits.device, dtype=logits.dtype)
        ce_loss = F.cross_entropy(logits, targets, weight=weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class OptimizedHybridLoss(nn.Module):
    """Combines Focal Tversky Loss + Weighted Focal CE Loss."""
    def __init__(self):
        super(OptimizedHybridLoss, self).__init__()
        weights = [1.0, 10.0, 20.0]
        self.ftl = FocalTverskyLoss(delta=0.8, gamma=0.75, class_weights=weights)
        self.wfce = WeightedFocalCELoss(gamma=2.0, class_weights=weights)

    def forward(self, inputs, targets):
        loss_ftl = self.ftl(inputs, targets)
        loss_wfce = self.wfce(inputs, targets)
        return loss_ftl + loss_wfce


# --- Training Loop ---

def train_groupnorm_model(data_dir, num_epochs=100, batch_size=16, learning_rate=1e-3, device='cuda'):
    print("=" * 60)
    print("Training AttentionUNet with GroupNorm")
    print("=" * 60)
    
    # Paths
    print("Initializing dataset...")
    input_paths = [
        os.path.join(data_dir, "lInputData_left.pt"),
        os.path.join(data_dir, "lInputData_right.pt")
    ]
    gt_paths = [
        os.path.join(data_dir, "lGroundTruthData_left.pt"),
        os.path.join(data_dir, "lGroundTruthData_right.pt")
    ]
    
    # Simple augmentation: random vertical flip only
    print("Using simple augmentation (random vertical flip only)...")
    
    class SimpleFlipAugmentation:
        def __init__(self, p=0.5):
            self.p = p
        
        def __call__(self, x, y):
            if torch.rand(1).item() < self.p:
                x = torch.flip(x, dims=[1])
                y = torch.flip(y, dims=[0])
            return x, y
    
    train_transform = SimpleFlipAugmentation(p=0.5)
    
    # Load data
    full_dataset = CrackDataset(input_paths, gt_paths, transform=None)
    
    # Split indices
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    indices = list(range(len(full_dataset)))
    import random
    random.seed(42)
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Wrapper dataset with transforms
    class TransformSubset(torch.utils.data.Dataset):
        def __init__(self, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            x, y = self.dataset.inputs[self.indices[idx]], self.dataset.targets[self.indices[idx]]
            if self.transform and y is not None:
                x, y = self.transform(x, y)
            return x, y
    
    train_ds = TransformSubset(full_dataset, train_indices, train_transform)
    val_ds = torch.utils.data.Subset(full_dataset, val_indices)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Model: Attention U-Net with GroupNorm
    print("Building AttentionUNet with GroupNorm...")
    model = AttentionUNetGN(n_channels=2, n_classes=3).to(device)
    
    # Loss
    print("Using Optimized Hybrid Loss (Focal Tversky + Focal CE)...")
    criterion = OptimizedHybridLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    # Mixed Precision
    scaler = torch.amp.GradScaler('cuda')
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    print(f"Starting training on {device} for {num_epochs} epochs...")
    print(f"Batch Size: {batch_size}, Workers: 4, AMP: Enabled")
    
    # Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    checkpoint_dir = os.path.join(PROJECT_ROOT, 'checkpoints')
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_ds)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = val_running_loss / len(val_ds)
        val_losses.append(epoch_val_loss)
        
        scheduler.step(epoch_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, LR: {current_lr:.1e}")
        
        # Save Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "attention_unet_groupnorm_best.pth"))
            print(f"  -> New best validation loss: {best_val_loss:.4f}. Saved model.")
            
    # Save final model
    save_path = os.path.join(checkpoint_dir, "attention_unet_groupnorm_last.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
    
    # Plot loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Training Curves (AttentionUNet + GroupNorm)")
    plt.savefig(os.path.join(output_dir, "groupnorm_training_loss.png"))
    print("Saved groupnorm_training_loss.png")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(PROJECT_ROOT, "data", "S_160_4.7", "interim")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_groupnorm_model(data_dir, device=device)
