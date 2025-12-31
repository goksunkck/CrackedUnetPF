import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dataset import CrackDataset
from src.model import UNet
import matplotlib.pyplot as plt
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W)
        # targets: (B, H, W)
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets
        # targets is (B, H, W) -> (B, H, W, C) -> (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Flatten for calculation
        probs_flat = probs.contiguous().view(probs.shape[0], probs.shape[1], -1)
        targets_flat = targets_one_hot.contiguous().view(targets_one_hot.shape[0], targets_one_hot.shape[1], -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=2)
        union = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes (including background) and batch
        return 1 - dice.mean()

def train_model(data_dir, num_epochs=100, batch_size=8, learning_rate=1e-3, device='cuda'):
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
    
    # Dataset
    dataset = CrackDataset(input_paths, gt_paths)
    
    # Split
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Model
    model = UNet(n_channels=2, n_classes=3).to(device)
    
    # Loss & Optimizer
    # Using DiceLoss as requested for unbalanced classes
    criterion = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    print(f"Starting training on {device}...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_ds)
        train_losses.append(epoch_loss)
        
        # QA / Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = val_running_loss / len(val_ds)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        
        # Save Best Model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "unet_crack_segmentation_best.pth")
            print(f"New best validation loss: {best_val_loss:.4f}. Saved model.")

    # Save final model
    save_path = "unet_crack_segmentation_last.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")
    
    # Plot loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title("Training Curves")
    plt.savefig("training_loss.png")
    print("Saved training_loss.png")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(PROJECT_ROOT, "data", "S_160_4.7", "interim")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(data_dir, device=device)
