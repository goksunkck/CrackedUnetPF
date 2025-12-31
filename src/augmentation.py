"""
Data augmentation transforms as specified in the original paper.

Augmentation steps (training only):
1. Random crop: size 120-180 pixels, left edge 10-30 pixels
2. Random rotation: -10 to +10 degrees, crop largest square
3. Random flip up/down: 50% probability
4. Upsample to 224x224 (linear for input, nearest for target)

No augmentation during validation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


class PaperAugmentation:
    """
    Applies the exact augmentation pipeline from the paper.
    Works on (input, target) pairs where:
    - input: (C, H, W) tensor (displacement fields)
    - target: (H, W) tensor (segmentation mask)
    """
    
    def __init__(self, output_size=224):
        self.output_size = output_size
    
    def __call__(self, input_tensor, target_tensor):
        """
        Args:
            input_tensor: (C, H, W) float tensor
            target_tensor: (H, W) long tensor
        Returns:
            augmented_input: (C, output_size, output_size)
            augmented_target: (output_size, output_size)
        """
        # Convert to numpy for easier manipulation
        input_np = input_tensor.numpy()  # (C, H, W)
        target_np = target_tensor.numpy()  # (H, W)
        
        C, H, W = input_np.shape
        
        # Step 1: Random Crop
        crop_size = np.random.randint(120, 181)  # 120 to 180
        left_edge = np.random.randint(10, 31)    # 10 to 30
        
        # Calculate crop boundaries
        top = max(0, (H - crop_size) // 2)  # Center vertically
        left = left_edge
        bottom = min(H, top + crop_size)
        right = min(W, left + crop_size)
        
        # Adjust if we go out of bounds
        if right > W:
            right = W
            left = max(0, W - crop_size)
        if bottom > H:
            bottom = H
            top = max(0, H - crop_size)
        
        input_cropped = input_np[:, top:bottom, left:right]
        target_cropped = target_np[top:bottom, left:right]
        
        # Step 2: Random Rotation (-10 to +10 degrees)
        angle = np.random.uniform(-10, 10)
        input_rotated, target_rotated = self._rotate_and_crop(
            input_cropped, target_cropped, angle
        )
        
        # Step 3: Random Vertical Flip (50% probability)
        if np.random.random() < 0.5:
            input_rotated = np.flip(input_rotated, axis=1).copy()  # Flip vertically (axis 1 for CHW)
            target_rotated = np.flip(target_rotated, axis=0).copy()  # Flip vertically (axis 0 for HW)
        
        # Step 4: Upsample to output_size x output_size
        input_tensor = torch.from_numpy(input_rotated).float()
        target_tensor = torch.from_numpy(target_rotated).long()
        
        # Upsample input with bilinear interpolation
        input_tensor = input_tensor.unsqueeze(0)  # (1, C, H, W)
        input_tensor = F.interpolate(
            input_tensor, 
            size=(self.output_size, self.output_size), 
            mode='bilinear', 
            align_corners=False
        )
        input_tensor = input_tensor.squeeze(0)  # (C, H, W)
        
        # Upsample target with nearest neighbor
        target_tensor = target_tensor.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        target_tensor = F.interpolate(
            target_tensor, 
            size=(self.output_size, self.output_size), 
            mode='nearest'
        )
        target_tensor = target_tensor.squeeze(0).squeeze(0).long()  # (H, W)
        
        return input_tensor, target_tensor
    
    def _rotate_and_crop(self, input_np, target_np, angle_degrees):
        """
        Rotate image and crop the largest inscribed square.
        """
        from scipy.ndimage import rotate as scipy_rotate
        
        C, H, W = input_np.shape
        
        # Rotate each channel of input
        rotated_input = np.zeros_like(input_np)
        for c in range(C):
            rotated_input[c] = scipy_rotate(
                input_np[c], angle_degrees, reshape=False, order=1, mode='constant', cval=0
            )
        
        # Rotate target with nearest neighbor (order=0)
        rotated_target = scipy_rotate(
            target_np, angle_degrees, reshape=False, order=0, mode='constant', cval=0
        )
        
        # Calculate the largest inscribed square after rotation
        # For small angles, this is approximately the original size minus some margin
        angle_rad = abs(angle_degrees) * math.pi / 180
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Size of largest inscribed square
        min_dim = min(H, W)
        new_size = int(min_dim / (cos_a + sin_a))
        new_size = max(new_size, 1)  # Ensure at least 1 pixel
        
        # Crop from center
        center_h, center_w = H // 2, W // 2
        half_size = new_size // 2
        
        top = max(0, center_h - half_size)
        bottom = min(H, top + new_size)
        left = max(0, center_w - half_size)
        right = min(W, left + new_size)
        
        cropped_input = rotated_input[:, top:bottom, left:right]
        cropped_target = rotated_target[top:bottom, left:right]
        
        return cropped_input, cropped_target


class ValidationTransform:
    """
    Transform for validation: NO augmentation, NO resizing.
    Input data stays at original size (256x256) as per paper.
    """
    
    def __init__(self):
        pass
    
    def __call__(self, input_tensor, target_tensor):
        """
        No transformation - return as-is.
        """
        return input_tensor, target_tensor
