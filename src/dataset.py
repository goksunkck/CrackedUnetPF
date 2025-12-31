import torch
from torch.utils.data import Dataset
import os

class CrackDataset(Dataset):
    def __init__(self, input_paths, gt_paths, transform=None):
        """
        Args:
            input_paths (list of str): Paths to .pt files containing input lists.
            gt_paths (list of str): Paths to .pt files containing GT lists.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.inputs = []
        self.targets = []
        self.transform = transform

        # Load inputs
        for p in input_paths:
            print(f"Loading inputs from {p}...")
            data_list = torch.load(p)
            self.inputs.extend([t.squeeze(0).float() for t in data_list])
            
        # Load targets if provided
        if gt_paths:
            for p in gt_paths:
                if os.path.exists(p):
                    print(f"Loading targets from {p}...")
                    data_list = torch.load(p)
                    self.targets.extend([t.squeeze(0).long() for t in data_list])
                else:
                    print(f"Warning: GT file {p} not found.")

        # Handle missing GT
        if not self.targets:
            print("No Ground Truth loaded. Targets will be None.")
            self.targets = [None] * len(self.inputs)
        else:
            assert len(self.inputs) == len(self.targets), "Input and Target lengths do not match!"
            
        print(f"Loaded {len(self.inputs)} samples.")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)
            
        return x, y
