"""
Generic Spike Dataset

For custom spike data formats.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict


class SpikeDataset(Dataset):
    """
    Generic dataset for spike count data.
    
    Use this for custom data formats or when not using MC_Maze.
    """
    
    def __init__(
        self,
        spike_counts: np.ndarray,
        targets: np.ndarray,
        tokenizer,
        transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            spike_counts: [n_samples, n_channels] spike count matrix
            targets: [n_samples, output_dim] target values (e.g., kinematics)
            tokenizer: SpikeTokenizer instance
            transform: Optional data transforms
        """
        super().__init__()
        
        assert len(spike_counts) == len(targets), \
            f"Spike counts ({len(spike_counts)}) and targets ({len(targets)}) must have same length"
        
        self.spike_counts = spike_counts
        self.targets = targets
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Fit tokenizer if needed
        if not self.tokenizer.is_fitted:
            self.tokenizer.fit(spike_counts)
    
    def __len__(self) -> int:
        return len(self.spike_counts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        spikes = self.spike_counts[idx]
        target = self.targets[idx]
        
        # Tokenize
        tokens = self.tokenizer.tokenize(spikes)
        
        sample = {
            'tokens': torch.from_numpy(tokens).long(),
            'kinematics': torch.from_numpy(target).float(),
            'spike_counts': torch.from_numpy(spikes).float()
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
