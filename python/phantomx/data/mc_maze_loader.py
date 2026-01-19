"""
MC_Maze Dataset Loader

Integrates with PhantomLink's data loader for MC_Maze neural recordings.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict
import sys

# Add PhantomLink to path for imports
sys.path.insert(0, str(Path(__file__).parents[5] / "PhantomLink" / "src"))

try:
    from loader import MCMazeLoader
except ImportError:
    print("Warning: Could not import MCMazeLoader from PhantomLink. Using mock loader.")
    MCMazeLoader = None


class MCMazeDataset(Dataset):
    """
    PyTorch Dataset for MC_Maze neural recordings.
    
    Loads spike counts and kinematics from PhantomLink's data loader
    and prepares them for VQ-VAE training.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        trial_indices: Optional[list] = None,
        sequence_length: int = 1,
        transform=None
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to MC_Maze NWB file
            tokenizer: SpikeTokenizer instance
            trial_indices: Specific trials to use (None = all trials)
            sequence_length: Length of spike sequences (1 = single timestep)
            transform: Optional data transforms
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load data using PhantomLink's loader
        if MCMazeLoader is not None:
            self.loader = MCMazeLoader(data_path)
            self.spike_counts = self.loader.spike_counts  # [T, 142]
            self.kinematics = self.loader.get_kinematics()  # [T, 4] (x, y, vx, vy)
        else:
            # Mock data for testing without PhantomLink
            print("Using mock MC_Maze data")
            T = 11760  # 294s * 40Hz
            self.spike_counts = np.random.poisson(2.0, size=(T, 142)).astype(np.float32)
            self.kinematics = np.random.randn(T, 4).astype(np.float32)
        
        # Extract velocity components (vx, vy)
        self.velocities = self.kinematics[:, 2:4]  # [T, 2]
        
        # Filter by trial indices if specified
        if trial_indices is not None:
            # TODO: Implement trial-based filtering
            pass
        
        # Fit tokenizer if needed
        if not self.tokenizer.is_fitted:
            print("Fitting tokenizer on training data...")
            self.tokenizer.fit(self.spike_counts)
    
    def __len__(self) -> int:
        """Number of samples"""
        return len(self.spike_counts) - self.sequence_length + 1
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - tokens: [n_tokens] discrete token IDs
                - kinematics: [2] velocity (vx, vy)
                - spike_counts: [n_channels] raw spike counts (optional)
        """
        # Get spike counts for this timestep
        spikes = self.spike_counts[idx]  # [142]
        
        # Tokenize
        tokens = self.tokenizer.tokenize(spikes)  # [n_tokens]
        
        # Get corresponding kinematics
        velocity = self.velocities[idx]  # [2]
        
        # Convert to torch tensors
        sample = {
            'tokens': torch.from_numpy(tokens).long(),
            'kinematics': torch.from_numpy(velocity).float(),
            'spike_counts': torch.from_numpy(spikes).float()
        }
        
        # Apply transforms if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class MCMazeDataLoader:
    """
    Convenience wrapper for creating train/val/test dataloaders.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        batch_size: int = 32,
        train_split: float = 0.7,
        val_split: float = 0.15,
        num_workers: int = 0,
        pin_memory: bool = True
    ):
        """
        Initialize data loaders.
        
        Args:
            data_path: Path to MC_Maze NWB file
            tokenizer: SpikeTokenizer instance
            batch_size: Batch size for training
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            num_workers: Number of worker processes
            pin_memory: Pin memory for faster GPU transfer
        """
        # Create full dataset
        full_dataset = MCMazeDataset(data_path, tokenizer)
        
        # Split into train/val/test
        total_len = len(full_dataset)
        train_len = int(total_len * train_split)
        val_len = int(total_len * val_split)
        test_len = total_len - train_len - val_len
        
        self.train_dataset, self.val_dataset, self.test_dataset = \
            torch.utils.data.random_split(
                full_dataset,
                [train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(42)
            )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, val, test loaders"""
        return self.train_loader, self.val_loader, self.test_loader
