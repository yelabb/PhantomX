"""
MC_Maze Dataset Loader

Loads neural recordings directly from NWB files for VQ-VAE training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict

# Try to import PyNWB for real data loading
try:
    from pynwb import NWBHDF5IO
    HAS_PYNWB = True
except ImportError:
    print("Warning: PyNWB not installed. Using mock data.")
    HAS_PYNWB = False


def load_mc_maze_from_nwb(file_path: str, bin_size_ms: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spike counts and kinematics from MC_Maze NWB file.
    
    Args:
        file_path: Path to the NWB file
        bin_size_ms: Time bin size in milliseconds (default 25ms = 40Hz)
        
    Returns:
        spike_counts: [T, n_channels] binned spike counts
        kinematics: [T, 4] cursor position and velocity (x, y, vx, vy)
    """
    if not HAS_PYNWB:
        raise ImportError("PyNWB is required for loading NWB files. Install with: pip install pynwb")
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NWB file not found: {file_path}")
    
    print(f"Loading NWB file: {file_path}")
    
    with NWBHDF5IO(str(path), mode='r', load_namespaces=True) as io:
        nwb = io.read()
        
        # Get neural units
        units = nwb.units
        n_channels = len(units.id[:])
        print(f"  Found {n_channels} neural units")
        
        # Get behavior timestamps and data
        behavior = nwb.processing.get('behavior')
        if behavior is None:
            raise RuntimeError("No 'behavior' processing module found")
        
        # Get cursor position
        cursor_pos_container = behavior.data_interfaces.get('cursor_pos')
        if cursor_pos_container is None:
            raise RuntimeError("No cursor_pos found in behavior")
        
        cursor_pos = cursor_pos_container.data[:]  # [T, 2] (x, y)
        timestamps = cursor_pos_container.timestamps[:]  # [T]
        
        # Get hand velocity
        hand_vel_container = behavior.data_interfaces.get('hand_vel')
        if hand_vel_container is not None:
            hand_vel = hand_vel_container.data[:]  # [T, 2] (vx, vy)
        else:
            # Compute velocity from position
            dt = np.diff(timestamps)
            dt = np.concatenate([[dt[0]], dt])
            hand_vel = np.gradient(cursor_pos, timestamps, axis=0)
        
        # Combine kinematics: [x, y, vx, vy]
        kinematics = np.hstack([cursor_pos, hand_vel]).astype(np.float32)
        
        duration = timestamps[-1] - timestamps[0]
        n_bins = int(duration * 1000 / bin_size_ms)
        print(f"  Duration: {duration:.1f}s, {n_bins} bins at {1000/bin_size_ms:.0f}Hz")
        
        # Bin spikes
        spike_counts = np.zeros((n_bins, n_channels), dtype=np.float32)
        
        for unit_idx in range(n_channels):
            spike_times = units.get_unit_spike_times(unit_idx)
            if spike_times is not None and len(spike_times) > 0:
                # Convert spike times to bin indices
                bin_indices = ((spike_times - timestamps[0]) * 1000 / bin_size_ms).astype(np.int32)
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                
                # Count spikes per bin
                for bin_idx in bin_indices:
                    spike_counts[bin_idx, unit_idx] += 1
        
        # Resample kinematics to match spike bins
        kin_timestamps = np.linspace(timestamps[0], timestamps[-1], n_bins)
        kinematics_resampled = np.zeros((n_bins, 4), dtype=np.float32)
        for i in range(4):
            kinematics_resampled[:, i] = np.interp(kin_timestamps, timestamps, kinematics[:, i])
        
        print(f"  Loaded: spike_counts {spike_counts.shape}, kinematics {kinematics_resampled.shape}")
        
        return spike_counts, kinematics_resampled


class MCMazeDataset(Dataset):
    """
    PyTorch Dataset for MC_Maze neural recordings.
    
    Loads spike counts and kinematics from NWB file
    and prepares them for VQ-VAE training.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        trial_indices: Optional[list] = None,
        sequence_length: int = 1,
        transform=None,
        use_mock: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to MC_Maze NWB file
            tokenizer: SpikeTokenizer instance
            trial_indices: Specific trials to use (None = all trials)
            sequence_length: Length of spike sequences (1 = single timestep)
            transform: Optional data transforms
            use_mock: Force use of mock data (for testing)
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Load data from NWB file or use mock
        if use_mock or not HAS_PYNWB or not Path(data_path).exists():
            print("Using mock MC_Maze data")
            T = 11760  # 294s * 40Hz
            self.spike_counts = np.random.poisson(2.0, size=(T, 142)).astype(np.float32)
            self.kinematics = np.random.randn(T, 4).astype(np.float32)
        else:
            self.spike_counts, self.kinematics = load_mc_maze_from_nwb(data_path)
        
        # Normalize spike counts to zero mean, unit variance per channel
        self.spike_mean = self.spike_counts.mean(axis=0, keepdims=True)
        self.spike_std = self.spike_counts.std(axis=0, keepdims=True) + 1e-6
        self.spike_counts = (self.spike_counts - self.spike_mean) / self.spike_std
        
        # Normalize kinematics to zero mean, unit variance per dimension
        self.kin_mean = self.kinematics.mean(axis=0, keepdims=True)
        self.kin_std = self.kinematics.std(axis=0, keepdims=True) + 1e-6
        self.kinematics = (self.kinematics - self.kin_mean) / self.kin_std
        
        print(f"  Normalized spike counts: mean={self.spike_counts.mean():.4f}, std={self.spike_counts.std():.4f}")
        
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
