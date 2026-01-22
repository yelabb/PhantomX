"""
Base DataModule

Abstract base class for all neural data modules.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from omegaconf import DictConfig


class WindowedNeuralDataset(Dataset):
    """
    Windowed neural dataset for velocity decoding.
    
    Extracts overlapping windows of neural data and corresponding targets.
    """
    
    def __init__(
        self,
        neural_data: np.ndarray,
        targets: np.ndarray,
        window_size: int = 10,
        stride: int = 1,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        """
        Args:
            neural_data: [T, n_channels] spike counts
            targets: [T, output_dim] velocity/position
            window_size: Number of time bins per window
            stride: Stride between windows
            normalize: Whether to z-score normalize
            mean, std: Pre-computed normalization stats
        """
        self.window_size = window_size
        self.stride = stride
        
        # Normalize if requested
        if normalize:
            if mean is None:
                mean = neural_data.mean(axis=0)
            if std is None:
                std = neural_data.std(axis=0) + 1e-8
            neural_data = (neural_data - mean) / std
        
        self.mean = mean
        self.std = std
        
        # Convert to tensors
        self.neural = torch.from_numpy(neural_data).float()
        self.targets = torch.from_numpy(targets).float()
        
        # Compute valid indices
        n_samples = len(neural_data) - window_size + 1
        self.indices = list(range(0, n_samples, stride))
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.window_size
        
        # Neural window
        neural_window = self.neural[start:end]  # [window_size, n_channels]
        
        # Target at end of window (or middle, depending on task)
        target = self.targets[end - 1]  # [output_dim]
        
        return neural_window, target


class BaseDataModule(ABC):
    """
    Abstract base class for data modules.
    
    Provides:
    - Data loading and preprocessing
    - Train/val/test splits
    - DataLoader creation
    - Augmentation integration
    """
    
    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        data_dir: str = "data",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        window_size: int = 10,
        stride: int = 1,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        shuffle: bool = True,
        normalize: bool = True,
        **kwargs
    ):
        # Use config if provided
        if cfg is not None:
            data_dir = cfg.get('data_dir', data_dir)
            batch_size = cfg.get('batch_size', batch_size)
            window_size = cfg.get('window_size', window_size) if 'window_size' in cfg else kwargs.get('window_size', window_size)
            train_ratio = cfg.get('train_ratio', train_ratio)
            val_ratio = cfg.get('val_ratio', val_ratio)
            test_ratio = cfg.get('test_ratio', test_ratio)
            shuffle = cfg.get('shuffle', shuffle)
            normalize = cfg.get('normalize', normalize)
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.window_size = window_size
        self.stride = stride
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.normalize = normalize
        
        # Data storage
        self.neural_data: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        
        # Normalization stats
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        
        # Dynamically determined from data
        self._n_channels: Optional[int] = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name."""
        pass
    
    @property
    def n_channels(self) -> int:
        """Number of neural channels (determined from loaded data)."""
        if self._n_channels is not None:
            return self._n_channels
        if self.neural_data is not None:
            return self.neural_data.shape[1]
        # Return a placeholder - will be updated after loading
        return self._get_expected_n_channels()
    
    def _get_expected_n_channels(self) -> int:
        """Override to provide expected channel count before data is loaded."""
        return 137  # Default
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Output dimension (e.g., 2 for velocity)."""
        pass
    
    @abstractmethod
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw neural data and targets."""
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        Setup data splits.
        
        Args:
            stage: 'fit', 'test', or None (all)
        """
        if self.neural_data is None:
            self.neural_data, self.targets = self._load_raw_data()
            # Update n_channels from loaded data
            self._n_channels = self.neural_data.shape[1]
        
        # Compute normalization stats from all data
        if self.normalize:
            self.mean = self.neural_data.mean(axis=0)
            self.std = self.neural_data.std(axis=0) + 1e-8
        
        # Create datasets with shared normalization
        full_dataset = WindowedNeuralDataset(
            self.neural_data,
            self.targets,
            window_size=self.window_size,
            stride=self.stride,
            normalize=self.normalize,
            mean=self.mean,
            std=self.std,
        )
        
        # Split based on WINDOWED dataset length (not raw data length)
        n_samples = len(full_dataset)
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)
        n_test = n_samples - n_train - n_val
        
        # Split
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"📊 {self.name}: train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}")
    
    def _should_pin_memory(self) -> bool:
        """Only pin memory if CUDA is available."""
        import torch
        return self.pin_memory and torch.cuda.is_available()
    
    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self._should_pin_memory(),
            drop_last=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self._should_pin_memory(),
        )
    
    def test_dataloader(self) -> DataLoader:
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self._should_pin_memory(),
        )
