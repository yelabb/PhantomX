"""
Data Augmentation Transforms

Neural data augmentation techniques for robust BCI decoding.
Addresses the "Exp 22 forgot augmentation" issue by centralizing transforms.
"""

from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
from omegaconf import DictConfig


@dataclass
class AugmentedSample:
    """Container for augmented neural data."""
    neural: torch.Tensor
    target: torch.Tensor
    mask: Optional[torch.Tensor] = None


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        self.transforms = [t for t in transforms if t is not None]
    
    def __call__(self, neural: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            neural, target = t(neural, target)
        return neural, target


class ElectrodeDropout:
    """
    Randomly drop electrodes during training.
    
    Key for robustness to electrode failure in real BCIs.
    From Exp 22b: Critical augmentation for generalization.
    
    Args:
        p: Probability of dropping each electrode
        structured: If True, drop contiguous electrode groups
    """
    
    def __init__(self, p: float = 0.1, structured: bool = False):
        self.p = p
        self.structured = structured
    
    def __call__(self, neural: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return neural, target
        
        if self.structured:
            return self._structured_dropout(neural, target)
        else:
            return self._random_dropout(neural, target)
    
    def _random_dropout(self, neural: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # neural: [batch, time, channels] or [batch, channels]
        if neural.dim() == 3:
            n_channels = neural.shape[-1]
            mask = torch.rand(n_channels, device=neural.device) > self.p
            neural = neural * mask.unsqueeze(0).unsqueeze(0)
        else:
            n_channels = neural.shape[-1]
            mask = torch.rand(n_channels, device=neural.device) > self.p
            neural = neural * mask.unsqueeze(0)
        return neural, target
    
    def _structured_dropout(self, neural: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Drop contiguous groups of electrodes
        n_channels = neural.shape[-1]
        if torch.rand(1).item() < self.p:
            # Drop a contiguous block
            block_size = int(n_channels * self.p * 2)
            start = torch.randint(0, n_channels - block_size, (1,)).item()
            mask = torch.ones(n_channels, device=neural.device)
            mask[start:start + block_size] = 0
            if neural.dim() == 3:
                neural = neural * mask.unsqueeze(0).unsqueeze(0)
            else:
                neural = neural * mask.unsqueeze(0)
        return neural, target
    
    @property
    def training(self) -> bool:
        return True  # Override in wrapper


class TemporalJitter:
    """
    Randomly shift neural data in time.
    
    Helps with temporal alignment robustness.
    
    Args:
        max_shift_bins: Maximum bins to shift (positive or negative)
        wrap: If True, wrap around; else zero-pad
    """
    
    def __init__(self, max_shift_bins: int = 2, wrap: bool = False):
        self.max_shift = max_shift_bins
        self.wrap = wrap
    
    def __call__(self, neural: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_shift == 0:
            return neural, target
        
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        if shift == 0:
            return neural, target
        
        if neural.dim() == 3:
            # [batch, time, channels]
            neural = torch.roll(neural, shifts=shift, dims=1)
            target = torch.roll(target, shifts=shift, dims=0 if target.dim() == 1 else 1)
            
            if not self.wrap:
                # Zero out wrapped values
                if shift > 0:
                    neural[:, :shift, :] = 0
                else:
                    neural[:, shift:, :] = 0
        
        return neural, target


class GaussianNoise:
    """
    Add Gaussian noise to neural data.
    
    Args:
        scale: Noise standard deviation (relative to data std)
    """
    
    def __init__(self, scale: float = 0.05):
        self.scale = scale
    
    def __call__(self, neural: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(neural) * self.scale
        neural = neural + noise
        return neural, target


class SpikeScaling:
    """
    Randomly scale spike counts.
    
    Helps with gain variability across sessions.
    
    Args:
        min_scale: Minimum scale factor
        max_scale: Maximum scale factor
    """
    
    def __init__(self, min_scale: float = 0.9, max_scale: float = 1.1):
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def __call__(self, neural: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()
        neural = neural * scale
        return neural, target


class Mixup:
    """
    Mixup augmentation for neural data.
    
    From Zhang et al. 2018 - interpolate between samples.
    
    Args:
        alpha: Beta distribution parameter
    """
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, neural: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # This needs batch-level mixing - implemented in dataloader
        return neural, target


def build_augmentation(cfg: DictConfig) -> Optional[Compose]:
    """
    Build augmentation pipeline from config.
    
    Args:
        cfg: Augmentation configuration
        
    Returns:
        Composed transforms or None if disabled
    """
    if not cfg.get('enabled', False):
        return None
    
    transforms = []
    
    # Electrode dropout
    if cfg.get('electrode_dropout', {}).get('enabled', False):
        transforms.append(ElectrodeDropout(
            p=cfg.electrode_dropout.get('p', 0.1),
            structured=cfg.electrode_dropout.get('structured', False)
        ))
    
    # Temporal jitter
    if cfg.get('temporal_jitter', {}).get('enabled', False):
        transforms.append(TemporalJitter(
            max_shift_bins=cfg.temporal_jitter.get('max_shift_bins', 2),
            wrap=cfg.temporal_jitter.get('wrap', False)
        ))
    
    # Noise
    if cfg.get('noise', {}).get('enabled', False):
        if cfg.noise.get('type', 'gaussian') == 'gaussian':
            transforms.append(GaussianNoise(
                scale=cfg.noise.get('scale', 0.05)
            ))
    
    # Spike scaling
    if cfg.get('spike_scaling', {}).get('enabled', False):
        transforms.append(SpikeScaling(
            min_scale=cfg.spike_scaling.get('min_scale', 0.9),
            max_scale=cfg.spike_scaling.get('max_scale', 1.1)
        ))
    
    if not transforms:
        return None
    
    return Compose(transforms)
