"""
Seeding and Reproducibility Utilities

Ensures experiments are reproducible across runs.
Critical for statistical validation (Exp 23).
"""

import os
import random
import subprocess
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """
    Seed all random number generators for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower but reproducible)
        
    Returns:
        The seed used
        
    Example:
        >>> seed_everything(42)
        >>> # All subsequent random operations are reproducible
    """
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    if deterministic:
        # Deterministic algorithms (may be slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # PyTorch 1.8+
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass
    else:
        # Faster but non-deterministic
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    print(f"🌱 Seeded everything with seed={seed}, deterministic={deterministic}")
    return seed


def get_git_hash() -> Optional[str]:
    """
    Get the current git commit hash for reproducibility tracking.
    
    Returns:
        Short git hash or None if not in a git repo
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def get_git_diff() -> Optional[str]:
    """
    Get uncommitted changes for debugging.
    
    Returns:
        Git diff output or None
    """
    try:
        result = subprocess.run(
            ['git', 'diff', '--stat'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def is_deterministic() -> bool:
    """Check if PyTorch is in deterministic mode."""
    return (
        torch.backends.cudnn.deterministic and 
        not torch.backends.cudnn.benchmark
    )


class SeedContext:
    """
    Context manager for temporary seeding.
    
    Useful for reproducible data augmentation while keeping
    training stochastic.
    
    Example:
        >>> with SeedContext(42):
        ...     # Reproducible operations
        ...     data = augment(data)
        >>> # Back to previous random state
    """
    
    def __init__(self, seed: int):
        self.seed = seed
        self.numpy_state = None
        self.torch_state = None
        self.random_state = None
    
    def __enter__(self):
        # Save current states
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        self.random_state = random.getstate()
        
        # Set temporary seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        
        return self
    
    def __exit__(self, *args):
        # Restore previous states
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)
        random.setstate(self.random_state)
