"""
PhantomX DataModules

PyTorch Lightning-style data wrappers for neural datasets.
"""

from typing import Dict, Type
from omegaconf import DictConfig

import hydra

# Registry
DATAMODULE_REGISTRY: Dict[str, Type] = {}


def register_datamodule(name: str):
    """Decorator to register a datamodule."""
    def decorator(cls):
        DATAMODULE_REGISTRY[name] = cls
        return cls
    return decorator


def build_datamodule(cfg: DictConfig):
    """
    Build a datamodule from configuration.
    
    Args:
        cfg: Full configuration (must contain 'dataset' key)
        
    Returns:
        Instantiated datamodule
    """
    dataset_cfg = cfg.dataset
    
    # Try Hydra instantiate first
    if '_target_' in dataset_cfg:
        return hydra.utils.instantiate(dataset_cfg)
    
    # Fall back to registry
    name = dataset_cfg.get('name', 'mc_maze')
    _import_all_datamodules()
    
    if name not in DATAMODULE_REGISTRY:
        available = ", ".join(sorted(DATAMODULE_REGISTRY.keys()))
        raise KeyError(f"DataModule '{name}' not found. Available: {available}")
    
    return DATAMODULE_REGISTRY[name](cfg=dataset_cfg)


def list_datamodules() -> list:
    """List all registered datamodules."""
    _import_all_datamodules()
    return sorted(DATAMODULE_REGISTRY.keys())


def _import_all_datamodules():
    """Import all datamodule modules."""
    try:
        from . import mc_maze
    except ImportError:
        pass
    try:
        from . import mc_rtt
    except ImportError:
        pass


from .base import BaseDataModule
from .mc_maze import MCMazeDataModule
from .mc_rtt import MCRTTDataModule

__all__ = [
    "build_datamodule",
    "list_datamodules",
    "BaseDataModule",
    "MCMazeDataModule",
    "MCRTTDataModule",
]
