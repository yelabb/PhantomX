"""
PhantomX: Neural Foundation Model for BCI Decoding

A configuration-driven research framework for neural decoding experiments.
"""

__version__ = "0.2.0"

from .models import build_model
from .datamodules import build_datamodule
from .trainer import Trainer

__all__ = ["build_model", "build_datamodule", "Trainer", "__version__"]
