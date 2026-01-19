"""
PhantomX: LaBraM-POYO Neural Foundation Model

Population-geometry BCI decoding with electrode-dropout robustness.
"""

__version__ = "0.1.0"
__author__ = "Youssef El abbassi"

from .tokenizer import SpikeTokenizer, TokenVocabulary
from .vqvae import VQVAE, VQVAETrainer, Codebook
from .tta import EntropyMinimizer, TTAOptimizer
from .inference import LabramDecoder

__all__ = [
    "SpikeTokenizer",
    "TokenVocabulary",
    "VQVAE",
    "VQVAETrainer",
    "Codebook",
    "EntropyMinimizer",
    "TTAOptimizer",
    "LabramDecoder",
]
