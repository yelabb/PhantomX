"""LaBraM VQ-VAE Module

Includes both traditional VQ-VAE and the improved FSQ-VAE architecture.

FSQ-VAE (Finite Scalar Quantization):
- Topology-preserving discrete representations
- Dual-head decoder (kinematics + spike reconstruction)
- No codebook collapse issues
"""

from .vqvae import VQVAE
from .codebook import Codebook, VectorQuantizer
from .encoder import SpikeEncoder
from .decoder import KinematicsDecoder
from .trainer import VQVAETrainer

# FSQ-VAE components (The Pivot)
from .fsq import FSQ, FiniteScalarQuantization, ResidualFSQ
from .spike_decoder import SpikeReconstructionDecoder, TokenReconstructionDecoder
from .fsq_vae import FSQVAE, FSQVAEWithCausalTransformer
from .fsq_trainer import FSQVAETrainer, compute_baseline_comparison

__all__ = [
    # Original VQ-VAE
    "VQVAE",
    "Codebook",
    "VectorQuantizer",
    "SpikeEncoder",
    "KinematicsDecoder",
    "VQVAETrainer",
    # FSQ-VAE (The Pivot)
    "FSQ",
    "FiniteScalarQuantization",
    "ResidualFSQ",
    "SpikeReconstructionDecoder",
    "TokenReconstructionDecoder",
    "FSQVAE",
    "FSQVAEWithCausalTransformer",
    "FSQVAETrainer",
    "compute_baseline_comparison",
]
