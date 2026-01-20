"""
Finite Scalar Quantization (FSQ)

Replaces Vector Quantization with topology-preserving scalar quantization.

Key Insight: FSQ creates an implicit codebook where neighboring codes are
semantically similar, addressing the "Categorical vs. Ordinal" mismatch
that limits standard VQ on continuous kinematic trajectories.

Reference: "Finite Scalar Quantization: VQ-VAE Made Simple" (arXiv:2309.15505)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class FSQ(nn.Module):
    """
    Finite Scalar Quantization Module.
    
    Instead of learning a discrete codebook, FSQ:
    1. Projects encoder output to D dimensions
    2. Bounds each dimension via tanh to [-1, 1]
    3. Quantizes each dimension to L discrete levels
    
    This creates an implicit codebook of size L₁ × L₂ × ... × Lₐ
    where codes have ordinal meaning: [1,0,0] is close to [2,0,0].
    
    The ordinal structure preserves the topology of the latent space,
    which is critical for smooth kinematic decoding.
    """
    
    def __init__(
        self,
        levels: List[int] = [8, 5, 5, 5],
        input_dim: int = 64,
        eps: float = 1e-3,
    ):
        """
        Initialize FSQ.
        
        Args:
            levels: Number of quantization levels per dimension.
                    E.g., [8, 5, 5, 5] creates 8*5*5*5 = 1000 implicit codes.
                    Use odd numbers for symmetric quantization around 0.
            input_dim: Input dimension from encoder (will be projected to len(levels))
            eps: Small constant for numerical stability in gradient computation
        """
        super().__init__()
        
        self.levels = levels
        self.n_dims = len(levels)
        self.input_dim = input_dim
        self.eps = eps
        
        # Compute implicit codebook size
        self.codebook_size = 1
        for L in levels:
            self.codebook_size *= L
        
        # Register levels as buffer for device movement
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.float32))
        
        # Projection layer: map encoder output to FSQ dimensions
        self.projection = nn.Linear(input_dim, self.n_dims)
        
        # Output projection: map FSQ dims back to embedding space
        # This allows the decoder to work with a richer representation
        self.embedding_dim = input_dim
        self.output_projection = nn.Linear(self.n_dims, self.embedding_dim)
        
    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """
        Bound values to approximately [-1, 1] using tanh.
        
        Uses a scaled tanh to ensure values can reach the extremes.
        """
        # Scale to make bounds reachable
        return torch.tanh(z)
    
    def _quantize(self, z_bounded: torch.Tensor) -> torch.Tensor:
        """
        Quantize bounded values to discrete levels.
        
        For each dimension d with L levels, quantizes to {0, 1, ..., L-1}
        then rescales back to approximately [-1, 1].
        
        Uses straight-through estimator for gradient flow.
        """
        # Get half-levels for each dimension: floor(L/2)
        half_levels = (self._levels - 1) / 2  # [n_dims]
        
        # Scale from [-1, 1] to [-half_levels, half_levels]
        z_scaled = z_bounded * half_levels  # [batch, n_dims]
        
        # Round to nearest integer (straight-through estimator)
        z_quantized = torch.round(z_scaled)
        
        # Straight-through: gradients flow through as if no rounding occurred
        z_quantized = z_scaled + (z_quantized - z_scaled).detach()
        
        # Scale back to [-1, 1] range
        z_normalized = z_quantized / half_levels
        
        return z_normalized
    
    def forward(
        self,
        z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Quantize encoder output using FSQ.
        
        Args:
            z_e: [batch, input_dim] continuous latent vectors from encoder
            
        Returns:
            z_q: [batch, embedding_dim] quantized vectors (projected back)
            info: Dictionary with quantization statistics
        """
        # Project to FSQ dimensions
        z_proj = self.projection(z_e)  # [batch, n_dims]
        
        # Bound to [-1, 1]
        z_bounded = self._bound(z_proj)  # [batch, n_dims]
        
        # Quantize each dimension
        z_quantized = self._quantize(z_bounded)  # [batch, n_dims]
        
        # Get discrete indices for analysis/codebook statistics
        indices = self._get_indices(z_quantized)  # [batch]
        
        # Project back to embedding dimension
        z_q = self.output_projection(z_quantized)  # [batch, embedding_dim]
        
        # Compute usage statistics
        unique_codes = torch.unique(indices)
        codebook_usage = len(unique_codes) / self.codebook_size
        
        # Perplexity calculation (entropy-based codebook utilization)
        # High perplexity = uniform usage, low perplexity = collapse
        if indices.numel() > 0:
            counts = torch.bincount(indices, minlength=self.codebook_size).float()
            probs = counts / counts.sum()
            probs = probs[probs > 0]  # Filter zeros for log
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            perplexity = torch.exp(entropy)
        else:
            perplexity = torch.tensor(1.0, device=z_e.device)
        
        info = {
            'commitment_loss': torch.tensor(0.0, device=z_e.device),  # FSQ has no commitment loss!
            'perplexity': perplexity,
            'indices': indices,
            'codebook_usage': codebook_usage,
            'z_fsq': z_quantized,  # The raw FSQ representation
            'z_bounded': z_bounded,  # Pre-quantization (for analysis)
        }
        
        return z_q, info
    
    def _get_indices(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """
        Convert quantized values to single integer indices.
        
        Maps the multi-dimensional quantized vector to a single index
        in the implicit codebook.
        """
        # Get half-levels
        half_levels = (self._levels - 1) / 2  # [n_dims]
        
        # Convert from [-1, 1] to [0, L-1] for each dimension
        z_int = ((z_quantized * half_levels) + half_levels).long()  # [batch, n_dims]
        
        # Compute single index: idx = z[0] + z[1]*L[0] + z[2]*L[0]*L[1] + ...
        indices = torch.zeros(z_quantized.shape[0], dtype=torch.long, device=z_quantized.device)
        stride = 1
        for d in range(self.n_dims):
            indices = indices + z_int[:, d] * stride
            stride *= self.levels[d]
        
        return indices
    
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert single indices back to FSQ codes.
        
        Args:
            indices: [batch] integer indices
            
        Returns:
            z_quantized: [batch, n_dims] FSQ codes in [-1, 1]
        """
        z_int = torch.zeros(indices.shape[0], self.n_dims, dtype=torch.long, device=indices.device)
        
        remaining = indices.clone()
        for d in range(self.n_dims):
            z_int[:, d] = remaining % self.levels[d]
            remaining = remaining // self.levels[d]
        
        # Convert back to [-1, 1] range
        half_levels = (self._levels - 1) / 2
        z_quantized = (z_int.float() - half_levels) / half_levels
        
        return z_quantized
    
    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embedding vectors by indices.
        
        Args:
            indices: [batch] discrete code indices
            
        Returns:
            embeddings: [batch, embedding_dim] codebook vectors
        """
        z_quantized = self.indices_to_codes(indices)
        return self.output_projection(z_quantized)


class ResidualFSQ(nn.Module):
    """
    Residual Finite Scalar Quantization.
    
    Stacks multiple FSQ layers to increase precision, similar to RVQ
    but with the topological advantages of FSQ.
    
    Each layer quantizes the residual from the previous layer.
    """
    
    def __init__(
        self,
        levels: List[int] = [8, 5, 5, 5],
        input_dim: int = 64,
        n_residual: int = 3,
    ):
        """
        Initialize Residual FSQ.
        
        Args:
            levels: FSQ levels per layer
            input_dim: Input dimension
            n_residual: Number of residual FSQ layers
        """
        super().__init__()
        
        self.n_residual = n_residual
        self.input_dim = input_dim
        
        # Create FSQ layers
        self.fsq_layers = nn.ModuleList([
            FSQ(levels=levels, input_dim=input_dim)
            for _ in range(n_residual)
        ])
        
        # Total codebook size is product of layer codebook sizes
        self.codebook_size = self.fsq_layers[0].codebook_size ** n_residual
        self.embedding_dim = input_dim
    
    def forward(
        self,
        z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Quantize with residual FSQ.
        
        Args:
            z_e: [batch, input_dim] continuous latent vectors
            
        Returns:
            z_q: [batch, embedding_dim] quantized vectors
            info: Dictionary with quantization statistics
        """
        residual = z_e
        z_q_total = torch.zeros_like(z_e)
        all_indices = []
        all_perplexities = []
        
        for layer in self.fsq_layers:
            z_q, info = layer(residual)
            z_q_total = z_q_total + z_q
            residual = z_e - z_q_total  # Compute next residual
            all_indices.append(info['indices'])
            all_perplexities.append(info['perplexity'])
        
        # Average perplexity across layers
        avg_perplexity = torch.stack(all_perplexities).mean()
        
        combined_info = {
            'commitment_loss': torch.tensor(0.0, device=z_e.device),
            'perplexity': avg_perplexity,
            'indices': torch.stack(all_indices, dim=1),  # [batch, n_residual]
            'codebook_usage': info['codebook_usage'],  # From last layer
        }
        
        return z_q_total, combined_info
