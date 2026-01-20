"""
Finite Scalar Quantization (FSQ)

Physics-aligned "Neural Tokenizer" that respects the continuous topology of the brain.

Key Innovation: Replace VQ lookup table with a rectilinear grid.
Code_A is topologically related to Code_B by their distance in the grid.

Reference: "Finite Scalar Quantization: VQ-VAE Made Simple" (arXiv:2309.15505)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict


class FiniteScalarQuantization(nn.Module):
    """
    Finite Scalar Quantization Module.
    
    Mechanism:
    1. Project: z ∈ R^D_model → ẑ ∈ R^d (d = len(levels))
    2. Bound: Apply tanh to force range (-1, 1)
    3. Scale: Map (-1, 1) to (-(L-1)/2, (L-1)/2)
    4. Quantize: z_q = round(ẑ_scaled)
    5. Straight-Through: Forward uses z_q, backward passes gradients to ẑ
    
    Example: levels=[6,6,6,6] creates 6^4 = 1296 effective codes in R^4
    """
    
    def __init__(
        self,
        levels: List[int] = [6, 6, 6, 6],
        input_dim: int = 128,
        eps: float = 1e-3,
    ):
        """
        Initialize FSQ.
        
        Args:
            levels: Number of quantization levels per dimension.
                    E.g., [6, 6, 6, 6] creates 6^4 = 1296 implicit codes.
            input_dim: Input dimension from encoder (D_model)
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        self.levels = levels
        self.n_dims = len(levels)
        self.input_dim = input_dim
        self.eps = eps
        
        # Compute implicit codebook size (mixed-radix base)
        self.codebook_size = 1
        for L in levels:
            self.codebook_size *= L
        
        # Register levels as buffer for device movement
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.float32))
        
        # Compute half-levels for scaling: (L-1)/2
        self.register_buffer('_half_levels', (self._levels - 1) / 2)
        
        # Projection layer: D_model → d (FSQ dimensions)
        self.projection = nn.Linear(input_dim, self.n_dims)
        
        # Output projection: d → D_model (for decoder compatibility)
        self.output_projection = nn.Linear(self.n_dims, input_dim)
        
    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound values to (-1, 1) using tanh."""
        return torch.tanh(z)
    
    def _scale(self, z_bounded: torch.Tensor) -> torch.Tensor:
        """Scale from (-1, 1) to (-(L-1)/2, (L-1)/2) per dimension."""
        return z_bounded * self._half_levels
    
    def _quantize(self, z_scaled: torch.Tensor) -> torch.Tensor:
        """
        Round to nearest integer with straight-through gradient.
        
        Forward: z_q = round(z_scaled)
        Backward: gradients pass through to z_scaled (as if no rounding)
        """
        z_quantized = torch.round(z_scaled)
        # Straight-through estimator
        return z_scaled + (z_quantized - z_scaled).detach()
    
    def _compute_indices(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """
        Convert quantized values to single integer indices.
        
        Uses mixed-radix encoding based on levels.
        E.g., for levels=[6,6,6,6]: idx = z[0] + z[1]*6 + z[2]*36 + z[3]*216
        """
        # Shift from [-(L-1)/2, (L-1)/2] to [0, L-1]
        z_shifted = (z_quantized + self._half_levels).long()
        
        # Mixed-radix encoding
        indices = torch.zeros(z_quantized.shape[0], dtype=torch.long, device=z_quantized.device)
        stride = 1
        for d in range(self.n_dims):
            indices = indices + z_shifted[:, d] * stride
            stride *= self.levels[d]
        
        return indices
    
    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Quantize encoder output using FSQ.
        
        Args:
            z_e: [batch, input_dim] continuous latent vectors from encoder
            
        Returns:
            z_q: [batch, input_dim] quantized vectors (projected back to D_model)
            info: Dictionary containing:
                - z_hat: [batch, n_dims] continuous pre-quantized (for dynamics loss)
                - z_fsq: [batch, n_dims] quantized FSQ codes
                - indices: [batch] integer code indices
                - perplexity: codebook utilization metric
                - codebook_usage: fraction of codebook used
        """
        batch_size = z_e.shape[0]
        
        # 1. Project to FSQ dimensions
        z_proj = self.projection(z_e)  # [batch, n_dims]
        
        # 2. Bound to (-1, 1)
        z_bounded = self._bound(z_proj)  # [batch, n_dims]
        
        # 3. Scale to (-(L-1)/2, (L-1)/2)
        z_scaled = self._scale(z_bounded)  # [batch, n_dims]
        
        # Store continuous pre-quantized values (for dynamics loss)
        z_hat = z_scaled  # This is what LatentPredictor targets
        
        # 4. Quantize with straight-through
        z_quantized = self._quantize(z_scaled)  # [batch, n_dims]
        
        # 5. Compute integer indices
        indices = self._compute_indices(z_quantized)  # [batch]
        
        # 6. Project back to D_model for decoder
        z_q = self.output_projection(z_quantized)  # [batch, input_dim]
        
        # Compute statistics
        unique_codes = torch.unique(indices)
        codebook_usage = len(unique_codes) / self.codebook_size
        
        # Perplexity (entropy-based utilization)
        if batch_size > 0:
            counts = torch.bincount(indices, minlength=self.codebook_size).float()
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            perplexity = torch.exp(entropy)
        else:
            perplexity = torch.tensor(1.0, device=z_e.device)
        
        # Quantization error (useful for TTA)
        quant_error = torch.mean((z_scaled - z_quantized.detach()) ** 2)
        
        info = {
            'z_hat': z_hat,  # Continuous pre-quantized (for LatentPredictor)
            'z_fsq': z_quantized,  # Quantized FSQ codes
            'indices': indices,
            'perplexity': perplexity,
            'codebook_usage': codebook_usage,
            'quant_error': quant_error,
            'commitment_loss': torch.tensor(0.0, device=z_e.device),  # FSQ has no commitment loss
        }
        
        return z_q, info
    
    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert integer indices back to FSQ codes.
        
        Args:
            indices: [batch] integer indices
            
        Returns:
            z_fsq: [batch, n_dims] FSQ codes
        """
        z_int = torch.zeros(indices.shape[0], self.n_dims, dtype=torch.long, device=indices.device)
        
        remaining = indices.clone()
        for d in range(self.n_dims):
            z_int[:, d] = remaining % self.levels[d]
            remaining = remaining // self.levels[d]
        
        # Shift from [0, L-1] to [-(L-1)/2, (L-1)/2]
        z_fsq = z_int.float() - self._half_levels
        
        return z_fsq
    
    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embedding vectors by indices.
        
        Args:
            indices: [batch] discrete code indices
            
        Returns:
            z_q: [batch, input_dim] embedding vectors
        """
        z_fsq = self.indices_to_codes(indices)
        return self.output_projection(z_fsq)
    
    @property
    def embedding_dim(self) -> int:
        return self.input_dim


class ResidualFSQ(nn.Module):
    """
    Residual Finite Scalar Quantization.
    
    Stacks multiple FSQ layers to increase precision, similar to RVQ
    but with the topological advantages of FSQ.
    
    Each layer quantizes the residual from the previous layer.
    """
    
    def __init__(
        self,
        levels: List[int] = [6, 6, 6, 6],
        input_dim: int = 128,
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
            FiniteScalarQuantization(levels=levels, input_dim=input_dim)
            for _ in range(n_residual)
        ])
        
        # Total codebook size is product of layer codebook sizes
        self.codebook_size = self.fsq_layers[0].codebook_size ** n_residual
        self.embedding_dim = input_dim
    
    def forward(
        self,
        z_e: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        all_z_hat = []
        
        for layer in self.fsq_layers:
            z_q, info = layer(residual)
            z_q_total = z_q_total + z_q
            residual = z_e - z_q_total  # Compute next residual
            all_indices.append(info['indices'])
            all_perplexities.append(info['perplexity'])
            all_z_hat.append(info['z_hat'])
        
        # Average perplexity across layers
        avg_perplexity = torch.stack(all_perplexities).mean()
        
        combined_info = {
            'commitment_loss': torch.tensor(0.0, device=z_e.device),
            'perplexity': avg_perplexity,
            'indices': torch.stack(all_indices, dim=1),  # [batch, n_residual]
            'z_hat': all_z_hat[0],  # Use first layer's z_hat for dynamics
            'codebook_usage': info['codebook_usage'],  # From last layer
            'quant_error': info['quant_error'],
        }
        
        return z_q_total, combined_info


# Alias for backward compatibility
FSQ = FiniteScalarQuantization