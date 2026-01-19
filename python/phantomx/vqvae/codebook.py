"""
Vector Quantizer and Codebook

Implements the discrete latent space for LaBraM.

Reference: "Large Brain Model: Universal Neural Dynamics via Vector Quantization" (ICLR 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Codebook(nn.Module):
    """
    Learnable codebook for vector quantization.
    
    The codebook stores K embedding vectors that represent
    discrete latent neural states.
    """
    
    def __init__(
        self,
        num_codes: int = 256,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        use_ema: bool = True
    ):
        """
        Initialize codebook.
        
        Args:
            num_codes: Number of discrete codes (K)
            embedding_dim: Dimension of each code vector (D)
            commitment_cost: Weight for commitment loss (β)
            decay: EMA decay rate for codebook updates
            epsilon: Small constant for numerical stability
            use_ema: If True, use exponential moving average for codebook updates
        """
        super().__init__()
        
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        self.use_ema = use_ema
        
        # Initialize codebook with uniform distribution
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        
        # EMA statistics for codebook updates
        if use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(num_codes))
            self.register_buffer('ema_embedding', self.embedding.weight.data.clone())
    
    def forward(
        self,
        z_e: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous latent vectors to discrete codes.
        
        Args:
            z_e: [batch, embedding_dim] continuous latent vectors from encoder
            temperature: Softmax temperature for soft quantization (1.0 = hard)
            
        Returns:
            z_q: [batch, embedding_dim] quantized vectors
            indices: [batch] discrete code indices
            commitment_loss: Scalar commitment loss
        """
        # Flatten input
        batch_size = z_e.shape[0]
        z_e_flat = z_e.view(-1, self.embedding_dim)  # [B, D]
        
        # Compute distances to all codebook vectors
        # d(x, c) = ||x||² + ||c||² - 2⟨x, c⟩
        z_e_norm = torch.sum(z_e_flat ** 2, dim=1, keepdim=True)  # [B, 1]
        codebook_norm = torch.sum(self.embedding.weight ** 2, dim=1)  # [K]
        distances = (
            z_e_norm
            + codebook_norm.unsqueeze(0)
            - 2 * torch.matmul(z_e_flat, self.embedding.weight.t())
        )  # [B, K]
        
        # Find nearest codebook vectors
        indices = torch.argmin(distances, dim=1)  # [B]
        
        # Look up quantized vectors
        z_q = self.embedding(indices)  # [B, D]
        
        # Compute losses
        # 1. Codebook loss: Move codebook vectors toward encoder outputs
        codebook_loss = F.mse_loss(z_q.detach(), z_e_flat)
        
        # 2. Commitment loss: Encourage encoder to commit to codebook
        commitment_loss = F.mse_loss(z_q, z_e_flat.detach()) * self.commitment_cost
        
        # Straight-through estimator: Copy gradients from z_q to z_e
        z_q = z_e_flat + (z_q - z_e_flat).detach()
        
        # Update codebook with EMA (if enabled and training)
        if self.training and self.use_ema:
            self._update_ema(z_e_flat, indices)
        
        # Reshape back
        z_q = z_q.view(batch_size, self.embedding_dim)
        
        return z_q, indices, commitment_loss
    
    def _update_ema(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        """
        Update codebook using exponential moving average.
        
        This is more stable than gradient-based updates.
        """
        with torch.no_grad():
            # One-hot encoding of indices
            encodings = F.one_hot(indices, self.num_codes).float()  # [B, K]
            
            # Update cluster sizes
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                (1 - self.decay) * torch.sum(encodings, dim=0)
            
            # Laplace smoothing
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_codes * self.epsilon)
                * n
            )
            
            # Update embeddings
            dw = torch.matmul(encodings.t(), z_e)  # [K, D]
            self.ema_embedding = self.ema_embedding * self.decay + (1 - self.decay) * dw
            
            # Normalize embeddings
            self.embedding.weight.data = self.ema_embedding / self.ema_cluster_size.unsqueeze(1)
    
    def get_codebook_usage(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute codebook usage statistics.
        
        Args:
            indices: [batch] discrete code indices from a batch
            
        Returns:
            usage: [num_codes] usage count for each code
        """
        usage = torch.zeros(self.num_codes, device=indices.device)
        unique, counts = torch.unique(indices, return_counts=True)
        usage[unique] = counts.float()
        return usage
    
    def reset_unused_codes(self, usage_threshold: float = 0.01) -> int:
        """
        Reset rarely used codes to random encoder outputs.
        
        Args:
            usage_threshold: Minimum usage fraction to keep a code
            
        Returns:
            Number of codes reset
        """
        # This should be called periodically during training
        # to prevent codebook collapse
        pass  # TODO: Implement if needed


class VectorQuantizer(nn.Module):
    """
    Full vector quantization module combining codebook and quantization logic.
    """
    
    def __init__(
        self,
        num_codes: int = 256,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        **kwargs
    ):
        super().__init__()
        self.codebook = Codebook(
            num_codes=num_codes,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            **kwargs
        )
    
    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Quantize encoder output.
        
        Args:
            z_e: [batch, embedding_dim] continuous latents
            
        Returns:
            z_q: [batch, embedding_dim] quantized latents
            info: Dictionary with quantization statistics
        """
        z_q, indices, commitment_loss = self.codebook(z_e)
        
        # Compute perplexity (measure of codebook utilization)
        # High perplexity → good utilization
        # Low perplexity → codebook collapse
        encodings = F.one_hot(indices, self.codebook.num_codes).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        info = {
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
            'indices': indices,
            'codebook_usage': self.codebook.get_codebook_usage(indices)
        }
        
        return z_q, info
    
    def quantize(self, z_e: torch.Tensor) -> torch.Tensor:
        """Quantize without returning statistics (for inference)"""
        z_q, _, _ = self.codebook(z_e)
        return z_q
    
    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up codebook vectors by indices.
        
        Args:
            indices: [batch] discrete code indices
            
        Returns:
            embeddings: [batch, embedding_dim] codebook vectors
        """
        return self.codebook.embedding(indices)
