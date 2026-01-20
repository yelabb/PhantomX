"""
Spike Reconstruction Decoder (Foundation Head)

Reconstructs the original spike input from the quantized latent space.
This forces the encoder to preserve neural information beyond just
what's needed for velocity decoding (Information Bottleneck regularization).

Key Insight: By forcing reconstruction of spikes, we prevent the encoder
from discarding "non-velocity" neural information (preparation states,
null-space activity), creating a true neural state representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpikeReconstructionDecoder(nn.Module):
    """
    Decoder B: Reconstructs original spike tokens/rates from latent codes.
    
    Uses Poisson NLL loss which is appropriate for spike count data.
    
    Architecture: MLP with residual connections, outputting rate parameters
    for a Poisson distribution over spike counts.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dims: list = [128, 256, 256],
        n_channels: int = 96,  # Number of neural channels
        n_bins: int = 10,  # Number of time bins to reconstruct
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize spike reconstruction decoder.
        
        Args:
            embedding_dim: Input latent dimension
            hidden_dims: List of hidden layer dimensions
            n_channels: Number of neural recording channels
            n_bins: Number of time bins in the input window
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_channels = n_channels
        self.n_bins = n_bins
        self.output_dim = n_channels * n_bins
        
        # Build decoder layers
        layers = []
        in_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Final projection to spike rates
        # Output is log-rate (for numerical stability with Poisson NLL)
        layers.append(nn.Linear(in_dim, self.output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
        # Softplus for converting to positive rates if needed
        self.softplus = nn.Softplus()
    
    def forward(
        self,
        z_q: torch.Tensor,
        return_rates: bool = True
    ) -> torch.Tensor:
        """
        Decode latent codes to spike rates.
        
        Args:
            z_q: [batch, embedding_dim] quantized latent vectors
            return_rates: If True, return positive rates via softplus.
                         If False, return log-rates (for Poisson NLL loss).
            
        Returns:
            spike_pred: [batch, n_channels, n_bins] predicted spike rates/log-rates
        """
        # Decode
        log_rates = self.decoder(z_q)  # [batch, n_channels * n_bins]
        
        # Reshape to spatial-temporal structure
        log_rates = log_rates.view(-1, self.n_channels, self.n_bins)
        
        if return_rates:
            # Convert to positive rates
            rates = self.softplus(log_rates)
            return rates
        else:
            return log_rates
    
    def compute_loss(
        self,
        z_q: torch.Tensor,
        target_spikes: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute Poisson NLL loss for spike reconstruction.
        
        The Poisson distribution is the natural choice for modeling
        spike counts, and NLL directly optimizes the likelihood.
        
        Args:
            z_q: [batch, embedding_dim] quantized latent vectors
            target_spikes: [batch, n_channels, n_bins] target spike counts
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            loss: Poisson NLL loss
        """
        # Get log-rates for Poisson NLL
        log_rates = self.forward(z_q, return_rates=False)
        
        # Poisson NLL: log(λ) - y*log(λ) + y*log(y) - y + 0.5*log(2πy)
        # Simplified (constant terms dropped): exp(log_λ) - y*log_λ
        # PyTorch's poisson_nll_loss expects (log_input, target)
        loss = F.poisson_nll_loss(
            log_rates,
            target_spikes,
            log_input=True,
            full=False,  # Don't include Stirling approximation
            reduction=reduction
        )
        
        return loss


class TokenReconstructionDecoder(nn.Module):
    """
    Alternative: Reconstructs discrete token IDs instead of raw spikes.
    
    Uses cross-entropy loss, treating reconstruction as classification.
    This is useful when working with pre-tokenized spike data.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dims: list = [128, 256],
        n_tokens: int = 16,  # Number of tokens per sample
        vocab_size: int = 256,  # Token vocabulary size
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize token reconstruction decoder.
        
        Args:
            embedding_dim: Input latent dimension
            hidden_dims: List of hidden layer dimensions
            n_tokens: Number of tokens to reconstruct
            vocab_size: Size of token vocabulary
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_tokens = n_tokens
        self.vocab_size = vocab_size
        
        # Build decoder layers
        layers = []
        in_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Output heads for each token position
        self.token_heads = nn.ModuleList([
            nn.Linear(in_dim, vocab_size)
            for _ in range(n_tokens)
        ])
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to token logits.
        
        Args:
            z_q: [batch, embedding_dim] quantized latent vectors
            
        Returns:
            logits: [batch, n_tokens, vocab_size] token logits
        """
        # Decode
        hidden = self.decoder(z_q)  # [batch, hidden_dim]
        
        # Compute logits for each token position
        logits = torch.stack([
            head(hidden) for head in self.token_heads
        ], dim=1)  # [batch, n_tokens, vocab_size]
        
        return logits
    
    def compute_loss(
        self,
        z_q: torch.Tensor,
        target_tokens: torch.Tensor,
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for token reconstruction.
        
        Args:
            z_q: [batch, embedding_dim] quantized latent vectors
            target_tokens: [batch, n_tokens] target token IDs
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            loss: Cross-entropy loss
        """
        logits = self.forward(z_q)  # [batch, n_tokens, vocab_size]
        
        # Reshape for cross-entropy
        logits_flat = logits.view(-1, self.vocab_size)  # [batch * n_tokens, vocab_size]
        targets_flat = target_tokens.view(-1)  # [batch * n_tokens]
        
        loss = F.cross_entropy(logits_flat, targets_flat, reduction=reduction)
        
        return loss
    
    def reconstruct_tokens(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct token IDs (argmax of logits).
        
        Args:
            z_q: [batch, embedding_dim] quantized latent vectors
            
        Returns:
            tokens: [batch, n_tokens] reconstructed token IDs
        """
        logits = self.forward(z_q)
        tokens = torch.argmax(logits, dim=-1)
        return tokens
