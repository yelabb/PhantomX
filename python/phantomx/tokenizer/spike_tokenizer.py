"""
POYO Spike Tokenizer

Converts spike counts into discrete tokens, making the representation
invariant to electrode dropout and permutation.

Reference: "Neural Population Dynamics as Discrete Token Sequences" (NeurIPS 2024)
"""

import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Configuration for spike tokenization"""
    n_channels: int = 142
    bin_size_ms: float = 25.0
    quantization_levels: int = 16
    use_population_norm: bool = True
    dropout_invariant: bool = True
    min_spike_threshold: float = 0.1


class SpikeTokenizer:
    """
    POYO-style neural tokenizer that converts spike counts to discrete tokens.
    
    Key features:
    - Electrode-dropout robust: Normalizes by active channel count
    - Permutation-invariant: Uses population statistics
    - Discrete representation: Quantizes to fixed vocabulary
    
    Example:
        tokenizer = SpikeTokenizer(n_channels=142, quantization_levels=16)
        spikes = np.random.poisson(2.0, size=142)  # [142] spike counts
        tokens = tokenizer.tokenize(spikes)         # [16] discrete tokens
    """
    
    def __init__(
        self,
        n_channels: int = 142,
        quantization_levels: int = 16,
        use_population_norm: bool = True,
        dropout_invariant: bool = True,
        min_spike_threshold: float = 0.1
    ):
        self.config = TokenizerConfig(
            n_channels=n_channels,
            quantization_levels=quantization_levels,
            use_population_norm=use_population_norm,
            dropout_invariant=dropout_invariant,
            min_spike_threshold=min_spike_threshold
        )
        
        # Statistics for normalization (learned from training data)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        
    def fit(self, spike_trains: np.ndarray) -> "SpikeTokenizer":
        """
        Fit tokenizer statistics on training data.
        
        Args:
            spike_trains: [n_samples, n_channels] spike count matrix
            
        Returns:
            self (for method chaining)
        """
        assert spike_trains.shape[1] == self.config.n_channels, \
            f"Expected {self.config.n_channels} channels, got {spike_trains.shape[1]}"
        
        # Compute per-channel statistics
        self.mean = np.mean(spike_trains, axis=0)
        self.std = np.std(spike_trains, axis=0) + 1e-6  # Avoid division by zero
        self.is_fitted = True
        
        return self
    
    def tokenize(
        self,
        spike_counts: np.ndarray,
        return_probs: bool = False
    ) -> np.ndarray:
        """
        Convert spike counts to discrete tokens.
        
        Args:
            spike_counts: [n_channels] or [batch, n_channels] spike counts
            return_probs: If True, return soft token probabilities instead of hard tokens
            
        Returns:
            tokens: [n_tokens] or [batch, n_tokens] discrete token IDs (0 to vocab_size-1)
        """
        # Handle batching
        is_batch = spike_counts.ndim == 2
        if not is_batch:
            spike_counts = spike_counts[np.newaxis, :]  # [1, n_channels]
        
        batch_size = spike_counts.shape[0]
        
        # Step 1: Detect active channels (handle dropout)
        if self.config.dropout_invariant:
            active_mask = spike_counts > self.config.min_spike_threshold
            n_active = np.sum(active_mask, axis=1, keepdims=True)  # [batch, 1]
            n_active = np.maximum(n_active, 1)  # Avoid division by zero
        else:
            active_mask = np.ones_like(spike_counts, dtype=bool)
            n_active = np.full((batch_size, 1), self.config.n_channels)
        
        # Step 2: Normalize (optional)
        if self.config.use_population_norm and self.is_fitted:
            spike_counts_norm = (spike_counts - self.mean) / self.std
        else:
            spike_counts_norm = spike_counts
        
        # Step 3: Population-level features (permutation-invariant)
        # Use order statistics to create permutation-invariant representation
        sorted_spikes = np.sort(spike_counts_norm, axis=1)[:, ::-1]  # Descending order
        
        # Take top-k channels (k = quantization_levels)
        k = min(self.config.quantization_levels, self.config.n_channels)
        top_k_spikes = sorted_spikes[:, :k]  # [batch, k]
        
        # Step 4: Quantize to discrete levels
        # Map continuous values to discrete bins
        tokens = self._quantize(top_k_spikes)  # [batch, k]
        
        if not is_batch:
            tokens = tokens[0]  # Remove batch dimension
        
        return tokens
    
    def _quantize(self, values: np.ndarray, n_bins: int = 256) -> np.ndarray:
        """
        Quantize continuous values to discrete bins.
        
        Args:
            values: [batch, k] continuous values
            n_bins: Number of discrete bins (vocabulary size)
            
        Returns:
            tokens: [batch, k] discrete token IDs in [0, n_bins-1]
        """
        # Clip to reasonable range (e.g., -3 to +3 standard deviations)
        values_clipped = np.clip(values, -3.0, 3.0)
        
        # Map [-3, 3] to [0, n_bins-1]
        tokens = ((values_clipped + 3.0) / 6.0 * (n_bins - 1)).astype(np.int32)
        tokens = np.clip(tokens, 0, n_bins - 1)
        
        return tokens
    
    def detokenize(self, tokens: np.ndarray) -> np.ndarray:
        """
        Convert tokens back to approximate spike counts (for visualization).
        
        Args:
            tokens: [n_tokens] or [batch, n_tokens] discrete token IDs
            
        Returns:
            spike_counts: [n_channels] or [batch, n_channels] approximate spike counts
        """
        # This is approximate and mainly for visualization
        # In practice, the VQ-VAE decoder will learn to map tokens → kinematics directly
        raise NotImplementedError("Detokenization is handled by VQ-VAE decoder")
    
    def to_embedding(self, tokens: np.ndarray, embedding_dim: int = 64) -> torch.Tensor:
        """
        Convert discrete tokens to continuous embeddings.
        
        Args:
            tokens: [n_tokens] or [batch, n_tokens] discrete token IDs
            embedding_dim: Embedding dimension
            
        Returns:
            embeddings: [embedding_dim] or [batch, embedding_dim] continuous embeddings
        """
        # Create learnable embedding layer
        if not hasattr(self, '_embedding'):
            vocab_size = 256  # Fixed vocabulary size
            self._embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        
        tokens_torch = torch.from_numpy(tokens).long()
        embeddings = self._embedding(tokens_torch)  # [batch, n_tokens, embedding_dim]
        
        # Pool over tokens (mean pooling)
        embeddings = torch.mean(embeddings, dim=-2)  # [batch, embedding_dim]
        
        return embeddings
    
    def save(self, path: str) -> None:
        """Save tokenizer state"""
        state = {
            'config': self.config.__dict__,
            'mean': self.mean,
            'std': self.std,
            'is_fitted': self.is_fitted
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path: str) -> "SpikeTokenizer":
        """Load tokenizer state"""
        state = torch.load(path)
        tokenizer = cls(**state['config'])
        tokenizer.mean = state['mean']
        tokenizer.std = state['std']
        tokenizer.is_fitted = state['is_fitted']
        return tokenizer


class BatchTokenizer:
    """Efficient batch tokenization for training"""
    
    def __init__(self, tokenizer: SpikeTokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, spike_batch: torch.Tensor) -> torch.Tensor:
        """
        Tokenize a batch of spike counts.
        
        Args:
            spike_batch: [batch, n_channels] spike counts (torch.Tensor)
            
        Returns:
            token_batch: [batch, n_tokens] discrete tokens
        """
        spike_batch_np = spike_batch.cpu().numpy()
        tokens_np = self.tokenizer.tokenize(spike_batch_np)
        return torch.from_numpy(tokens_np).to(spike_batch.device)
