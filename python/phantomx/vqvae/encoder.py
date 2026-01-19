"""
Spike Encoder Network

Encodes tokenized spikes into continuous latent space.
"""

import torch
import torch.nn as nn
from typing import Optional


class SpikeEncoder(nn.Module):
    """
    Encoder network: Tokens → Latent codes
    
    Architecture: MLP with residual connections
    """
    
    def __init__(
        self,
        n_tokens: int = 16,
        token_dim: int = 256,
        hidden_dims: list = [256, 128, 64],
        embedding_dim: int = 64,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize encoder.
        
        Args:
            n_tokens: Number of input tokens
            token_dim: Vocabulary size (for embedding)
            hidden_dims: List of hidden layer dimensions
            embedding_dim: Output latent dimension
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.embedding_dim = embedding_dim
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(token_dim, hidden_dims[0] // n_tokens)
        
        # Build encoder layers
        layers = []
        in_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Final projection to latent space
        layers.append(nn.Linear(in_dim, embedding_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode tokens to latent codes.
        
        Args:
            tokens: [batch, n_tokens] discrete token IDs
            
        Returns:
            z_e: [batch, embedding_dim] continuous latent vectors
        """
        # Embed tokens
        token_embeds = self.token_embedding(tokens)  # [batch, n_tokens, embed_dim]
        
        # Flatten token embeddings
        token_embeds_flat = token_embeds.view(tokens.shape[0], -1)  # [batch, n_tokens * embed_dim]
        
        # Pass through encoder
        z_e = self.encoder(token_embeds_flat)  # [batch, embedding_dim]
        
        return z_e


class TransformerEncoder(nn.Module):
    """
    Alternative: Transformer-based encoder for better permutation invariance.
    
    This is more advanced and can handle variable numbers of tokens.
    """
    
    def __init__(
        self,
        token_dim: int = 256,
        embedding_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize transformer encoder.
        
        Args:
            token_dim: Vocabulary size
            embedding_dim: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.token_embedding = nn.Embedding(token_dim, embedding_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, embedding_dim))  # Max 512 tokens
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode tokens using transformer.
        
        Args:
            tokens: [batch, n_tokens] discrete token IDs
            
        Returns:
            z_e: [batch, embedding_dim] continuous latent vectors
        """
        batch_size, seq_len = tokens.shape
        
        # Embed tokens
        x = self.token_embedding(tokens)  # [batch, n_tokens, embedding_dim]
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Pass through transformer
        x = self.transformer(x)  # [batch, n_tokens, embedding_dim]
        
        # Pool over sequence dimension
        x = x.transpose(1, 2)  # [batch, embedding_dim, n_tokens]
        z_e = self.pool(x).squeeze(-1)  # [batch, embedding_dim]
        
        return z_e
