"""
Kinematics Decoder Network

Decodes quantized latent codes to predicted kinematics (velocity).
"""

import torch
import torch.nn as nn


class KinematicsDecoder(nn.Module):
    """
    Decoder network: Latent codes → Kinematics (vx, vy)
    
    Architecture: MLP with residual connections
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dims: list = [128, 256, 256],
        output_dim: int = 2,  # (vx, vy)
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        """
        Initialize decoder.
        
        Args:
            embedding_dim: Input latent dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (2 for vx, vy)
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        
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
        
        # Final projection to kinematics
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to kinematics.
        
        Args:
            z_q: [batch, embedding_dim] quantized latent vectors
            
        Returns:
            kinematics: [batch, output_dim] predicted velocities (vx, vy)
        """
        kinematics = self.decoder(z_q)  # [batch, output_dim]
        return kinematics


class RecurrentDecoder(nn.Module):
    """
    Alternative: Recurrent decoder for temporal smoothing.
    
    Uses LSTM/GRU to decode sequences of latent codes.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = 'lstm'
    ):
        """
        Initialize recurrent decoder.
        
        Args:
            embedding_dim: Input latent dimension
            hidden_dim: RNN hidden dimension
            output_dim: Output dimension (2 for vx, vy)
            n_layers: Number of RNN layers
            dropout: Dropout rate
            rnn_type: 'lstm' or 'gru'
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        # RNN layer
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(
        self,
        z_q: torch.Tensor,
        hidden: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent codes with RNN.
        
        Args:
            z_q: [batch, seq_len, embedding_dim] sequence of quantized latents
            hidden: Optional hidden state from previous timestep
            
        Returns:
            kinematics: [batch, seq_len, output_dim] predicted velocities
            hidden: Updated hidden state
        """
        # Pass through RNN
        rnn_out, hidden = self.rnn(z_q, hidden)  # [batch, seq_len, hidden_dim]
        
        # Project to kinematics
        kinematics = self.output_proj(rnn_out)  # [batch, seq_len, output_dim]
        
        return kinematics, hidden
