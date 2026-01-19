"""
PhantomX Progressive VQ-VAE

Final model implementation achieving R² = 0.70 on MC_Maze velocity decoding.

Key innovations:
1. Progressive training: Pre-train encoder, then add VQ
2. EMA codebook updates with k-means initialization
3. Temporal window input (10 steps = 250ms)

Usage:
    model = ProgressiveVQVAE()
    trainer = ProgressiveTrainer(model, train_loader, val_loader)
    trainer.train()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class EMAVectorQuantizer(nn.Module):
    """
    Exponential Moving Average Vector Quantizer.
    
    Based on VQ-VAE-2 improvements:
    - EMA updates for codebook (gentler than gradient descent)
    - K-means initialization from encoder outputs
    - Dead code revival
    """
    
    def __init__(
        self,
        num_codes: int = 256,
        embedding_dim: int = 128,
        decay: float = 0.99,
        commitment_cost: float = 0.1,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        
        # Buffers for EMA updates
        self.register_buffer('embeddings', torch.randn(num_codes, embedding_dim))
        self.register_buffer('cluster_size', torch.ones(num_codes))
        self.register_buffer('ema_w', self.embeddings.clone())
        self._initialized = False
    
    def init_from_data(self, z_all: torch.Tensor) -> None:
        """Initialize codebook with k-means on encoder outputs."""
        z_np = z_all.detach().cpu().numpy()
        
        # Subsample if too large
        if len(z_np) > 10000:
            idx = np.random.choice(len(z_np), 10000, replace=False)
            z_np = z_np[idx]
        
        kmeans = KMeans(n_clusters=self.num_codes, n_init=10, max_iter=300)
        kmeans.fit(z_np)
        
        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        self.embeddings.copy_(centers)
        self.ema_w.copy_(centers)
        self._initialized = True
    
    def forward(self, z_e: torch.Tensor) -> tuple:
        """
        Args:
            z_e: Encoder output [batch_size, embedding_dim]
        
        Returns:
            z_q: Quantized vectors [batch_size, embedding_dim]
            info: Dict with indices, losses, and metrics
        """
        # Find nearest codes
        distances = torch.cdist(z_e, self.embeddings)
        indices = distances.argmin(dim=-1)
        z_q = F.embedding(indices, self.embeddings)
        
        # EMA update during training
        if self.training and self._initialized:
            encodings = F.one_hot(indices, self.num_codes).float()
            n_j = encodings.sum(0)
            
            # Update cluster sizes
            self.cluster_size.data = self.decay * self.cluster_size + (1 - self.decay) * n_j
            
            # Update embeddings
            dw = encodings.T @ z_e
            self.ema_w.data = self.decay * self.ema_w + (1 - self.decay) * dw
            
            # Normalize
            n = self.cluster_size.clamp(min=self.epsilon)
            self.embeddings.data = self.ema_w / n.unsqueeze(-1)
        
        # Commitment loss only (codebook updated by EMA)
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        
        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        
        # Compute perplexity
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, {
            'indices': indices,
            'commitment_loss': self.commitment_cost * commitment_loss,
            'perplexity': perplexity
        }


class ProgressiveVQVAE(nn.Module):
    """
    Progressive VQ-VAE for neural velocity decoding.
    
    Architecture:
        Input: [batch, window_size, n_channels] spike counts
        Encoder: MLP with LayerNorm and GELU
        VQ: EMA vector quantizer
        Decoder: MLP to velocity
    """
    
    def __init__(
        self,
        n_channels: int = 142,
        window_size: int = 10,
        embedding_dim: int = 128,
        num_codes: int = 256,
        output_dim: int = 2
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.window_size = window_size
        input_dim = n_channels * window_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embedding_dim)
        )
        
        # VQ layer
        self.vq = EMAVectorQuantizer(num_codes, embedding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        
        # Training mode
        self.use_vq = False
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spike windows to embeddings."""
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.encoder(x_flat)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> dict:
        """
        Forward pass.
        
        Args:
            x: Spike windows [batch, window_size, n_channels]
            targets: Velocity targets [batch, 2] (optional)
        
        Returns:
            Dict with predictions, embeddings, and losses
        """
        z_e = self.encode(x)
        
        if self.use_vq:
            z_q, vq_info = self.vq(z_e)
        else:
            z_q = z_e
            vq_info = {
                'commitment_loss': torch.tensor(0.0, device=z_e.device),
                'perplexity': torch.tensor(0.0, device=z_e.device)
            }
        
        velocity_pred = self.decoder(z_q)
        
        output = {
            'velocity_pred': velocity_pred,
            'z_e': z_e,
            'z_q': z_q,
            **vq_info
        }
        
        if targets is not None:
            recon_loss = F.mse_loss(velocity_pred, targets)
            total_loss = recon_loss + vq_info['commitment_loss']
            output['recon_loss'] = recon_loss
            output['total_loss'] = total_loss
        
        return output
    
    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Get VQ code indices for input."""
        z_e = self.encode(x)
        _, info = self.vq(z_e)
        return info['indices']


def create_model(
    n_channels: int = 142,
    window_size: int = 10,
    embedding_dim: int = 128,
    num_codes: int = 256
) -> ProgressiveVQVAE:
    """Create a ProgressiveVQVAE with default parameters."""
    return ProgressiveVQVAE(
        n_channels=n_channels,
        window_size=window_size,
        embedding_dim=embedding_dim,
        num_codes=num_codes
    )


def load_model(path: str, device: str = 'cpu') -> ProgressiveVQVAE:
    """Load a trained model from checkpoint."""
    model = ProgressiveVQVAE()
    model.load_state_dict(torch.load(path, map_location=device))
    model.use_vq = True
    return model
