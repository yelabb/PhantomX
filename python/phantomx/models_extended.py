"""
PhantomX Extended Architectures

New model variants:
1. TransformerVQVAE - Self-attention for temporal modeling
2. GumbelVQVAE - Differentiable discrete bottleneck
3. HybridVQVAE - Combines both innovations
4. StatelessMambaDecoder - Mamba for temporal decoding (Exp 15)
5. LatentPredictor - Predict z_{t+1} from z_t (Exp 15)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from typing import Optional, Tuple, Dict


# ============================================================
# Stateless Mamba Decoder (Experiment 15)
# ============================================================

class StatelessMambaDecoder(nn.Module):
    """
    Stateless Mamba Decoder for velocity prediction.
    
    Treats each 250ms window (10 timesteps) as a complete sequence,
    resetting state every batch. This avoids the "shuffling suicide"
    issue from Experiment 12 while maintaining SSM benefits.
    
    Input: Quantized latents [B, T, d_fsq] projected up to D_model
    Output: Velocity predictions [B, T, 2]
    """
    
    def __init__(
        self,
        d_fsq: int = 4,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
        output_dim: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize Stateless Mamba Decoder.
        
        Args:
            d_fsq: FSQ latent dimension (e.g., 4 for [6,6,6,6])
            d_model: Internal model dimension
            d_state: SSM state dimension
            d_conv: Conv kernel size
            expand: FFN expansion factor
            n_layers: Number of Mamba blocks
            output_dim: Output dimension (2 for vx, vy)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_fsq = d_fsq
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        
        # Project FSQ latents to D_model
        self.input_proj = nn.Linear(d_fsq, d_model)
        
        # Mamba blocks (simplified S4D-style for portability)
        self.blocks = nn.ModuleList([
            SimplifiedMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)
    
    def forward(
        self,
        z_seq: torch.Tensor,
        hidden: Optional[torch.Tensor] = None  # Ignored for stateless
    ) -> torch.Tensor:
        """
        Forward pass through stateless Mamba.
        
        Args:
            z_seq: [batch, seq_len, d_fsq] FSQ latent sequence
            hidden: Ignored (stateless - reset every batch)
            
        Returns:
            velocity: [batch, seq_len, output_dim] velocity predictions
        """
        # Project to D_model
        x = self.input_proj(z_seq)  # [B, T, D_model]
        
        # Pass through Mamba blocks (stateless - no hidden state propagation)
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.output_norm(x)
        velocity = self.output_proj(x)  # [B, T, 2]
        
        return velocity


class SimplifiedMambaBlock(nn.Module):
    """
    Simplified Mamba-style block.
    
    Uses a selective scan approximation that's portable
    (doesn't require mamba_ssm CUDA kernels).
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        # Input projection (2x for gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # 1D conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # SSM parameters (simplified)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # Discretization
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # State matrix A (learnable, diagonal)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.register_buffer('A', -A.unsqueeze(0))  # [1, d_state]
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Norm and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        residual = x
        x = self.norm(x)
        batch, seq_len, _ = x.shape
        
        # Input projection with gate
        xz = self.in_proj(x)  # [B, T, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, T, d_inner]
        
        # 1D conv (causal)
        x_conv = x_proj.transpose(1, 2)  # [B, d_inner, T]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal: take first T
        x_conv = x_conv.transpose(1, 2)  # [B, T, d_inner]
        x_conv = F.silu(x_conv)
        
        # SSM parameters from input
        ssm_params = self.x_proj(x_conv)  # [B, T, 2*d_state + 1]
        B, C, dt_raw = torch.split(ssm_params, [self.d_state, self.d_state, 1], dim=-1)
        
        # Discretization step
        dt = F.softplus(self.dt_proj(dt_raw))  # [B, T, d_inner]
        
        # Simplified selective scan (RNN-style for portability)
        y = self._selective_scan_simple(x_conv, dt, B, C)
        
        # Gate and output
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y + residual
    
    def _selective_scan_simple(
        self,
        x: torch.Tensor,  # [B, T, d_inner]
        dt: torch.Tensor,  # [B, T, d_inner]
        B: torch.Tensor,  # [B, T, d_state]
        C: torch.Tensor,  # [B, T, d_state]
    ) -> torch.Tensor:
        """
        Simplified selective scan (sequential for portability).
        
        h_t = exp(dt * A) * h_{t-1} + dt * B * x
        y_t = C * h_t
        """
        batch, seq_len, d_inner = x.shape
        
        # Initialize state
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            # Discretize
            dt_t = dt[:, t, :].unsqueeze(-1)  # [B, d_inner, 1]
            A_bar = torch.exp(dt_t * self.A)  # [B, d_inner, d_state]
            B_bar = dt_t * B[:, t, :].unsqueeze(1)  # [B, d_inner, d_state]
            
            # State update
            x_t = x[:, t, :].unsqueeze(-1)  # [B, d_inner, 1]
            h = A_bar * h + B_bar * x_t  # [B, d_inner, d_state]
            
            # Output
            C_t = C[:, t, :]  # [B, d_state]
            y_t = torch.einsum('bds,bs->bd', h, C_t)  # [B, d_inner]
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)  # [B, T, d_inner]


# ============================================================
# Latent Dynamics Predictor (Experiment 15)
# ============================================================

class LatentPredictor(nn.Module):
    """
    Predict z_{t+1} from z_t.
    
    Forces the latent space to capture temporal dynamics.
    Predicts the CONTINUOUS pre-quantized z_hat (before rounding),
    which is easier to learn than predicting discrete jumps.
    
    Architecture: Simple MLP with residual connection
    """
    
    def __init__(
        self,
        d_fsq: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize Latent Predictor.
        
        Args:
            d_fsq: FSQ latent dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_fsq = d_fsq
        
        self.predictor = nn.Sequential(
            nn.Linear(d_fsq, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_fsq)
        )
        
        # Residual scaling
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        """
        Predict z_{t+1} from z_t.
        
        Args:
            z_t: [batch, d_fsq] current latent
            
        Returns:
            z_next_pred: [batch, d_fsq] predicted next latent
        """
        # Predict delta + residual connection
        delta = self.predictor(z_t)
        z_next_pred = z_t + self.residual_scale * delta
        
        return z_next_pred
    
    def compute_loss(
        self,
        z_t: torch.Tensor,
        z_next_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamics prediction loss.
        
        Args:
            z_t: [batch, d_fsq] current latent
            z_next_target: [batch, d_fsq] target next latent (z_hat, not quantized)
            
        Returns:
            loss: MSE loss for dynamics prediction
        """
        z_next_pred = self.forward(z_t)
        return F.mse_loss(z_next_pred, z_next_target)


# ============================================================
# Transformer Components
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for neural time series.
    
    Projects each timestep to d_model, applies self-attention,
    then aggregates to a single embedding.
    """
    
    def __init__(
        self,
        n_channels: int = 142,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_dim: int = 128
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_channels]
        Returns:
            [batch, output_dim]
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_proj(x)  # [B, T, d_model]
        x = self.pos_encoder(x)
        
        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, d_model]
        
        # Transformer
        x = self.transformer(x)
        
        # Use CLS token output
        cls_output = x[:, 0, :]  # [B, d_model]
        
        return self.output_proj(cls_output)


# ============================================================
# Gumbel-Softmax VQ
# ============================================================

class GumbelVectorQuantizer(nn.Module):
    """
    Differentiable vector quantization using Gumbel-Softmax.
    
    Instead of hard argmin, uses temperature-controlled soft assignment
    that anneals to discrete during training.
    """
    
    def __init__(
        self,
        num_codes: int = 256,
        embedding_dim: int = 128,
        temp_init: float = 2.0,
        temp_min: float = 0.1,
        temp_decay: float = 0.99995
    ):
        super().__init__()
        
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        
        self.embeddings = nn.Embedding(num_codes, embedding_dim)
        nn.init.uniform_(self.embeddings.weight, -1/num_codes, 1/num_codes)
        
        # Learnable logits projection
        self.to_logits = nn.Linear(embedding_dim, num_codes)
        
        self.register_buffer('temperature', torch.tensor(temp_init))
    
    def forward(self, z_e: torch.Tensor, hard: bool = None) -> tuple:
        """
        Args:
            z_e: Encoder output [batch, embedding_dim]
            hard: Force hard/soft. If None, hard during eval, soft during train
        """
        # Compute logits
        logits = self.to_logits(z_e)  # [B, K]
        
        if hard is None:
            hard = not self.training
        
        if hard:
            # Hard assignment (eval mode)
            indices = logits.argmax(dim=-1)
            z_q = self.embeddings(indices)
        else:
            # Gumbel-softmax (train mode)
            soft_onehot = F.gumbel_softmax(logits, tau=self.temperature.item(), hard=False)
            z_q = soft_onehot @ self.embeddings.weight  # [B, D]
            indices = logits.argmax(dim=-1)  # For logging
            
            # Anneal temperature
            if self.training:
                self.temperature.copy_(
                    torch.clamp(self.temperature * self.temp_decay, min=self.temp_min)
                )
        
        # Metrics
        probs = F.softmax(logits, dim=-1)
        avg_probs = probs.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Diversity loss: encourage uniform usage
        diversity_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=-1))
        
        return z_q, {
            'indices': indices,
            'perplexity': perplexity,
            'temperature': self.temperature,
            'diversity_loss': 0.1 * diversity_loss  # Small weight
        }


# ============================================================
# EMA VQ (from model.py, included for completeness)
# ============================================================

class EMAVectorQuantizer(nn.Module):
    """EMA Vector Quantizer with k-means initialization."""
    
    def __init__(self, num_codes=256, embedding_dim=128, decay=0.99, commitment_cost=0.1):
        super().__init__()
        
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.commitment_cost = commitment_cost
        
        self.register_buffer('embeddings', torch.randn(num_codes, embedding_dim))
        self.register_buffer('cluster_size', torch.ones(num_codes))
        self.register_buffer('ema_w', self.embeddings.clone())
        self._initialized = False
    
    def init_from_data(self, z_all: torch.Tensor):
        z_np = z_all.detach().cpu().numpy()
        if len(z_np) > 10000:
            idx = np.random.choice(len(z_np), 10000, replace=False)
            z_np = z_np[idx]
        
        kmeans = KMeans(n_clusters=self.num_codes, n_init=10, max_iter=300)
        kmeans.fit(z_np)
        
        self.embeddings.copy_(torch.from_numpy(kmeans.cluster_centers_).float())
        self.ema_w.copy_(self.embeddings)
        self._initialized = True
    
    def forward(self, z_e):
        distances = torch.cdist(z_e, self.embeddings)
        indices = distances.argmin(dim=-1)
        z_q = F.embedding(indices, self.embeddings)
        
        if self.training and self._initialized:
            encodings = F.one_hot(indices, self.num_codes).float()
            n_j = encodings.sum(0)
            self.cluster_size.data = self.decay * self.cluster_size + (1 - self.decay) * n_j
            dw = encodings.T @ z_e
            self.ema_w.data = self.decay * self.ema_w + (1 - self.decay) * dw
            n = self.cluster_size.clamp(min=1e-5)
            self.embeddings.data = self.ema_w / n.unsqueeze(-1)
        
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        z_q = z_e + (z_q - z_e).detach()
        
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, {
            'indices': indices,
            'commitment_loss': self.commitment_cost * commitment_loss,
            'perplexity': perplexity
        }


# ============================================================
# Model Variants
# ============================================================

class TransformerVQVAE(nn.Module):
    """VQ-VAE with Transformer encoder."""
    
    def __init__(
        self,
        n_channels: int = 142,
        window_size: int = 10,
        embedding_dim: int = 128,
        num_codes: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        vq_type: str = 'ema'  # 'ema' or 'gumbel'
    ):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            n_channels=n_channels,
            d_model=256,
            nhead=nhead,
            num_layers=num_layers,
            output_dim=embedding_dim
        )
        
        if vq_type == 'ema':
            self.vq = EMAVectorQuantizer(num_codes, embedding_dim)
        else:
            self.vq = GumbelVectorQuantizer(num_codes, embedding_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        
        self.use_vq = False
        self.vq_type = vq_type
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> dict:
        z_e = self.encode(x)
        
        if self.use_vq:
            z_q, vq_info = self.vq(z_e)
        else:
            z_q = z_e
            vq_info = {'commitment_loss': torch.tensor(0.0, device=z_e.device),
                       'perplexity': torch.tensor(0.0, device=z_e.device)}
        
        velocity_pred = self.decoder(z_q)
        
        output = {'velocity_pred': velocity_pred, 'z_e': z_e, 'z_q': z_q, **vq_info}
        
        if targets is not None:
            recon_loss = F.mse_loss(velocity_pred, targets)
            total_loss = recon_loss
            if 'commitment_loss' in vq_info:
                total_loss = total_loss + vq_info['commitment_loss']
            if 'diversity_loss' in vq_info:
                total_loss = total_loss + vq_info['diversity_loss']
            output['recon_loss'] = recon_loss
            output['total_loss'] = total_loss
        
        return output


class GumbelVQVAE(nn.Module):
    """VQ-VAE with Gumbel-Softmax (MLP encoder)."""
    
    def __init__(
        self,
        n_channels: int = 142,
        window_size: int = 10,
        embedding_dim: int = 128,
        num_codes: int = 256
    ):
        super().__init__()
        
        input_dim = n_channels * window_size
        
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
        
        self.vq = GumbelVectorQuantizer(num_codes, embedding_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return self.encoder(x.view(batch_size, -1))
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> dict:
        z_e = self.encode(x)
        z_q, vq_info = self.vq(z_e)
        velocity_pred = self.decoder(z_q)
        
        output = {'velocity_pred': velocity_pred, 'z_e': z_e, 'z_q': z_q, **vq_info}
        
        if targets is not None:
            recon_loss = F.mse_loss(velocity_pred, targets)
            total_loss = recon_loss + vq_info.get('diversity_loss', 0)
            output['recon_loss'] = recon_loss
            output['total_loss'] = total_loss
        
        return output


# ============================================================
# Factory
# ============================================================

def create_model(
    variant: str = 'progressive',
    n_channels: int = 142,
    window_size: int = 10,
    embedding_dim: int = 128,
    num_codes: int = 256
) -> nn.Module:
    """
    Create a model variant.
    
    Args:
        variant: 'progressive', 'transformer', 'gumbel', 'transformer_gumbel'
    """
    if variant == 'progressive':
        from .model import ProgressiveVQVAE
        return ProgressiveVQVAE(n_channels, window_size, embedding_dim, num_codes)
    
    elif variant == 'transformer':
        return TransformerVQVAE(n_channels, window_size, embedding_dim, num_codes, vq_type='ema')
    
    elif variant == 'gumbel':
        return GumbelVQVAE(n_channels, window_size, embedding_dim, num_codes)
    
    elif variant == 'transformer_gumbel':
        return TransformerVQVAE(n_channels, window_size, embedding_dim, num_codes, vq_type='gumbel')
    
    else:
        raise ValueError(f"Unknown variant: {variant}")
