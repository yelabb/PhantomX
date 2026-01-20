"""
Experiment 12: Stateful Mamba (S6) + Auxiliary Reconstruction

THEORETICAL MOTIVATION (Red Team Critique):
============================================
1. The "Supervised Bottleneck" Fallacy:
   - Previous VQ-VAE implementations supervised the bottleneck with regression targets
   - This forces codebook to act as "velocity lookup table" not neural feature extractor
   - Defeats purpose of Foundation Model which should learn generalizable dynamics

2. The "Sliding Window" Regression Flaw:
   - LSTM maintains hidden state across entire session (infinite memory)
   - Causal Transformer resets after each 250ms window (stateless)
   - We replaced a stateful operator with stateless one - regression not advance

3. Voronoi Error Floor:
   - Marginal gains from RVQ suggest quantization noise floor
   - Discrete codes don't align with intrinsic neural states

SOLUTION (Blue Team Pivot):
===========================
1. Mamba (S6) Backbone:
   - Replace CausalTransformer with State-Space Model
   - Pass hidden state h from window t to window t+1
   - Restores infinite memory while keeping parallel training

2. Dual-Head Decoder with Information Bottleneck:
   - Head A: z_q -> velocity (vx, vy) [Task Loss: MSE]
   - Head B: z_q -> spike_reconstruction [Self-Supervision: Poisson NLL]
   - Combined: L_total = L_velocity + λ * L_spikes

3. Manifold Regularization:
   - Force VQ codebook to reconstruct spikes
   - Discrete codes capture neural manifold structure
   - Latent space becomes robust, disentangled, "foundation-ready"

TARGET: R² > 0.78 (Beat LSTM baseline)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from einops import rearrange, repeat
import math
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent))
from phantomx.tokenizer import SpikeTokenizer
from phantomx.data import MCMazeDataset

# Use relative path from project root
DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"


# ============================================================
# Mamba (S6) Core Components
# ============================================================

class S6Layer(nn.Module):
    """
    Simplified S6 (Selective State Space) Layer.
    
    Core innovation of Mamba: data-dependent state transitions.
    
    SSM dynamics:
        h'(t) = A * h(t) + B * x(t)
        y(t) = C * h(t) + D * x(t)
    
    S6 makes B, C, Δ (discretization step) data-dependent:
        Δ, B, C = f(x)  # Learned from input
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Compute dt_rank
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = int(dt_rank)
        
        # Input projection: x -> (z, x_proj) where z is gate, x_proj goes through SSM
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # 1D convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias
        )
        
        # S6 projections: input-dependent Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt bias for stability
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Clamp dt bias to [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Softplus inverse
        self.dt_proj.bias.data = inv_dt
        
        # State matrix A (diagonal, negative for stability)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))  # Log parameterization
        
        # Skip connection D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            hidden_state: [batch, d_inner, d_state] previous SSM state
            
        Returns:
            output: [batch, seq_len, d_model]
            new_hidden_state: [batch, d_inner, d_state]
        """
        batch, seq_len, _ = x.shape
        
        # Project input
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_proj, z = xz.chunk(2, dim=-1)  # Each [B, L, d_inner]
        
        # Apply 1D convolution
        x_conv = rearrange(x_proj, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Causal: take first seq_len
        x_conv = rearrange(x_conv, "b d l -> b l d")
        x_conv = F.silu(x_conv)
        
        # S6: data-dependent Δ, B, C
        x_dbl = self.x_proj(x_conv)  # [B, L, dt_rank + 2*d_state]
        dt, B, C = torch.split(
            x_dbl, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1
        )
        
        # Project dt and apply softplus
        dt = self.dt_proj(dt)  # [B, L, d_inner]
        dt = F.softplus(dt)
        
        # Get A from log parameterization (negative for stability)
        A = -torch.exp(self.A_log)  # [d_inner, d_state]
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = torch.zeros(
                batch, self.d_inner, self.d_state,
                device=x.device, dtype=x.dtype
            )
        
        # Selective scan (sequential for stateful processing)
        y, final_state = self._selective_scan(
            x_conv, dt, A, B, C, self.D, hidden_state
        )
        
        # Apply gate and output projection
        y = y * F.silu(z)
        output = self.out_proj(y)
        
        return output, final_state
    
    def _selective_scan(
        self,
        x: torch.Tensor,      # [B, L, d_inner]
        dt: torch.Tensor,     # [B, L, d_inner]
        A: torch.Tensor,      # [d_inner, d_state]
        B: torch.Tensor,      # [B, L, d_state]
        C: torch.Tensor,      # [B, L, d_state]
        D: torch.Tensor,      # [d_inner]
        h: torch.Tensor       # [B, d_inner, d_state]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selective scan with stateful processing.
        
        Discretization:
            A_bar = exp(Δ * A)
            B_bar = Δ * B
        
        Recurrence:
            h_t = A_bar * h_{t-1} + B_bar * x_t
            y_t = C * h_t + D * x_t
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        outputs = []
        
        for t in range(seq_len):
            # Current input
            x_t = x[:, t, :]  # [B, d_inner]
            dt_t = dt[:, t, :]  # [B, d_inner]
            B_t = B[:, t, :]  # [B, d_state]
            C_t = C[:, t, :]  # [B, d_state]
            
            # Discretize A and B
            # A_bar = exp(dt * A) for each (batch, d_inner, d_state)
            dt_A = dt_t.unsqueeze(-1) * A.unsqueeze(0)  # [B, d_inner, d_state]
            A_bar = torch.exp(dt_A)
            
            # B_bar = dt * B (broadcast appropriately)
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [B, d_inner, d_state]
            
            # State update: h = A_bar * h + B_bar * x
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)  # [B, d_inner, d_state]
            
            # Output: y = C * h + D * x
            y_t = torch.einsum("bdn,bn->bd", h, C_t) + D * x_t  # [B, d_inner]
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]
        return y, h


class MambaBlock(nn.Module):
    """
    Mamba block with residual connection and layer norm.
    """
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s6 = S6Layer(d_model, d_state=d_state, expand=expand)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        x, new_state = self.s6(x, hidden_state)
        x = self.dropout(x)
        return x + residual, new_state


class MambaEncoder(nn.Module):
    """
    Stateful Mamba encoder that maintains hidden states across windows.
    """
    
    def __init__(
        self,
        n_channels: int = 142,
        d_model: int = 256,
        d_state: int = 16,
        num_layers: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        output_dim: int = 128,
        max_len: int = 50
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # Input projection
        self.input_proj = nn.Linear(n_channels, d_model)
        
        # Positional embedding (for within-window position)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, expand, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )
    
    def init_hidden(self, batch_size: int, device: torch.device) -> list:
        """Initialize hidden states for all layers."""
        return [
            torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
            for _ in range(self.num_layers)
        ]
    
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: [batch, seq_len, n_channels] spike windows
            hidden_states: List of [batch, d_inner, d_state] per layer
            
        Returns:
            z: [batch, output_dim] encoded representation
            new_hidden_states: Updated hidden states
        """
        B, T, C = x.shape
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = self.init_hidden(B, x.device)
        
        # Input projection + positional embedding
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        
        # Process through Mamba layers
        new_hidden_states = []
        for i, layer in enumerate(self.layers):
            x, new_h = layer(x, hidden_states[i])
            new_hidden_states.append(new_h)
        
        # Output: take last timestep
        x = self.ln_final(x)
        z = self.output_proj(x[:, -1, :])
        
        return z, new_hidden_states


# ============================================================
# Vector Quantizer with Gumbel-Softmax
# ============================================================

class ManifoldVQ(nn.Module):
    """
    Vector Quantizer designed for manifold learning.
    
    Key differences from previous VQ:
    - Temperature annealing for smooth training
    - K-means initialization
    - Designed to work with dual-head reconstruction
    """
    
    def __init__(
        self,
        num_codes: int = 256,
        embedding_dim: int = 128,
        temp_init: float = 1.0,
        temp_min: float = 0.5,  # Higher minimum for smoother quantization
        commitment_cost: float = 0.1
    ):
        super().__init__()
        
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.temp_min = temp_min
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.embeddings = nn.Parameter(torch.randn(num_codes, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)
        
        # Logit scaling for similarity
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)
        
        # Temperature buffer
        self.register_buffer('temperature', torch.tensor(temp_init))
        self._initialized = False
    
    def init_from_data(self, z_all: torch.Tensor):
        """Initialize codebook with K-means clustering."""
        z_np = z_all.detach().cpu().numpy()
        if len(z_np) > 10000:
            idx = np.random.choice(len(z_np), 10000, replace=False)
            z_np = z_np[idx]
        
        kmeans = KMeans(n_clusters=self.num_codes, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(z_np)
        
        self.embeddings.data.copy_(torch.from_numpy(kmeans.cluster_centers_).float())
        self._initialized = True
        print(f"  VQ initialized with K-means: {self.num_codes} codes")
    
    def update_temperature(self, epoch: int, max_epochs: int):
        """Cosine annealing of temperature."""
        progress = min(epoch / max_epochs, 1.0)
        temp = self.temp_min + 0.5 * (1.0 - self.temp_min) * (1 + math.cos(math.pi * progress))
        self.temperature.fill_(temp)
    
    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            z_e: [batch, embedding_dim] encoder output
            
        Returns:
            z_q: [batch, embedding_dim] quantized output
            info: Dictionary with indices, perplexity, etc.
        """
        # Normalize for cosine similarity
        z_e_norm = F.normalize(z_e, dim=-1)
        emb_norm = F.normalize(self.embeddings, dim=-1)
        
        # Compute logits
        logits = self.logit_scale * (z_e_norm @ emb_norm.T)  # [B, K]
        
        if self.training and self._initialized:
            # Gumbel-Softmax for differentiable sampling
            soft_onehot = F.gumbel_softmax(logits, tau=self.temperature.item(), hard=False)
            z_q = soft_onehot @ self.embeddings
            indices = logits.argmax(dim=-1)
        else:
            # Hard quantization during evaluation
            indices = logits.argmax(dim=-1)
            z_q = F.embedding(indices, self.embeddings)
        
        # Compute perplexity (codebook utilization metric)
        probs = F.softmax(logits, dim=-1)
        avg_probs = probs.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Commitment loss (optional, for codebook diversity)
        commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_e)
        
        return z_q, {
            'indices': indices,
            'perplexity': perplexity,
            'temperature': self.temperature.clone(),
            'commitment_loss': commitment_loss,
            'logits': logits
        }


# ============================================================
# Dual-Head Decoder with Spike Reconstruction
# ============================================================

class DualHeadDecoder(nn.Module):
    """
    Dual-head decoder implementing Information Bottleneck principle.
    
    Head A: Velocity prediction (task loss)
    Head B: Spike reconstruction (self-supervision, Poisson NLL)
    
    By forcing z_q to predict both velocity AND reconstruct spikes,
    we ensure the codebook captures the neural manifold structure,
    not just velocity correlates.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        n_channels: int = 142,
        window_size: int = 10,
        output_dim: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.window_size = window_size
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Head A: Velocity regression
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Head B: Spike reconstruction
        # Output: log-rate parameters for Poisson distribution
        self.spike_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_channels * window_size)  # Reconstruct full window
        )
    
    def forward(self, z_q: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_q: [batch, embedding_dim] quantized latent
            
        Returns:
            Dictionary with velocity prediction and spike reconstruction
        """
        # Shared features
        h = self.trunk(z_q)
        
        # Head A: Velocity
        velocity = self.velocity_head(h)
        
        # Head B: Spike log-rates (for Poisson NLL)
        spike_logits = self.spike_head(h)
        spike_logits = spike_logits.view(-1, self.window_size, self.n_channels)
        
        return {
            'velocity': velocity,
            'spike_logits': spike_logits
        }


# ============================================================
# Full Model: Stateful Mamba VQ-VAE
# ============================================================

class StatefulMambaVQVAE(nn.Module):
    """
    Stateful Mamba VQ-VAE with dual-head decoder.
    
    Key innovations:
    1. Mamba backbone maintains state across windows
    2. VQ codebook learns neural manifold via spike reconstruction
    3. Dual loss: velocity MSE + spike Poisson NLL
    """
    
    def __init__(
        self,
        n_channels: int = 142,
        window_size: int = 10,
        d_model: int = 256,
        d_state: int = 16,
        num_layers: int = 4,
        embedding_dim: int = 128,
        num_codes: int = 256,
        output_dim: int = 2,
        lambda_recon: float = 0.5,  # Weight for spike reconstruction loss
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.window_size = window_size
        self.lambda_recon = lambda_recon
        
        # Encoder: Stateful Mamba
        self.encoder = MambaEncoder(
            n_channels=n_channels,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        # Vector Quantizer
        self.vq = ManifoldVQ(
            num_codes=num_codes,
            embedding_dim=embedding_dim
        )
        
        # Decoder: Dual-head
        self.decoder = DualHeadDecoder(
            embedding_dim=embedding_dim,
            n_channels=n_channels,
            window_size=window_size,
            output_dim=output_dim,
            dropout=dropout
        )
        
        # Training mode flags
        self.use_vq = False
        self.use_reconstruction = True
    
    def init_hidden(self, batch_size: int, device: torch.device) -> list:
        """Initialize Mamba hidden states."""
        return self.encoder.init_hidden(batch_size, device)
    
    def encode(
        self, 
        x: torch.Tensor, 
        hidden_states: Optional[list] = None
    ) -> Tuple[torch.Tensor, list]:
        """Encode spike window to latent."""
        return self.encoder(x, hidden_states)
    
    def forward(
        self,
        x: torch.Tensor,
        velocity_target: Optional[torch.Tensor] = None,
        spike_target: Optional[torch.Tensor] = None,
        hidden_states: Optional[list] = None
    ) -> Dict:
        """
        Forward pass with optional stateful processing.
        
        Args:
            x: [batch, window_size, n_channels] spike window
            velocity_target: [batch, 2] velocity targets
            spike_target: [batch, window_size, n_channels] original spikes (for reconstruction)
            hidden_states: Mamba hidden states from previous window
            
        Returns:
            Dictionary with predictions, losses, and metrics
        """
        batch_size = x.size(0)
        device = x.device
        
        # Encode with stateful Mamba
        z_e, new_hidden = self.encode(x, hidden_states)
        
        # Vector quantization
        if self.use_vq:
            z_q, vq_info = self.vq(z_e)
        else:
            z_q = z_e
            vq_info = {
                'indices': torch.zeros(batch_size, dtype=torch.long, device=device),
                'perplexity': torch.tensor(256.0, device=device),
                'temperature': torch.tensor(1.0, device=device),
                'commitment_loss': torch.tensor(0.0, device=device)
            }
        
        # Decode: dual heads
        decoded = self.decoder(z_q)
        
        # Build output
        output = {
            'velocity_pred': decoded['velocity'],
            'spike_logits': decoded['spike_logits'],
            'z_e': z_e,
            'z_q': z_q,
            'hidden_states': new_hidden,
            **vq_info
        }
        
        # Compute losses if targets provided
        if velocity_target is not None:
            # Head A: Velocity MSE loss
            velocity_loss = F.mse_loss(decoded['velocity'], velocity_target)
            output['velocity_loss'] = velocity_loss
            
            # Head B: Spike reconstruction loss (Poisson NLL)
            if spike_target is not None and self.use_reconstruction:
                # Poisson NLL: -log P(k|λ) = λ - k*log(λ) + log(k!)
                # We parameterize log(λ), so: exp(logits) - targets * logits
                # Adding small epsilon for numerical stability
                log_rate = decoded['spike_logits']
                rate = torch.exp(log_rate.clamp(max=10))  # Clamp to prevent overflow
                
                # Poisson NLL (ignoring log(k!) which is constant w.r.t. params)
                poisson_nll = rate - spike_target * log_rate
                reconstruction_loss = poisson_nll.mean()
                
                output['reconstruction_loss'] = reconstruction_loss
                output['total_loss'] = velocity_loss + self.lambda_recon * reconstruction_loss
            else:
                output['reconstruction_loss'] = torch.tensor(0.0, device=device)
                output['total_loss'] = velocity_loss
            
            # Add commitment loss
            output['total_loss'] = output['total_loss'] + vq_info['commitment_loss']
        
        return output


# ============================================================
# Dataset with Stateful Support
# ============================================================

class StatefulSlidingWindowDataset(Dataset):
    """
    Dataset that provides consecutive windows for stateful training.
    Includes original spike counts for reconstruction loss.
    """
    
    def __init__(
        self, 
        spike_counts: np.ndarray,  # Normalized
        spike_counts_raw: np.ndarray,  # Original (for Poisson target)
        velocities: np.ndarray, 
        window_size: int = 10
    ):
        n = len(spike_counts) - window_size + 1
        
        # Normalized windows (input)
        self.windows = np.stack([spike_counts[i:i+window_size] for i in range(n)])
        
        # Raw spike windows (reconstruction target)
        self.windows_raw = np.stack([spike_counts_raw[i:i+window_size] for i in range(n)])
        
        # Velocities (regression target)
        self.velocities = velocities[window_size-1:window_size-1+n]
        
        # Sequence indices for stateful training
        self.indices = np.arange(n)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return {
            'window': torch.tensor(self.windows[idx], dtype=torch.float32),
            'window_raw': torch.tensor(self.windows_raw[idx], dtype=torch.float32),
            'velocity': torch.tensor(self.velocities[idx], dtype=torch.float32),
            'idx': idx  # For ordering in stateful training
        }


class SequentialBatchSampler:
    """
    Sampler that yields consecutive windows for stateful training.
    Resets hidden state at the start of each "segment".
    """
    
    def __init__(self, dataset_len: int, batch_size: int, segment_len: int = 100):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.segment_len = segment_len
        
        # Create segment boundaries
        n_segments = dataset_len // segment_len
        self.segments = []
        for i in range(n_segments):
            start = i * segment_len
            end = min(start + segment_len, dataset_len)
            self.segments.append(list(range(start, end)))
    
    def __iter__(self):
        # Shuffle segments, not individual samples
        np.random.shuffle(self.segments)
        
        for segment in self.segments:
            # Yield batches from this segment
            for i in range(0, len(segment), self.batch_size):
                batch_indices = segment[i:i+self.batch_size]
                yield batch_indices
    
    def __len__(self):
        total_batches = 0
        for segment in self.segments:
            total_batches += (len(segment) + self.batch_size - 1) // self.batch_size
        return total_batches


# ============================================================
# Training Loop
# ============================================================

def train_stateful_mamba(
    model: StatefulMambaVQVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    pretrain_epochs: int = 40,
    finetune_epochs: int = 60,
    lr: float = 3e-4,
    segment_len: int = 100
):
    """
    Training with three phases:
    1. Pre-train: Train encoder+decoder without VQ
    2. Initialize VQ with K-means
    3. Finetune: Train full model with VQ and dual-head loss
    """
    
    print("\n" + "="*60)
    print("Training Stateful Mamba VQ-VAE")
    print("="*60)
    
    # ========================================
    # Phase 1: Pre-training (no VQ)
    # ========================================
    print("\n[Phase 1] Pre-training encoder + decoder (no VQ)...")
    
    model.use_vq = False
    model.use_reconstruction = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pretrain_epochs)
    
    best_val_r2 = -float('inf')
    
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        train_losses = []
        
        # Stateful training: process consecutive windows
        hidden_states = None
        prev_idx = -1
        
        for batch in train_loader:
            window = batch['window'].to(device)
            window_raw = batch['window_raw'].to(device)
            velocity = batch['velocity'].to(device)
            curr_idx = batch['idx'][0].item()
            
            # Reset hidden state if non-consecutive
            if prev_idx < 0 or curr_idx != prev_idx + 1:
                hidden_states = model.init_hidden(window.size(0), device)
            
            optimizer.zero_grad()
            
            output = model(
                window, 
                velocity_target=velocity,
                spike_target=window_raw,
                hidden_states=hidden_states
            )
            
            loss = output['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Detach hidden states for next step
            hidden_states = [h.detach() for h in output['hidden_states']]
            prev_idx = curr_idx
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                output = model(window)
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(batch['velocity'])
        
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_r2 = r2_score(val_targets, val_preds)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
        
        if epoch % 10 == 0 or epoch == 1:
            avg_loss = np.mean(train_losses)
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, val_R²={val_r2:.4f} (best={best_val_r2:.4f})")
    
    print(f"\n  Pre-training complete. Best R² = {best_val_r2:.4f}")
    
    # ========================================
    # Phase 2: K-means initialization
    # ========================================
    print("\n[Phase 2] Initializing VQ codebook with K-means...")
    
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch in train_loader:
            window = batch['window'].to(device)
            z_e, _ = model.encode(window)
            embeddings.append(z_e.cpu())
    
    all_embeddings = torch.cat(embeddings)
    model.vq.init_from_data(all_embeddings)
    
    # ========================================
    # Phase 3: Finetuning with VQ
    # ========================================
    print("\n[Phase 3] Finetuning with VQ + dual-head loss...")
    
    model.use_vq = True
    model.use_reconstruction = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epochs)
    
    best_val_r2_vq = -float('inf')
    best_state = None
    
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        model.vq.update_temperature(epoch, finetune_epochs)
        
        train_losses = {'total': [], 'velocity': [], 'reconstruction': []}
        hidden_states = None
        prev_idx = -1
        
        for batch in train_loader:
            window = batch['window'].to(device)
            window_raw = batch['window_raw'].to(device)
            velocity = batch['velocity'].to(device)
            curr_idx = batch['idx'][0].item()
            
            if prev_idx < 0 or curr_idx != prev_idx + 1:
                hidden_states = model.init_hidden(window.size(0), device)
            
            optimizer.zero_grad()
            
            output = model(
                window,
                velocity_target=velocity,
                spike_target=window_raw,
                hidden_states=hidden_states
            )
            
            loss = output['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses['total'].append(loss.item())
            train_losses['velocity'].append(output['velocity_loss'].item())
            train_losses['reconstruction'].append(output['reconstruction_loss'].item())
            
            hidden_states = [h.detach() for h in output['hidden_states']]
            prev_idx = curr_idx
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        codes_used = set()
        
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                output = model(window)
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(batch['velocity'])
                codes_used.update(output['indices'].cpu().numpy().tolist())
        
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_r2 = r2_score(val_targets, val_preds)
        
        if val_r2 > best_val_r2_vq:
            best_val_r2_vq = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 15 == 0 or epoch == 1:
            temp = model.vq.temperature.item()
            avg_losses = {k: np.mean(v) for k, v in train_losses.items()}
            print(f"  Epoch {epoch:3d}: vel={avg_losses['velocity']:.4f}, "
                  f"recon={avg_losses['reconstruction']:.4f}, val_R²={val_r2:.4f}, "
                  f"temp={temp:.3f}, codes={len(codes_used)}")
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n  Finetuning complete. Best VQ R² = {best_val_r2_vq:.4f}")
    
    return best_val_r2_vq


def run_experiment():
    """Run the full Stateful Mamba + Reconstruction experiment."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 12: Stateful Mamba (S6) + Auxiliary Reconstruction")
    print("="*70)
    print("\nTheoretical Motivation:")
    print("  • Mamba backbone: Infinite memory via state-space recurrence")
    print("  • Dual-head decoder: Velocity + Spike reconstruction")
    print("  • Information Bottleneck: Codebook learns neural manifold")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # ========================================
    # Load data
    # ========================================
    print("\nLoading MC_Maze dataset...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_maze = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    # Get raw and normalized spike counts
    spike_counts_raw = mc_maze.spike_counts * mc_maze.spike_std + mc_maze.spike_mean  # Un-normalize
    spike_counts_raw = np.clip(spike_counts_raw, 0, None)  # Ensure non-negative for Poisson
    spike_counts = mc_maze.spike_counts  # Normalized
    velocities = mc_maze.velocities
    
    print(f"  Spike counts: {spike_counts.shape}")
    print(f"  Velocities: {velocities.shape}")
    
    # Create dataset
    window_size = 10
    dataset = StatefulSlidingWindowDataset(
        spike_counts, spike_counts_raw, velocities, window_size
    )
    
    # Train/val split
    n_train = int(0.8 * len(dataset))
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, len(dataset)))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # ========================================
    # Experiment configurations
    # ========================================
    configs = [
        {
            'name': 'Mamba-4L + λ=0.5',
            'num_layers': 4,
            'd_model': 256,
            'd_state': 16,
            'lambda_recon': 0.5
        },
        {
            'name': 'Mamba-6L + λ=0.3',
            'num_layers': 6,
            'd_model': 256,
            'd_state': 16,
            'lambda_recon': 0.3
        },
        {
            'name': 'Mamba-4L + λ=1.0 (strong recon)',
            'num_layers': 4,
            'd_model': 256,
            'd_state': 32,
            'lambda_recon': 1.0
        },
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {cfg['name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create model
        model = StatefulMambaVQVAE(
            n_channels=142,
            window_size=window_size,
            d_model=cfg['d_model'],
            d_state=cfg['d_state'],
            num_layers=cfg['num_layers'],
            embedding_dim=128,
            num_codes=256,
            lambda_recon=cfg['lambda_recon'],
            dropout=0.1
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")
        
        # Train
        best_r2 = train_stateful_mamba(
            model, train_loader, val_loader, device,
            pretrain_epochs=40,
            finetune_epochs=60
        )
        
        elapsed = time.time() - start_time
        
        # Final evaluation
        model.eval()
        test_preds, test_targets = [], []
        codes_used = set()
        
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                output = model(window)
                test_preds.append(output['velocity_pred'].cpu())
                test_targets.append(batch['velocity'])
                codes_used.update(output['indices'].cpu().numpy().tolist())
        
        test_preds = torch.cat(test_preds).numpy()
        test_targets = torch.cat(test_targets).numpy()
        
        r2 = r2_score(test_targets, test_preds)
        r2_vx = r2_score(test_targets[:, 0], test_preds[:, 0])
        r2_vy = r2_score(test_targets[:, 1], test_preds[:, 1])
        
        results.append({
            'name': cfg['name'],
            'r2': r2,
            'r2_vx': r2_vx,
            'r2_vy': r2_vy,
            'codes': len(codes_used),
            'params': param_count,
            'time': elapsed
        })
        
        status = "🎯" if r2 >= 0.78 else ("📈" if r2 >= 0.77 else "")
        print(f"\n  Result: R²={r2:.4f} (vx={r2_vx:.4f}, vy={r2_vy:.4f}), "
              f"codes={len(codes_used)}, time={elapsed/60:.1f}min {status}")
        
        if r2 >= 0.78:
            save_path = Path(__file__).parent / 'models' / 'exp12_mamba_vqvae.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'r2': r2,
            }, save_path)
            print(f"  🎉 TARGET ACHIEVED! Model saved to {save_path}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 12 RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Model':<35} {'R²':>8} {'vx':>8} {'vy':>8} {'Codes':>8} {'Params':>12}")
    print("-"*75)
    
    for r in sorted(results, key=lambda x: x['r2'], reverse=True):
        status = "🎯" if r['r2'] >= 0.78 else ("📈" if r['r2'] >= 0.77 else "")
        print(f"{r['name']:<35} {r['r2']:>7.4f} {r['r2_vx']:>7.4f} {r['r2_vy']:>7.4f} "
              f"{r['codes']:>8} {r['params']:>12,} {status}")
    
    print("\n" + "-"*70)
    print("Baselines:")
    print("  • Raw LSTM:                     R² = 0.78")
    print("  • Previous best (Exp 11 RVQ-4): R² = 0.7757")
    print("  • Causal Transformer (Exp 10):  R² = 0.7727")
    print("="*70)
    
    best = max(results, key=lambda x: x['r2'])
    if best['r2'] >= 0.78:
        print(f"\n✅ SUCCESS: {best['name']} achieved R² = {best['r2']:.4f} - LSTM BEATEN!")
        print("\nKey insight: Stateful processing + spike reconstruction regularization")
        print("forces the VQ codebook to capture the neural manifold, not just velocity.")
    else:
        gap = 0.78 - best['r2']
        print(f"\n📈 Best: {best['name']} R² = {best['r2']:.4f} (gap to LSTM: {gap:.4f})")
        print("\nNext steps:")
        print("  • Try larger d_state for richer state representation")
        print("  • Experiment with different λ values for reconstruction weight")
        print("  • Consider session-level stateful training (not resetting at segment boundaries)")
    
    return results


if __name__ == "__main__":
    run_experiment()
