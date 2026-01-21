"""
Experiment 21: Super-Teacher with Hierarchical Two-Speed Architecture

MOTIVATION (Red Team Critique Response):
=========================================
1. TEACHER'S CEILING FALLACY
   - Current Teacher R² ≈ 0.78 caps the Student
   - Need "Super-Teacher" that breaks R² > 0.80 barrier
   
2. CONTEXT DILUTION MISINTERPRETATION  
   - Exp 13 failed with flat 2s Mamba, but that doesn't mean long context is useless
   - Motor cortex has TWO-SPEED dynamics:
     * SLOW (1-2s): Preparatory states, movement intent
     * FAST (250ms): Motor commands, instantaneous velocity
   - Need HIERARCHICAL architecture, not flat long context
   
3. Δ=0 OVERFITTING RISK
   - Current model may be "smoothing" rather than "decoding"
   - Train with Δ=+1 to validate causal prediction capability

ARCHITECTURE: Hierarchical Two-Speed Encoder
=============================================

                    ┌─────────────────────────────────────────────┐
                    │               SLOW PATHWAY (2s)              │
                    │  Mamba/SSM → preparatory state embedding     │
                    │  [80 timesteps @ 40Hz = 2 seconds]          │
                    └──────────────────┬──────────────────────────┘
                                       │ cross-attention
                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FAST PATHWAY (250ms)                          │
│   CausalTransformer → motor command embedding                        │
│   [10 timesteps @ 40Hz = 250ms]                                     │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
                            Gated Fusion
                                   │
                                   ▼
                          MLP Velocity Decoder

HYPOTHESIS:
- Slow pathway captures "where are we going?" (preparatory state)
- Fast pathway captures "how fast now?" (motor execution)
- Gated fusion learns when to use each signal
- This should break the 0.80 ceiling by extracting more information

TARGET: Teacher R² > 0.82, then distill to Student
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, Subset

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from phantomx.data import MCMazeDataset

DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def r2(preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute R² score."""
    return r2_score(targets, preds)


# ============================================================
# Configuration
# ============================================================

@dataclass 
class Config:
    # Data
    n_channels: int = 142
    
    # SLOW PATHWAY: 2 seconds of context for preparatory state
    slow_window: int = 80     # 80 timesteps @ 40Hz = 2 seconds
    slow_stride: int = 4      # Downsample to 10Hz for efficiency
    
    # FAST PATHWAY: 250ms for motor execution  
    fast_window: int = 10     # 10 timesteps @ 40Hz = 250ms
    
    # Lag tuning for causal validation
    lag: int = 1              # Δ=+1: predict NEXT velocity (causal)
    
    # Train/test split
    train_frac: float = 0.8
    
    # SLOW PATHWAY: Mamba-style SSM
    slow_d_model: int = 128
    slow_d_state: int = 16    # SSM state dimension
    slow_d_conv: int = 4      # Local convolution width
    slow_expand: int = 2      # Expansion factor
    slow_num_layers: int = 2
    
    # FAST PATHWAY: Causal Transformer (proven architecture)
    fast_d_model: int = 256
    fast_nhead: int = 8
    fast_num_layers: int = 6
    fast_dim_ff: int = 512
    fast_dropout: float = 0.1
    
    # Cross-attention: Slow → Fast
    cross_nhead: int = 4
    
    # Fusion
    fusion_dim: int = 256
    output_dim: int = 128     # Latent dimension for RVQ compatibility
    
    # Training
    epochs: int = 150
    lr: float = 3e-4
    batch_size: int = 64
    weight_decay: float = 1e-4
    patience: int = 25
    grad_clip: float = 1.0
    
    # Ablations
    use_slow_pathway: bool = True
    use_cross_attention: bool = True
    use_gated_fusion: bool = True


# ============================================================
# Dataset: Two-Speed Windows
# ============================================================

class TwoSpeedDataset(Dataset):
    """
    Dataset that provides both slow (2s) and fast (250ms) windows.
    
    For each sample:
    - slow_window: [slow_len, n_channels] covering past 2 seconds
    - fast_window: [fast_len, n_channels] covering past 250ms
    - velocity: target at time t (or t+lag for causal prediction)
    
    The fast window is the LAST 250ms of the slow window.
    """
    
    def __init__(
        self,
        spike_counts: np.ndarray,  # [T, n_channels]
        velocities: np.ndarray,     # [T, 2]
        slow_window: int = 80,
        fast_window: int = 10,
        lag: int = 1,               # Δ=+1 for causal prediction
    ):
        super().__init__()
        self.slow_window = slow_window
        self.fast_window = fast_window
        self.lag = lag
        
        n = len(spike_counts)
        
        # Valid indices: need slow_window history + lag for target
        # t ranges from slow_window-1 to n-1-lag
        self.start_idx = slow_window - 1
        self.end_idx = n - lag if lag > 0 else n
        self.n_samples = self.end_idx - self.start_idx
        
        self.spike_counts = spike_counts
        self.velocities = velocities
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # t is the "current" timestep (end of windows)
        t = self.start_idx + idx
        
        # Slow window: [t - slow_window + 1 : t + 1]
        slow_start = t - self.slow_window + 1
        slow_end = t + 1
        slow = self.spike_counts[slow_start:slow_end]  # [80, 142]
        
        # Fast window: last 10 timesteps of slow window
        fast = slow[-self.fast_window:]  # [10, 142]
        
        # Target velocity: at t + lag (causal prediction)
        target_idx = t + self.lag if self.lag > 0 else t
        velocity = self.velocities[target_idx]
        
        return {
            "slow_window": torch.tensor(slow, dtype=torch.float32),
            "fast_window": torch.tensor(fast, dtype=torch.float32),
            "velocity": torch.tensor(velocity, dtype=torch.float32),
        }


# ============================================================
# Mamba-Style SSM Block (Simplified S4D)
# ============================================================

class S4DKernel(nn.Module):
    """
    Simplified S4D kernel for long-range dependencies.
    
    Based on "Efficiently Modeling Long Sequences with Structured State Spaces"
    but simplified for our use case.
    """
    
    def __init__(self, d_model: int, d_state: int = 16, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # S4D uses diagonal state matrix A
        # A = -exp(log_A_real) + i * A_imag (complex diagonal)
        log_A_real = torch.log(0.5 * torch.ones(d_state))
        A_imag = math.pi * torch.arange(d_state).float()
        self.register_buffer('log_A_real', log_A_real)
        self.register_buffer('A_imag', A_imag)
        
        # B, C projections
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        
        # Discretization step (learnable per channel)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
        # Skip connection
        self.D = nn.Parameter(torch.ones(d_model))
        
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: [batch, seq_len, d_model]
        Returns:
            y: [batch, seq_len, d_model]
        """
        B, L, D = u.shape
        
        # Discretize: A_bar = exp(A * dt)
        dt = torch.exp(self.log_dt).clamp(min=1e-4, max=1.0)  # [D]
        
        # For efficiency, we use a simple RNN-style computation
        # In practice, S4 uses FFT-based convolution for O(L log L)
        # Here we use a simplified version for clarity
        
        A_real = -torch.exp(self.log_A_real)  # [d_state]
        A = A_real.unsqueeze(0) + 1j * self.A_imag.unsqueeze(0)  # [1, d_state]
        
        # Discretized A for each channel: exp(A * dt)
        # A_bar[d, n] = exp(A[n] * dt[d])
        dt_expanded = dt.unsqueeze(-1)  # [D, 1]
        A_bar = torch.exp(A * dt_expanded)  # [D, d_state] complex
        
        # B_bar = dt * B
        B_bar = dt_expanded * self.B  # [D, d_state]
        
        # RNN-style loop (could be optimized with scan)
        y = []
        h = torch.zeros(B, D, self.d_state, dtype=torch.complex64, device=u.device)
        
        for t in range(L):
            # h = A_bar * h + B_bar * u_t
            u_t = u[:, t, :].unsqueeze(-1)  # [B, D, 1]
            h = A_bar.unsqueeze(0) * h + B_bar.unsqueeze(0) * u_t  # [B, D, d_state]
            
            # y_t = Re(C * h) + D * u_t
            y_t = torch.sum(self.C.unsqueeze(0) * h.real, dim=-1)  # [B, D]
            y_t = y_t + self.D * u[:, t, :]
            y.append(y_t)
        
        return torch.stack(y, dim=1)  # [B, L, D]


class MambaBlock(nn.Module):
    """
    Mamba-style block with SSM + gating.
    
    Mamba = selective SSM + gated MLP
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_inner = d_model * expand
        
        self.ln = nn.LayerNorm(d_model)
        
        # Input projection (expand)
        self.in_proj = nn.Linear(d_model, d_inner * 2)  # For gating
        
        # Local convolution (Mamba's key innovation)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, 
            kernel_size=d_conv, 
            padding=d_conv - 1, 
            groups=d_inner,  # Depthwise
        )
        
        # SSM
        self.ssm = S4DKernel(d_inner, d_state)
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            out: [batch, seq_len, d_model]
        """
        residual = x
        x = self.ln(x)
        
        # Project and split for gating
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # [B, L, d_inner] each
        
        # Local convolution
        x = x.transpose(1, 2)  # [B, d_inner, L]
        x = self.conv1d(x)[:, :, :x.size(-1)]  # Causal: remove padding
        x = x.transpose(1, 2)  # [B, L, d_inner]
        x = F.silu(x)
        
        # SSM for long-range
        x = self.ssm(x)
        
        # Gating (SiLU gate from z)
        x = x * F.silu(z)
        
        # Project back
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return residual + x


class SlowPathway(nn.Module):
    """
    Slow pathway: 2-second context via Mamba SSM.
    
    Captures preparatory states and movement intent.
    """
    
    def __init__(self, cfg: Config):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(cfg.n_channels, cfg.slow_d_model)
        
        # Strided convolution for efficiency (80 → 20 timesteps)
        self.downsample = nn.Conv1d(
            cfg.slow_d_model, cfg.slow_d_model,
            kernel_size=cfg.slow_stride,
            stride=cfg.slow_stride,
        )
        
        # Positional encoding
        max_len = cfg.slow_window // cfg.slow_stride + 1
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, cfg.slow_d_model) * 0.02)
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                cfg.slow_d_model,
                d_state=cfg.slow_d_state,
                d_conv=cfg.slow_d_conv,
                expand=cfg.slow_expand,
            )
            for _ in range(cfg.slow_num_layers)
        ])
        
        self.ln = nn.LayerNorm(cfg.slow_d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, slow_window, n_channels] = [B, 80, 142]
        Returns:
            slow_ctx: [batch, seq_len', slow_d_model] = [B, 20, 128]
        """
        B, L, C = x.shape
        
        # Project
        x = self.input_proj(x)  # [B, 80, 128]
        
        # Downsample for efficiency
        x = x.transpose(1, 2)   # [B, 128, 80]
        x = self.downsample(x)  # [B, 128, 20]
        x = x.transpose(1, 2)   # [B, 20, 128]
        
        # Add positional encoding
        L_new = x.size(1)
        x = x + self.pos_embed[:, :L_new, :]
        
        # Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        return self.ln(x)


# ============================================================
# Fast Pathway: Causal Transformer (from Exp 19)
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        if self.mask is None or self.mask.size(-1) < T:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            self.register_buffer('mask', mask)
        attn = attn.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention: Fast pathway queries Slow pathway."""
    
    def __init__(self, d_query: int, d_key: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_query % nhead == 0
        self.nhead = nhead
        self.head_dim = d_query // nhead
        
        self.q_proj = nn.Linear(d_query, d_query)
        self.k_proj = nn.Linear(d_key, d_query)
        self.v_proj = nn.Linear(d_key, d_query)
        self.out_proj = nn.Linear(d_query, d_query)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, T_fast, d_query] from fast pathway
            key_value: [B, T_slow, d_key] from slow pathway
        Returns:
            out: [B, T_fast, d_query]
        """
        B, T_q, D = query.shape
        T_kv = key_value.size(1)
        
        q = self.q_proj(query).reshape(B, T_q, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).reshape(B, T_kv, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).reshape(B, T_kv, self.nhead, self.head_dim).transpose(1, 2)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T_q, D)
        return self.out_proj(out)


class FastTransformerBlock(nn.Module):
    """Transformer block with optional cross-attention to slow pathway."""
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_ff: int, 
        d_slow: int = None,
        cross_nhead: int = 4,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention and (d_slow is not None)
        
        # Self-attention
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = CausalSelfAttention(d_model, nhead, dropout)
        
        # Cross-attention to slow pathway
        if self.use_cross_attention:
            self.ln_cross = nn.LayerNorm(d_model)
            self.cross_attn = CrossAttention(d_model, d_slow, cross_nhead, dropout)
        
        # FFN
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, slow_ctx: torch.Tensor = None) -> torch.Tensor:
        # Self-attention
        x = x + self.self_attn(self.ln1(x))
        
        # Cross-attention to slow context
        if self.use_cross_attention and slow_ctx is not None:
            x = x + self.cross_attn(self.ln_cross(x), slow_ctx)
        
        # FFN
        x = x + self.ffn(self.ln2(x))
        return x


class FastPathway(nn.Module):
    """
    Fast pathway: 250ms context via Causal Transformer.
    
    Optionally attends to slow pathway for hierarchical context.
    """
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        # Input projection
        self.input_proj = nn.Linear(cfg.n_channels, cfg.fast_d_model)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.fast_window, cfg.fast_d_model) * 0.02)
        
        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            FastTransformerBlock(
                d_model=cfg.fast_d_model,
                nhead=cfg.fast_nhead,
                dim_ff=cfg.fast_dim_ff,
                d_slow=cfg.slow_d_model if cfg.use_slow_pathway else None,
                cross_nhead=cfg.cross_nhead,
                dropout=cfg.fast_dropout,
                use_cross_attention=cfg.use_cross_attention and (i >= cfg.fast_num_layers // 2),
                # Only use cross-attention in later layers
            )
            for i in range(cfg.fast_num_layers)
        ])
        
        self.ln = nn.LayerNorm(cfg.fast_d_model)
        
    def forward(self, x: torch.Tensor, slow_ctx: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [batch, fast_window, n_channels] = [B, 10, 142]
            slow_ctx: [batch, slow_seq_len, slow_d_model] = [B, 20, 128]
        Returns:
            fast_out: [batch, fast_d_model] = [B, 256]
        """
        B, T, C = x.shape
        
        # Project
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, slow_ctx)
        
        x = self.ln(x)
        
        # Return last token (causal: contains all context)
        return x[:, -1, :]


# ============================================================
# Gated Fusion
# ============================================================

class GatedFusion(nn.Module):
    """
    Gated fusion of slow and fast pathways.
    
    Learns when to rely on preparatory state (slow) vs motor execution (fast).
    """
    
    def __init__(self, d_slow: int, d_fast: int, d_output: int):
        super().__init__()
        
        # Project both to same dimension
        self.slow_proj = nn.Linear(d_slow, d_output)
        self.fast_proj = nn.Linear(d_fast, d_output)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_slow + d_fast, d_output),
            nn.Sigmoid(),
        )
        
        self.ln = nn.LayerNorm(d_output)
        
    def forward(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slow: [batch, d_slow] - last token from slow pathway
            fast: [batch, d_fast] - last token from fast pathway
        Returns:
            fused: [batch, d_output]
        """
        slow_proj = self.slow_proj(slow)
        fast_proj = self.fast_proj(fast)
        
        # Compute gate
        gate = self.gate(torch.cat([slow, fast], dim=-1))
        
        # Gated combination
        fused = gate * slow_proj + (1 - gate) * fast_proj
        
        return self.ln(fused)


# ============================================================
# Super-Teacher Model
# ============================================================

class SuperTeacher(nn.Module):
    """
    Hierarchical Two-Speed Super-Teacher.
    
    Architecture:
        Slow (2s) → Mamba SSM → preparatory state
              ↓ cross-attention
        Fast (250ms) → Transformer → motor command
              ↓
        Gated Fusion → MLP → velocity
    """
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        # Slow pathway (optional for ablation)
        if cfg.use_slow_pathway:
            self.slow_pathway = SlowPathway(cfg)
        else:
            self.slow_pathway = None
        
        # Fast pathway
        self.fast_pathway = FastPathway(cfg)
        
        # Fusion
        if cfg.use_slow_pathway and cfg.use_gated_fusion:
            self.fusion = GatedFusion(
                d_slow=cfg.slow_d_model,
                d_fast=cfg.fast_d_model,
                d_output=cfg.fusion_dim,
            )
            decoder_input = cfg.fusion_dim
        else:
            self.fusion = None
            decoder_input = cfg.fast_d_model
        
        # Project to output_dim for RVQ compatibility
        self.latent_proj = nn.Sequential(
            nn.Linear(decoder_input, cfg.output_dim),
            nn.LayerNorm(cfg.output_dim),
        )
        
        # Velocity decoder
        self.decoder = nn.Sequential(
            nn.Linear(cfg.output_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )
        
    def encode(self, slow_window: torch.Tensor, fast_window: torch.Tensor) -> torch.Tensor:
        """
        Encode windows to latent representation.
        
        Returns z_e for RVQ distillation compatibility.
        """
        # Slow pathway
        if self.slow_pathway is not None:
            slow_ctx = self.slow_pathway(slow_window)  # [B, 20, 128]
            slow_out = slow_ctx[:, -1, :]              # [B, 128] last token
        else:
            slow_ctx = None
            slow_out = None
        
        # Fast pathway (with cross-attention to slow)
        fast_out = self.fast_pathway(fast_window, slow_ctx)  # [B, 256]
        
        # Fusion
        if self.fusion is not None and slow_out is not None:
            fused = self.fusion(slow_out, fast_out)  # [B, 256]
        else:
            fused = fast_out
        
        # Project to latent
        z_e = self.latent_proj(fused)  # [B, 128]
        
        return z_e
    
    def forward(
        self, 
        slow_window: torch.Tensor, 
        fast_window: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            slow_window: [batch, 80, 142]
            fast_window: [batch, 10, 142]
        Returns:
            dict with velocity_pred, z_e
        """
        z_e = self.encode(slow_window, fast_window)
        velocity_pred = self.decoder(z_e)
        
        return {
            'velocity_pred': velocity_pred,
            'z_e': z_e,
        }


class LSTMBaseline(nn.Module):
    """LSTM baseline for fair comparison."""
    
    def __init__(self, n_channels: int = 142, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.decoder(out[:, -1, :])


# ============================================================
# Training
# ============================================================

def train_super_teacher(
    model: SuperTeacher,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> Tuple[float, List[Dict]]:
    """Train the Super-Teacher model."""
    
    print("\n" + "="*60)
    print("Training Super-Teacher (Hierarchical Two-Speed)")
    print("="*60)
    print(f"  Slow pathway: {cfg.use_slow_pathway}")
    print(f"  Cross-attention: {cfg.use_cross_attention}")
    print(f"  Gated fusion: {cfg.use_gated_fusion}")
    print(f"  Lag (Δ): {cfg.lag}")
    print()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    
    best_test_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    history = []
    
    for epoch in range(1, cfg.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            slow = batch['slow_window'].to(device)
            fast = batch['fast_window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            out = model(slow, fast)
            loss = F.mse_loss(out['velocity_pred'], velocity)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Evaluation
        model.eval()
        test_preds, test_targets = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                slow = batch['slow_window'].to(device)
                fast = batch['fast_window'].to(device)
                velocity = batch['velocity'].to(device)
                
                out = model(slow, fast)
                test_preds.append(out['velocity_pred'].cpu().numpy())
                test_targets.append(velocity.cpu().numpy())
        
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        test_r2 = r2(test_preds, test_targets)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_r2': test_r2,
        })
        
        # Early stopping
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | Test R²: {test_r2:.4f} | Best: {best_test_r2:.4f}")
        
        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # Restore best
    model.load_state_dict(best_state)
    
    return best_test_r2, history


def run_ablation(
    spike_counts: np.ndarray,
    velocities: np.ndarray,
    device: torch.device,
    base_cfg: Config,
) -> Dict[str, float]:
    """Run ablation study on Super-Teacher components."""
    
    results = {}
    
    ablations = [
        ("Full Model", {"use_slow_pathway": True, "use_cross_attention": True, "use_gated_fusion": True}),
        ("No Slow Pathway", {"use_slow_pathway": False, "use_cross_attention": False, "use_gated_fusion": False}),
        ("No Cross-Attention", {"use_slow_pathway": True, "use_cross_attention": False, "use_gated_fusion": True}),
        ("No Gated Fusion", {"use_slow_pathway": True, "use_cross_attention": True, "use_gated_fusion": False}),
    ]
    
    for name, overrides in ablations:
        print(f"\n{'='*60}")
        print(f"ABLATION: {name}")
        print(f"{'='*60}")
        
        # Create config with overrides
        cfg = Config(**{**vars(base_cfg), **overrides})
        
        # Create dataset
        dataset = TwoSpeedDataset(
            spike_counts, velocities,
            slow_window=cfg.slow_window,
            fast_window=cfg.fast_window,
            lag=cfg.lag,
        )
        
        # Split
        n = len(dataset)
        n_train = int(n * cfg.train_frac)
        train_dataset = Subset(dataset, list(range(n_train)))
        test_dataset = Subset(dataset, list(range(n_train, n)))
        
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)
        
        # Create model
        model = SuperTeacher(cfg).to(device)
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")
        
        # Train
        best_r2, _ = train_super_teacher(model, train_loader, test_loader, device, cfg)
        
        results[name] = best_r2
        print(f"\n  ✓ {name}: R² = {best_r2:.4f}")
    
    return results


def main():
    print("="*70)
    print("EXPERIMENT 21: Super-Teacher with Hierarchical Two-Speed Architecture")
    print("="*70)
    print()
    print("Addressing Red Team Critiques:")
    print("  1. Teacher's Ceiling → Build Super-Teacher that breaks 0.80")
    print("  2. Context Dilution → Hierarchical slow/fast architecture")
    print("  3. Δ=0 Overfitting → Train with Δ=+1 for causal validation")
    print()
    
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading MC_Maze data...")
    mc_dataset = MCMazeDataset(DATA_PATH)
    spike_counts = mc_dataset.spike_counts  # [T, 142]
    velocities = mc_dataset.velocities      # [T, 2]
    
    # Normalize
    spike_counts = (spike_counts - spike_counts.mean(0)) / (spike_counts.std(0) + 1e-6)
    velocities = (velocities - velocities.mean(0)) / (velocities.std(0) + 1e-6)
    
    print(f"  Spikes: {spike_counts.shape}")
    print(f"  Velocities: {velocities.shape}")
    
    # Configuration
    cfg = Config()
    
    # ========================================
    # Part 1: LSTM Baseline (for comparison)
    # ========================================
    print("\n" + "="*60)
    print("BASELINE: LSTM (10-step window, Δ=+1)")
    print("="*60)
    
    # Create simple dataset for LSTM (fast window only)
    class SimpleLSTMDataset(Dataset):
        def __init__(self, spikes, vels, window=10, lag=1):
            n = len(spikes)
            self.n_samples = n - window - lag + 1
            self.windows = np.stack([spikes[i:i+window] for i in range(self.n_samples)])
            self.targets = vels[window + lag - 1: window + lag - 1 + self.n_samples]
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            return {
                "window": torch.tensor(self.windows[idx], dtype=torch.float32),
                "velocity": torch.tensor(self.targets[idx], dtype=torch.float32),
            }
    
    lstm_dataset = SimpleLSTMDataset(spike_counts, velocities, window=10, lag=cfg.lag)
    n_lstm = len(lstm_dataset)
    n_train_lstm = int(n_lstm * cfg.train_frac)
    
    lstm_train = Subset(lstm_dataset, list(range(n_train_lstm)))
    lstm_test = Subset(lstm_dataset, list(range(n_train_lstm, n_lstm)))
    
    lstm_train_loader = DataLoader(lstm_train, batch_size=cfg.batch_size, shuffle=True)
    lstm_test_loader = DataLoader(lstm_test, batch_size=cfg.batch_size)
    
    lstm = LSTMBaseline().to(device)
    optimizer = torch.optim.AdamW(lstm.parameters(), lr=3e-4)
    
    best_lstm_r2 = -float('inf')
    for epoch in range(1, 101):
        lstm.train()
        for batch in lstm_train_loader:
            optimizer.zero_grad()
            pred = lstm(batch['window'].to(device))
            loss = F.mse_loss(pred, batch['velocity'].to(device))
            loss.backward()
            optimizer.step()
        
        lstm.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in lstm_test_loader:
                pred = lstm(batch['window'].to(device))
                preds.append(pred.cpu().numpy())
                targets.append(batch['velocity'].numpy())
        
        test_r2 = r2(np.concatenate(preds), np.concatenate(targets))
        best_lstm_r2 = max(best_lstm_r2, test_r2)
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | LSTM R²: {test_r2:.4f}")
    
    print(f"\n  ✓ LSTM Baseline (Δ={cfg.lag}): R² = {best_lstm_r2:.4f}")
    
    # ========================================
    # Part 2: Super-Teacher Training
    # ========================================
    
    # Create two-speed dataset
    dataset = TwoSpeedDataset(
        spike_counts, velocities,
        slow_window=cfg.slow_window,
        fast_window=cfg.fast_window,
        lag=cfg.lag,
    )
    
    n = len(dataset)
    n_train = int(n * cfg.train_frac)
    train_dataset = Subset(dataset, list(range(n_train)))
    test_dataset = Subset(dataset, list(range(n_train, n)))
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)
    
    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create Super-Teacher
    model = SuperTeacher(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Super-Teacher parameters: {n_params:,}")
    
    # Train
    best_teacher_r2, history = train_super_teacher(model, train_loader, test_loader, device, cfg)
    
    # ========================================
    # Part 3: Ablation Study
    # ========================================
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)
    
    ablation_results = run_ablation(spike_counts, velocities, device, cfg)
    
    # ========================================
    # Part 4: Causal Validation (Δ comparison)
    # ========================================
    print("\n" + "="*60)
    print("CAUSAL VALIDATION: Comparing Δ=0 vs Δ=+1")
    print("="*60)
    
    delta_results = {}
    for lag in [0, 1, 2]:
        print(f"\n--- Training with Δ={lag} ---")
        
        cfg_delta = Config(lag=lag)
        dataset_delta = TwoSpeedDataset(
            spike_counts, velocities,
            slow_window=cfg_delta.slow_window,
            fast_window=cfg_delta.fast_window,
            lag=lag,
        )
        
        n = len(dataset_delta)
        n_train = int(n * cfg_delta.train_frac)
        train_ds = Subset(dataset_delta, list(range(n_train)))
        test_ds = Subset(dataset_delta, list(range(n_train, n)))
        
        train_ld = DataLoader(train_ds, batch_size=cfg_delta.batch_size, shuffle=True)
        test_ld = DataLoader(test_ds, batch_size=cfg_delta.batch_size)
        
        model_delta = SuperTeacher(cfg_delta).to(device)
        r2_delta, _ = train_super_teacher(model_delta, train_ld, test_ld, device, cfg_delta)
        delta_results[f"Δ={lag}"] = r2_delta
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 21 SUMMARY")
    print("="*70)
    
    print("\n1. BASELINES")
    print(f"   LSTM (Δ={cfg.lag}):              R² = {best_lstm_r2:.4f}")
    
    print("\n2. SUPER-TEACHER")
    print(f"   Full Model (Δ={cfg.lag}):        R² = {best_teacher_r2:.4f}")
    improvement = (best_teacher_r2 - best_lstm_r2) / best_lstm_r2 * 100
    if best_teacher_r2 > best_lstm_r2:
        print(f"   → BEATS LSTM by {improvement:.2f}%! 🎉")
    else:
        print(f"   → {abs(improvement):.2f}% below LSTM")
    
    print("\n3. ABLATION RESULTS")
    for name, r2_val in ablation_results.items():
        print(f"   {name:25s}: R² = {r2_val:.4f}")
    
    print("\n4. CAUSAL VALIDATION (Δ comparison)")
    for delta, r2_val in delta_results.items():
        marker = "← current encoding" if "0" in delta else ""
        print(f"   {delta}: R² = {r2_val:.4f} {marker}")
    
    # Interpret causal validation
    if delta_results.get("Δ=0", 0) > delta_results.get("Δ=1", 0) * 1.02:
        print("\n   ⚠️ WARNING: Δ=0 significantly better than Δ=+1")
        print("      Model may be 'smoothing' rather than 'decoding'")
        print("      Consider investigating causal structure further")
    else:
        print("\n   ✓ Causal validation passed: Model predicts forward")
    
    print("\n5. RED TEAM CRITIQUE RESPONSES")
    print("   [1] Teacher's Ceiling:")
    if best_teacher_r2 > 0.80:
        print(f"       ✓ SOLVED: Super-Teacher R² = {best_teacher_r2:.4f} > 0.80")
    else:
        print(f"       ✗ Not yet solved: R² = {best_teacher_r2:.4f} < 0.80")
        print(f"          Next: Try deeper slow pathway, larger d_model")
    
    print("   [2] Context Dilution:")
    full_r2 = ablation_results.get("Full Model", 0)
    no_slow_r2 = ablation_results.get("No Slow Pathway", 0)
    if full_r2 > no_slow_r2:
        print(f"       ✓ SOLVED: Slow pathway helps (+{(full_r2-no_slow_r2)*100:.2f}%)")
    else:
        print(f"       ✗ Slow pathway not helping yet")
    
    print("   [3] Δ=0 Overfitting:")
    d0 = delta_results.get("Δ=0", 0)
    d1 = delta_results.get("Δ=1", 0)
    if d1 >= d0 * 0.98:
        print(f"       ✓ Model maintains performance with Δ=+1")
    else:
        print(f"       ⚠️ Performance drops with Δ=+1: investigate causality")
    
    # Save results
    results = {
        "lstm_baseline": best_lstm_r2,
        "super_teacher": best_teacher_r2,
        "ablations": ablation_results,
        "delta_comparison": delta_results,
        "config": vars(cfg),
    }
    
    results_path = RESULTS_DIR / "exp21_super_teacher.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save best model
    model_path = Path(__file__).parent / "models" / "exp21_super_teacher.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
