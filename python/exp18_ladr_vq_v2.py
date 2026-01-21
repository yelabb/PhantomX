"""
Experiment 18: Lag-Aware Distilled RVQ (LADR-VQ) v2

CRITICAL FIXES FROM EXP 17:
1. Codebook initialization AFTER teacher pre-training (not before!)
2. Proper latent distillation: L_distill = MSE(z_q, z_e.detach())
3. Lag tuning: Δ=+1 aligns motor cortex planning (~25ms) with execution

STRATEGY: Teacher-Student Distillation + Lag Tuning
- The Transformer Encoder (Teacher) achieves R² = 0.784 without quantization
- The drop to 0.776 is the "discretization tax" 
- Goal: Close this gap via distillation

WHY THIS WORKS:
1. Gap Closure: Distillation explicitly minimizes distance between 
   "Perfect" teacher vector (z_e) and "Quantized" student vector (z_q)
2. Ceiling Raise: Lag tuning (Δ=+1) typically raises decoding ceiling
   by 2-5% in motor cortex data (aligning neural planning to execution)
3. Expected: If Lag raises Teacher to ~0.80, and Distillation keeps 
   the drop to <1%, we land at R² ≈ 0.79, beating LSTM

THREE-PHASE TRAINING:
  Phase 1 (Teacher): Pre-train encoder + decoder without VQ (use z_e directly)
  Phase 2 (Init):    Pass dataset through trained encoder, K-Means init RVQ
  Phase 3 (Student): Fine-tune with VQ enabled, using Distillation Loss
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from phantomx.tokenizer import SpikeTokenizer
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
    window_size: int = 10
    lag: int = 1  # Δ = +1 (25ms forward shift for motor cortex planning)
    train_frac: float = 0.7
    val_frac: float = 0.15
    
    # Model architecture
    n_channels: int = 142
    d_model: int = 256
    embedding_dim: int = 128
    num_layers: int = 6
    nhead: int = 8
    dim_ff: int = 512
    dropout: float = 0.1
    
    # RVQ
    num_quantizers: int = 4
    num_codes: int = 128
    commitment_cost: float = 0.25
    
    # Training - Phase 1 (Teacher)
    pretrain_epochs: int = 60
    pretrain_lr: float = 3e-4
    
    # Training - Phase 3 (Student with distillation)
    finetune_epochs: int = 100
    lr_encoder: float = 1e-5   # Very low - encoder mostly frozen
    lr_vq: float = 1e-4        # Train codebooks
    lr_decoder: float = 1e-4   # Fine-tune decoder
    
    # Distillation loss weights
    alpha: float = 1.0   # Weight for velocity MSE (ground truth)
    beta: float = 0.5    # Weight for latent distillation (z_q → z_e)
    
    # General
    batch_size: int = 64
    weight_decay: float = 1e-4
    patience: int = 30
    grad_clip: float = 1.0


# ============================================================
# Dataset with Lag Tuning (Δ = +1)
# ============================================================

class LaggedSlidingWindowDataset(Dataset):
    """
    Sliding window dataset with lagged targets.
    
    For motor cortex decoding, the brain plans movements ~25ms in advance.
    With 25ms bins, lag=1 aligns neural activity with the NEXT velocity,
    which is what the brain is actually computing.
    
    Args:
        spike_counts: [T, C] neural spike counts
        velocities: [T, 2] velocity targets
        window_size: Number of time bins per window
        lag: Target shift (default=1 means predict velocity at t+1)
    """
    
    def __init__(
        self,
        spike_counts: np.ndarray,
        velocities: np.ndarray,
        window_size: int = 10,
        lag: int = 1,
    ):
        self.window_size = window_size
        self.lag = lag
        
        # Calculate valid indices accounting for window and lag
        # For window ending at t, target is velocity[t + lag]
        n = len(spike_counts)
        max_window_end = n - 1 - lag if lag > 0 else n - 1 + lag
        min_window_end = window_size - 1
        
        if lag >= 0:
            n_samples = max_window_end - min_window_end + 1
        else:
            n_samples = min(n - window_size + 1, n + lag)
        
        n_samples = max(0, n_samples)
        
        # Build windows and targets
        self.windows = np.stack([
            spike_counts[i:i + window_size]
            for i in range(n_samples)
        ])
        
        # Target velocity is shifted by lag
        target_indices = np.arange(window_size - 1 + lag, window_size - 1 + lag + n_samples)
        target_indices = np.clip(target_indices, 0, n - 1)
        self.velocities = velocities[target_indices]
        
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "window": torch.tensor(self.windows[idx], dtype=torch.float32),
            "velocity": torch.tensor(self.velocities[idx], dtype=torch.float32),
        }


# ============================================================
# Causal Transformer Encoder (from exp12)
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


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CausalTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int = 142,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_ff: int = 512,
        dropout: float = 0.1,
        output_dim: int = 128,
        max_len: int = 50,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.output_proj(x[:, -1, :])  # Last token


# ============================================================
# Residual Vector Quantization (RVQ)
# ============================================================

class ResidualVQLayer(nn.Module):
    """
    Single layer of residual vector quantization with EMA updates.
    """
    
    def __init__(
        self,
        num_codes: int = 128,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_codes))
        self.register_buffer('ema_embedding', self.embedding.weight.data.clone())
        self.register_buffer('initialized', torch.tensor(False))
    
    def init_from_data(self, z: torch.Tensor):
        """Initialize codebook using K-means on encoder outputs."""
        z_np = z.detach().cpu().numpy()
        n_samples = len(z_np)
        n_clusters = min(self.num_codes, n_samples // 2)
        
        if n_clusters < self.num_codes:
            print(f"      Warning: Using {n_clusters} clusters (need more samples for {self.num_codes})")
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(z_np)
        
        centroids = torch.from_numpy(kmeans.cluster_centers_).float()
        
        # Fill remaining codes with perturbed centroids
        if n_clusters < self.num_codes:
            remaining = self.num_codes - n_clusters
            random_indices = torch.randint(0, n_clusters, (remaining,))
            noise = torch.randn(remaining, self.embedding_dim) * 0.1
            extra_centroids = centroids[random_indices] + noise
            centroids = torch.cat([centroids, extra_centroids], dim=0)
        
        self.embedding.weight.data.copy_(centroids.to(self.embedding.weight.device))
        self.ema_embedding.copy_(self.embedding.weight.data)
        self.initialized.fill_(True)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        z_flat = z.view(-1, self.embedding_dim)
        
        # Compute distances
        z_norm = torch.sum(z_flat ** 2, dim=1, keepdim=True)
        e_norm = torch.sum(self.embedding.weight ** 2, dim=1)
        distances = z_norm + e_norm.unsqueeze(0) - 2 * z_flat @ self.embedding.weight.t()
        
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)
        
        commitment_loss = F.mse_loss(z_q.detach(), z_flat) * self.commitment_cost
        
        # Straight-through estimator
        z_q_st = z_flat + (z_q - z_flat).detach()
        
        if self.training:
            self._ema_update(z_flat, indices)
        
        # Perplexity (codebook utilization)
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        residual = z_flat - z_q_st
        
        return z_q_st.view_as(z), residual.view_as(z), {
            'indices': indices,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity
        }
    
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        with torch.no_grad():
            encodings = F.one_hot(indices, self.num_codes).float()
            self.ema_cluster_size = self.decay * self.ema_cluster_size + (1 - self.decay) * encodings.sum(0)
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon) /
                (n + self.num_codes * self.epsilon) * n
            )
            dw = encodings.t() @ z
            self.ema_embedding = self.decay * self.ema_embedding + (1 - self.decay) * dw
            self.embedding.weight.data = self.ema_embedding / self.ema_cluster_size.unsqueeze(1)


class ResidualVectorQuantizer(nn.Module):
    """
    Multi-stage Residual Vector Quantizer (RVQ-N).
    
    Applies N quantization stages in cascade for exponential variance reduction.
    """
    
    def __init__(
        self,
        num_quantizers: int = 4,
        num_codes: int = 128,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        
        self.quantizers = nn.ModuleList([
            ResidualVQLayer(num_codes, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])
    
    def init_from_data(self, z_all: torch.Tensor):
        """Initialize all codebooks progressively from encoder outputs."""
        device = z_all.device
        residual = z_all.clone()
        
        for i, quantizer in enumerate(self.quantizers):
            print(f"      Initializing RVQ layer {i+1}/{self.num_quantizers}...")
            quantizer.init_from_data(residual)
            
            with torch.no_grad():
                z_q, new_residual, _ = quantizer(residual)
                residual = new_residual.to(device)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        z_q_sum = torch.zeros_like(z)
        residual = z
        
        all_indices = []
        total_commitment_loss = 0.0
        total_perplexity = 0.0
        layer_perplexities = []
        
        for i, quantizer in enumerate(self.quantizers):
            z_q_i, residual, info = quantizer(residual)
            z_q_sum = z_q_sum + z_q_i
            
            all_indices.append(info['indices'])
            total_commitment_loss = total_commitment_loss + info['commitment_loss']
            total_perplexity = total_perplexity + info['perplexity']
            layer_perplexities.append(info['perplexity'].item())
        
        return z_q_sum, {
            'indices': all_indices[0],
            'all_indices': all_indices,
            'commitment_loss': total_commitment_loss / self.num_quantizers,
            'perplexity': total_perplexity / self.num_quantizers,
            'layer_perplexities': layer_perplexities,
            'residual_norm': residual.norm(dim=-1).mean()
        }


# ============================================================
# LADR-VQ Model
# ============================================================

class LADRVQModel(nn.Module):
    """
    Lag-Aware Distilled RVQ Model.
    
    Architecture:
        Spikes → CausalTransformer → [Optional RVQ] → MLP Decoder → Velocity
    
    The model can operate in two modes:
    - Teacher mode (use_vq=False): Direct z_e → decoder
    - Student mode (use_vq=True): z_e → RVQ → z_q → decoder
    """
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.use_vq = False  # Start in teacher mode
        
        self.encoder = CausalTransformerEncoder(
            n_channels=cfg.n_channels,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_ff=cfg.dim_ff,
            dropout=cfg.dropout,
            output_dim=cfg.embedding_dim,
        )
        
        self.vq = ResidualVectorQuantizer(
            num_quantizers=cfg.num_quantizers,
            num_codes=cfg.num_codes,
            embedding_dim=cfg.embedding_dim,
            commitment_cost=cfg.commitment_cost,
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(cfg.embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spike windows to continuous latent z_e."""
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns dict with:
            - velocity_pred: [B, 2] predicted velocity
            - z_e: [B, embedding_dim] continuous encoder output
            - z_q: [B, embedding_dim] quantized output (=z_e if use_vq=False)
            - commitment_loss: RVQ commitment loss
            - perplexity: codebook utilization metric
        """
        z_e = self.encode(x)
        
        if self.use_vq:
            z_q, vq_info = self.vq(z_e)
        else:
            z_q = z_e
            vq_info = {
                'indices': torch.zeros(x.size(0), dtype=torch.long, device=x.device),
                'perplexity': torch.tensor(0.0, device=x.device),
                'commitment_loss': torch.tensor(0.0, device=x.device),
                'residual_norm': torch.tensor(0.0, device=x.device),
            }
        
        velocity_pred = self.decoder(z_q)
        
        return {
            'velocity_pred': velocity_pred,
            'z_e': z_e,
            'z_q': z_q,
            **vq_info,
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
# Training Functions
# ============================================================

def train_phase1_teacher(
    model: LADRVQModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> float:
    """
    Phase 1: Train the Teacher (encoder + decoder without VQ).
    
    This establishes the upper bound for what the quantized model can achieve.
    With lag tuning (Δ=+1), we expect R² > 0.784.
    """
    print("\n" + "="*60)
    print("PHASE 1: Training Teacher (Continuous, No Quantization)")
    print("="*60)
    
    model.use_vq = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.pretrain_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.pretrain_epochs)
    
    best_val_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, cfg.pretrain_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window)
            loss = F.mse_loss(output['velocity_pred'], velocity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['window'].to(device))
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(batch['velocity'])
        
        val_r2 = r2(torch.cat(val_preds).numpy(), torch.cat(val_targets).numpy())
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: val_R²={val_r2:.4f} (best={best_val_r2:.4f})")
        
        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n  ✓ Teacher training complete. Best R² = {best_val_r2:.4f}")
    return best_val_r2


def init_phase2_codebooks(
    model: LADRVQModel,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Phase 2: Initialize RVQ codebooks from trained encoder outputs.
    
    CRITICAL FIX from Exp 17: This must happen AFTER teacher pre-training,
    not before! Otherwise K-means runs on random encoder outputs and 
    collapses to ~4 active codes.
    """
    print("\n" + "="*60)
    print("PHASE 2: Initializing Codebooks from Teacher Latents")
    print("="*60)
    print("  (Running K-means on trained encoder outputs)")
    
    model.eval()
    all_latents = []
    
    with torch.no_grad():
        for batch in train_loader:
            z_e = model.encode(batch['window'].to(device))
            all_latents.append(z_e)
    
    all_latents = torch.cat(all_latents)
    print(f"  Collected {len(all_latents)} latent vectors (dim={all_latents.shape[1]})")
    
    # Initialize RVQ on the learned latent manifold
    model.vq.init_from_data(all_latents)
    
    print("  ✓ Codebook initialization complete.")


def train_phase3_student(
    model: LADRVQModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> float:
    """
    Phase 3: Train the Student (with VQ) using Distillation Loss.
    
    The distillation loss forces z_q to stay close to z_e, minimizing
    the "discretization tax" while maintaining discrete representations.
    
    Loss = α * L_velocity + β * L_distill + L_commitment
    
    Where:
        L_velocity = MSE(velocity_pred, velocity_target)
        L_distill  = MSE(z_q, z_e.detach())  # Force quantized to match continuous
        L_commitment = standard VQ commitment loss
    """
    print("\n" + "="*60)
    print("PHASE 3: Student Distillation Fine-tuning")
    print("="*60)
    print(f"  Loss weights: α={cfg.alpha} (velocity), β={cfg.beta} (distillation)")
    
    model.use_vq = True
    
    # Differential learning rates:
    # - Encoder: very low (mostly frozen, slight adaptation)
    # - VQ: higher (train codebooks to match latent manifold)
    # - Decoder: moderate (adapt to quantized representations)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': cfg.lr_encoder},
        {'params': model.vq.parameters(), 'lr': cfg.lr_vq},
        {'params': model.decoder.parameters(), 'lr': cfg.lr_decoder},
    ], weight_decay=cfg.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.finetune_epochs)
    
    best_val_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, cfg.finetune_epochs + 1):
        # Training
        model.train()
        epoch_l_vel = 0.0
        epoch_l_distill = 0.0
        epoch_l_commit = 0.0
        
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window)
            
            # === DISTILLATION LOSS ===
            # L_velocity: Ground truth supervision
            l_velocity = F.mse_loss(output['velocity_pred'], velocity)
            
            # L_distill: Force z_q to match z_e (teacher's continuous thought)
            # Detach z_e so gradients only flow to z_q (codebook updates)
            l_distill = F.mse_loss(output['z_q'], output['z_e'].detach())
            
            # Commitment loss from RVQ
            l_commit = output['commitment_loss']
            
            # Total loss
            total_loss = cfg.alpha * l_velocity + cfg.beta * l_distill + l_commit
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            
            epoch_l_vel += l_velocity.item()
            epoch_l_distill += l_distill.item()
            epoch_l_commit += l_commit.item() if isinstance(l_commit, torch.Tensor) else l_commit
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        codes_per_layer = [set() for _ in range(cfg.num_quantizers)]
        
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['window'].to(device))
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(batch['velocity'])
                
                if 'all_indices' in output:
                    for i, indices in enumerate(output['all_indices']):
                        codes_per_layer[i].update(indices.cpu().numpy().tolist())
        
        val_r2 = r2(torch.cat(val_preds).numpy(), torch.cat(val_targets).numpy())
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == 1:
            n_batches = len(train_loader)
            codes_info = "/".join([str(len(s)) for s in codes_per_layer])
            print(f"  Epoch {epoch:3d}: val_R²={val_r2:.4f} (best={best_val_r2:.4f}) "
                  f"| L_vel={epoch_l_vel/n_batches:.4f} L_distill={epoch_l_distill/n_batches:.4f} "
                  f"| codes=[{codes_info}]")
        
        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n  ✓ Student training complete. Best R² = {best_val_r2:.4f}")
    return best_val_r2


def train_lstm_baseline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
) -> Tuple[float, float]:
    """Train LSTM baseline for fair comparison."""
    model = LSTMBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_r2 = -float('inf')
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['window'].to(device))
            loss = F.mse_loss(pred, batch['velocity'].to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['window'].to(device))
                val_preds.append(pred.cpu())
                val_targets.append(batch['velocity'])
        
        val_r2_score = r2(torch.cat(val_preds).numpy(), torch.cat(val_targets).numpy())
        if val_r2_score > best_val_r2:
            best_val_r2 = val_r2_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Test evaluation
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch['window'].to(device))
            test_preds.append(pred.cpu())
            test_targets.append(batch['velocity'])
    
    test_r2 = r2(torch.cat(test_preds).numpy(), torch.cat(test_targets).numpy())
    return best_val_r2, test_r2


def evaluate_model(
    model: LADRVQModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model and return detailed metrics."""
    model.eval()
    preds, targets = [], []
    codes_per_layer = [set() for _ in range(model.cfg.num_quantizers)]
    z_distances = []
    
    with torch.no_grad():
        for batch in loader:
            output = model(batch['window'].to(device))
            preds.append(output['velocity_pred'].cpu())
            targets.append(batch['velocity'])
            
            if model.use_vq and 'all_indices' in output:
                for i, indices in enumerate(output['all_indices']):
                    codes_per_layer[i].update(indices.cpu().numpy().tolist())
                
                # Measure distillation quality: ||z_q - z_e||
                z_dist = (output['z_q'] - output['z_e']).norm(dim=-1).mean()
                z_distances.append(z_dist.item())
    
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    
    r2_total = r2(preds, targets)
    r2_vx = r2(preds[:, 0], targets[:, 0])
    r2_vy = r2(preds[:, 1], targets[:, 1])
    
    return {
        'r2': r2_total,
        'r2_vx': r2_vx,
        'r2_vy': r2_vy,
        'codes_per_layer': [len(s) for s in codes_per_layer],
        'z_distance': np.mean(z_distances) if z_distances else 0.0,
    }


# ============================================================
# Data Loading
# ============================================================

def load_and_prepare_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Load and normalize MC_Maze dataset."""
    tokenizer = SpikeTokenizer(n_channels=cfg.n_channels)
    mc_maze = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    spikes = mc_maze.spike_counts
    velocities = mc_maze.velocities
    
    # Train-only normalization (prevent data leakage)
    train_end = int(cfg.train_frac * len(spikes))
    spike_mean = spikes[:train_end].mean(0, keepdims=True)
    spike_std = spikes[:train_end].std(0, keepdims=True) + 1e-6
    vel_mean = velocities[:train_end].mean(0, keepdims=True)
    vel_std = velocities[:train_end].std(0, keepdims=True) + 1e-6
    
    spikes = (spikes - spike_mean) / spike_std
    velocities = (velocities - vel_mean) / vel_std
    
    return spikes, velocities


def create_dataloaders(
    spikes: np.ndarray,
    velocities: np.ndarray,
    cfg: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders with lag tuning."""
    dataset = LaggedSlidingWindowDataset(
        spikes, velocities,
        window_size=cfg.window_size,
        lag=cfg.lag,
    )
    
    n_total = len(dataset)
    n_train = int(cfg.train_frac * n_total)
    n_val = int(cfg.val_frac * n_total)
    
    train_ds = Subset(dataset, list(range(0, n_train)))
    val_ds = Subset(dataset, list(range(n_train, n_train + n_val)))
    test_ds = Subset(dataset, list(range(n_train + n_val, n_total)))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    print("\n" + "="*70)
    print("EXPERIMENT 18: Lag-Aware Distilled RVQ (LADR-VQ) v2")
    print("="*70)
    print("\nObjective: Close the 'discretization tax' gap via distillation")
    print("Strategy:  Teacher-Student Distillation + Lag Tuning (Δ=+1)")
    print("Target:    R² > 0.78 (beat LSTM baseline)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    cfg = Config()
    print(f"\nConfiguration:")
    print(f"  Lag (Δ): {cfg.lag} bins ({cfg.lag * 25}ms forward shift)")
    print(f"  Window: {cfg.window_size} bins ({cfg.window_size * 25}ms)")
    print(f"  RVQ: {cfg.num_quantizers} layers × {cfg.num_codes} codes")
    print(f"  Distillation: α={cfg.alpha} (velocity), β={cfg.beta} (latent)")
    
    # Load data
    print("\n" + "-"*60)
    print("Loading MC_Maze dataset...")
    spikes, velocities = load_and_prepare_data(cfg)
    train_loader, val_loader, test_loader = create_dataloaders(spikes, velocities, cfg)
    
    print(f"Dataset sizes: train={len(train_loader.dataset)}, "
          f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize model
    model = LADRVQModel(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    start_time = time.time()
    
    # ========================================
    # PHASE 1: Train Teacher (No VQ)
    # ========================================
    teacher_r2 = train_phase1_teacher(model, train_loader, val_loader, device, cfg)
    
    # ========================================
    # PHASE 2: Initialize Codebooks (THE FIX!)
    # ========================================
    init_phase2_codebooks(model, train_loader, device)
    
    # ========================================
    # PHASE 3: Train Student with Distillation
    # ========================================
    student_r2 = train_phase3_student(model, train_loader, val_loader, device, cfg)
    
    elapsed = time.time() - start_time
    
    # Final evaluation on test set
    print("\n" + "-"*60)
    print("Final Evaluation on Test Set")
    print("-"*60)
    
    # Evaluate Student (with VQ)
    model.use_vq = True
    student_metrics = evaluate_model(model, test_loader, device)
    
    # Evaluate Teacher (without VQ, same model but VQ disabled)
    model.use_vq = False
    teacher_metrics = evaluate_model(model, test_loader, device)
    
    # LSTM Baseline
    print("\nTraining LSTM baseline for comparison...")
    lstm_val_r2, lstm_test_r2 = train_lstm_baseline(
        train_loader, val_loader, test_loader, device
    )
    
    # Results Summary
    print("\n" + "="*70)
    print("EXPERIMENT 18 RESULTS")
    print("="*70)
    
    discretization_tax = teacher_metrics['r2'] - student_metrics['r2']
    beat_lstm = student_metrics['r2'] > lstm_test_r2
    
    print(f"\n{'Model':<25} {'Val R²':>10} {'Test R²':>10} {'vx':>8} {'vy':>8}")
    print("-"*70)
    print(f"{'Teacher (no VQ)':<25} {teacher_r2:>10.4f} {teacher_metrics['r2']:>10.4f} "
          f"{teacher_metrics['r2_vx']:>8.4f} {teacher_metrics['r2_vy']:>8.4f}")
    print(f"{'Student (LADR-VQ)':<25} {student_r2:>10.4f} {student_metrics['r2']:>10.4f} "
          f"{student_metrics['r2_vx']:>8.4f} {student_metrics['r2_vy']:>8.4f}")
    print(f"{'LSTM Baseline':<25} {lstm_val_r2:>10.4f} {lstm_test_r2:>10.4f}")
    
    print(f"\n{'Analysis:':<25}")
    print(f"  Discretization tax: {discretization_tax:.4f} ({discretization_tax*100:.2f}%)")
    print(f"  Latent distance (||z_q - z_e||): {student_metrics['z_distance']:.4f}")
    print(f"  Codes per layer: {student_metrics['codes_per_layer']}")
    print(f"  Training time: {elapsed/60:.1f} min")
    
    print("\n" + "-"*70)
    if beat_lstm:
        print(f"🎉 SUCCESS: Student R²={student_metrics['r2']:.4f} > LSTM R²={lstm_test_r2:.4f}")
        print("   LADR-VQ has beaten the LSTM baseline!")
        
        # Save best model
        save_path = Path(__file__).parent / 'models' / 'exp18_ladr_vq_best.pt'
        torch.save(model.state_dict(), save_path)
        print(f"   Model saved to {save_path}")
    else:
        gap = lstm_test_r2 - student_metrics['r2']
        print(f"📈 Gap to LSTM: {gap:.4f}")
        print(f"   Student: {student_metrics['r2']:.4f} vs LSTM: {lstm_test_r2:.4f}")
        print("\n   Suggestions for improvement:")
        print("   1. Increase β (distillation weight) to reduce discretization tax")
        print("   2. Try different lag values (sweep Δ from -5 to +5)")
        print("   3. Add dequant repair MLP after RVQ")
        print("   4. Increase codebook size or number of RVQ layers")
    
    # Save results
    results = {
        'config': cfg.__dict__,
        'teacher': {
            'val_r2': teacher_r2,
            'test_r2': teacher_metrics['r2'],
            'r2_vx': teacher_metrics['r2_vx'],
            'r2_vy': teacher_metrics['r2_vy'],
        },
        'student': {
            'val_r2': student_r2,
            'test_r2': student_metrics['r2'],
            'r2_vx': student_metrics['r2_vx'],
            'r2_vy': student_metrics['r2_vy'],
            'codes_per_layer': student_metrics['codes_per_layer'],
            'z_distance': student_metrics['z_distance'],
        },
        'lstm': {
            'val_r2': lstm_val_r2,
            'test_r2': lstm_test_r2,
        },
        'discretization_tax': discretization_tax,
        'beat_lstm': beat_lstm,
        'elapsed_time': elapsed,
    }
    
    results_path = RESULTS_DIR / 'exp18_ladr_vq_v2_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*70)
    print("Comparison with previous experiments:")
    print("  • Exp 10 (Deep Causal, no VQ):  R² = 0.7727")
    print("  • Exp 11 (Gumbel VQ):           R² = 0.7127")
    print("  • Exp 12 (RVQ-4):               R² = 0.776")
    print(f"  • Exp 18 (LADR-VQ):             R² = {student_metrics['r2']:.4f}")
    print("  • LSTM Baseline:                R² = 0.78")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_experiment()
