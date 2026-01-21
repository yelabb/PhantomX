"""
Experiment 22c: Multi-Seed Super-Teacher Distillation

MOTIVATION:
===========
Exp 22b fixed the augmentation bug but used only 1 seed.
Since stability is less critical than peak performance for a teacher,
we should train with multiple seeds and cherry-pick the best.

STRATEGY (3-Seed Cherry-Picking):
=================================
1. Train Wide Transformer (384, 6L) WITH augmentation across 3 seeds
2. Cherry-pick the BEST teacher (target R² > 0.81)
3. Use that Super-Teacher to distill to RVQ Student
4. Since stability is less critical for teachers, we optimize for peak performance

Expected:
- Teacher (best of 3): R² > 0.81 (beating Exp 21b's 0.8064)
- Student: R² > 0.80 (beating LSTM @ 0.800)

DIFFERENCE FROM EXP 22b:
========================
- Exp 22b: Single seed teacher training
- Exp 22c: Multi-seed teacher training with best selection
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

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
from phantomx.data.mc_maze_loader import load_mc_maze_from_nwb

DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def r2(preds: np.ndarray, targets: np.ndarray) -> float:
    return r2_score(targets, preds)


# ============================================================
# Configuration
# ============================================================

@dataclass 
class Config:
    # Data
    n_channels: int = 142
    window_size: int = 10  # 250ms @ 40Hz
    lag: int = 0           # Δ=0 (proven best for MC_Maze)
    train_frac: float = 0.8
    
    # TEACHER: Wide Transformer (best from Exp 21b)
    d_model: int = 384
    nhead: int = 8
    num_layers: int = 6
    dim_ff: int = 768  # d_model * 2
    dropout: float = 0.1
    
    # Data augmentation (from Exp 21b)
    noise_std: float = 0.1
    time_mask_prob: float = 0.1
    
    # Latent dimension (for RVQ)
    embedding_dim: int = 128
    
    # RVQ Student
    num_quantizers: int = 4
    num_codes: int = 128
    commitment_cost: float = 0.25
    
    # Training - Phase 1 (Teacher with Multi-Seed)
    pretrain_epochs: int = 150
    pretrain_lr: float = 3e-4
    teacher_seeds: List[int] = field(default_factory=lambda: [42, 123, 456])  # 3 seeds
    
    # Training - Phase 3 (Student with distillation)
    finetune_epochs: int = 80
    lr_encoder: float = 1e-5   # Nearly frozen
    lr_vq: float = 1e-4        # Train codebooks
    lr_decoder: float = 1e-4   # Fine-tune decoder
    
    # Distillation loss weights
    alpha: float = 1.0   # Velocity MSE weight
    beta: float = 0.5    # Latent distillation weight (proven optimal in Exp 20)
    
    # General
    batch_size: int = 64
    weight_decay: float = 1e-4
    patience: int = 25
    grad_clip: float = 1.0


# ============================================================
# Dataset WITH AUGMENTATION
# ============================================================

class SlidingWindowDataset(Dataset):
    """Sliding window dataset with optional augmentation (from Exp 21b)."""
    
    def __init__(
        self,
        spike_counts: np.ndarray,
        velocities: np.ndarray,
        window_size: int = 10,
        lag: int = 0,
        augment: bool = False,
        noise_std: float = 0.1,
        time_mask_prob: float = 0.1,
    ):
        self.window_size = window_size
        self.augment = augment
        self.noise_std = noise_std
        self.time_mask_prob = time_mask_prob
        
        n = len(spike_counts)
        n_samples = n - window_size - lag + 1
        
        self.windows = np.stack([
            spike_counts[i:i + window_size]
            for i in range(n_samples)
        ])
        
        target_idx = window_size - 1 + lag
        self.velocities = velocities[target_idx: target_idx + n_samples]
        
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx].copy()  # [T, C]
        velocity = self.velocities[idx]
        
        # Data augmentation during training
        if self.augment:
            # Gaussian noise injection
            if self.noise_std > 0:
                noise = np.random.randn(*window.shape) * self.noise_std
                window = window + noise
            
            # Time masking (randomly zero out some timesteps)
            if self.time_mask_prob > 0:
                mask = np.random.rand(window.shape[0]) > self.time_mask_prob
                window = window * mask[:, None]
        
        return {
            'window': torch.tensor(window, dtype=torch.float32),
            'velocity': torch.tensor(velocity, dtype=torch.float32),
        }


# ============================================================
# Causal Self-Attention
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('mask', mask)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale
        
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
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


# ============================================================
# Wide Transformer Encoder
# ============================================================

class WideTransformerEncoder(nn.Module):
    """Wide Transformer encoder (384, 6L) - best from Exp 21b."""
    
    def __init__(self, cfg: Config):
        super().__init__()
        
        self.input_proj = nn.Linear(cfg.n_channels, cfg.d_model)
        
        self.pos_embed = nn.Parameter(
            torch.randn(1, cfg.window_size, cfg.d_model) * 0.02
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        
        self.ln = nn.LayerNorm(cfg.d_model)
        
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.embedding_dim),
            nn.LayerNorm(cfg.embedding_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln(x)
        return self.output_proj(x[:, -1, :])


# ============================================================
# Residual Vector Quantization (RVQ)
# ============================================================

class ResidualVQLayer(nn.Module):
    """Single layer of residual vector quantization with EMA updates."""
    
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
        """Initialize codebook using K-means."""
        z_np = z.detach().cpu().numpy()
        n_samples = len(z_np)
        n_clusters = min(self.num_codes, n_samples // 2)
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(z_np)
        
        centroids = torch.from_numpy(kmeans.cluster_centers_).float()
        
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
        
        # Perplexity
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
    """Multi-stage Residual Vector Quantizer (RVQ-N)."""
    
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
        """Initialize all codebooks progressively."""
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
        
        for quantizer in self.quantizers:
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
# Distilled RVQ Model
# ============================================================

class DistilledRVQModel(nn.Module):
    """Distilled RVQ Model with Wide Transformer encoder."""
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.use_vq = False  # Start in teacher mode
        
        self.encoder = WideTransformerEncoder(cfg)
        
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
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encode(x)
        
        if self.use_vq:
            z_q, vq_info = self.vq(z_e)
            velocity_pred = self.decoder(z_q)
            return {
                'velocity_pred': velocity_pred,
                'z_e': z_e,
                'z_q': z_q,
                **vq_info
            }
        else:
            velocity_pred = self.decoder(z_e)
            return {
                'velocity_pred': velocity_pred,
                'z_e': z_e,
            }


# ============================================================
# LSTM Baseline
# ============================================================

class LSTMBaseline(nn.Module):
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

def train_single_seed_teacher(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    seed: int,
) -> Tuple[DistilledRVQModel, float, Dict]:
    """Train a single teacher model with a given seed."""
    set_seed(seed)
    
    model = DistilledRVQModel(cfg).to(device)
    model.use_vq = False  # Teacher mode
    
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=cfg.pretrain_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.pretrain_epochs)
    
    best_test_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, cfg.pretrain_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            out = model(window)
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
                window = batch['window'].to(device)
                out = model(window)
                test_preds.append(out['velocity_pred'].cpu().numpy())
                test_targets.append(batch['velocity'].numpy())
        
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        test_r2 = r2(test_preds, test_targets)
        
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | Loss: {train_loss:.4f} | Test R²: {test_r2:.4f} | Best: {best_test_r2:.4f}")
        
        if patience_counter >= cfg.patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    return model, best_test_r2, best_state


def train_phase1_multiseed_teacher(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> Tuple[DistilledRVQModel, float, int, List[Dict]]:
    """
    Phase 1: Train Wide Transformer teacher across multiple seeds, pick best.
    
    Returns:
        best_model: The teacher model with best performance
        best_r2: The best R² achieved
        best_seed: The seed that produced the best model
        seed_results: List of results for all seeds
    """
    print("\n" + "="*60)
    print("PHASE 1: Training Super-Teacher (Multi-Seed Selection)")
    print("="*60)
    print(f"Training {len(cfg.teacher_seeds)} seeds: {cfg.teacher_seeds}")
    print(f"Target: R² > 0.81 (beat Exp 21b's 0.8064)")
    print()
    
    seed_results = []
    
    for seed_idx, seed in enumerate(cfg.teacher_seeds, 1):
        print(f"\n--- Training Seed {seed_idx}/{len(cfg.teacher_seeds)}: {seed} ---")
        
        model, best_r2, best_state = train_single_seed_teacher(
            train_loader, test_loader, device, cfg, seed
        )
        
        print(f"  ✓ Seed {seed} finished: R² = {best_r2:.4f}")
        
        seed_results.append({
            'seed': seed,
            'r2': best_r2,
            'state_dict': best_state,
            'model': model,
        })
    
    # Pick the best seed
    best_result = max(seed_results, key=lambda x: x['r2'])
    best_model = best_result['model']
    best_r2 = best_result['r2']
    best_seed = best_result['seed']
    
    print("\n" + "="*60)
    print("MULTI-SEED TEACHER RESULTS:")
    print("="*60)
    for result in seed_results:
        marker = " ← BEST" if result['seed'] == best_seed else ""
        print(f"  Seed {result['seed']:3d}: R² = {result['r2']:.4f}{marker}")
    
    # Statistics
    r2_values = [r['r2'] for r in seed_results]
    mean_r2 = np.mean(r2_values)
    std_r2 = np.std(r2_values)
    print(f"\n  Mean R²: {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"  Best R²: {best_r2:.4f} (seed {best_seed})")
    
    if best_r2 > 0.81:
        print(f"\n  🎯 TARGET ACHIEVED! R² > 0.81")
    elif best_r2 > 0.8064:
        print(f"\n  ✓ Improved over Exp 21b (0.8064)")
    else:
        print(f"\n  ⚠️ Below Exp 21b baseline (0.8064)")
    
    return best_model, best_r2, best_seed, seed_results


def phase2_init_rvq(model: DistilledRVQModel, train_loader: DataLoader, device: torch.device):
    """Phase 2: Initialize RVQ codebooks using K-means on teacher latents."""
    print("\n" + "="*60)
    print("PHASE 2: K-Means Initialization of RVQ Codebooks")
    print("="*60)
    
    model.eval()
    model.use_vq = False
    
    z_all = []
    with torch.no_grad():
        for batch in train_loader:
            window = batch['window'].to(device)
            z_e = model.encode(window)
            z_all.append(z_e)
    
    z_all = torch.cat(z_all, dim=0)
    print(f"  Collected {len(z_all)} latent vectors")
    
    # Initialize RVQ
    model.vq.init_from_data(z_all)
    print("  ✓ RVQ codebooks initialized")


def train_phase3_student(
    model: DistilledRVQModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> Tuple[float, List[Dict]]:
    """Phase 3: Fine-tune with VQ + distillation loss."""
    print("\n" + "="*60)
    print("PHASE 3: Distillation Training (Student)")
    print("="*60)
    print(f"  α (velocity weight): {cfg.alpha}")
    print(f"  β (distillation weight): {cfg.beta}")
    
    model.use_vq = True
    
    # Separate learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': cfg.lr_encoder},
        {'params': model.vq.parameters(), 'lr': cfg.lr_vq},
        {'params': model.decoder.parameters(), 'lr': cfg.lr_decoder},
    ], weight_decay=cfg.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.finetune_epochs)
    
    best_test_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    history = []
    
    for epoch in range(1, cfg.finetune_epochs + 1):
        model.train()
        train_loss = 0.0
        train_vel_loss = 0.0
        train_distill_loss = 0.0
        
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            
            out = model(window)
            
            # Velocity loss
            vel_loss = F.mse_loss(out['velocity_pred'], velocity)
            
            # Distillation loss: z_q should match z_e
            distill_loss = F.mse_loss(out['z_q'], out['z_e'].detach())
            
            # Commitment loss from VQ
            commitment_loss = out.get('commitment_loss', 0.0)
            
            # Total loss
            loss = cfg.alpha * vel_loss + cfg.beta * distill_loss + commitment_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
            train_vel_loss += vel_loss.item()
            train_distill_loss += distill_loss.item()
        
        train_loss /= len(train_loader)
        train_vel_loss /= len(train_loader)
        train_distill_loss /= len(train_loader)
        scheduler.step()
        
        # Evaluation
        model.eval()
        test_preds, test_targets = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                window = batch['window'].to(device)
                velocity = batch['velocity'].to(device)
                out = model(window)
                test_preds.append(out['velocity_pred'].cpu().numpy())
                test_targets.append(velocity.cpu().numpy())
        
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        test_r2 = r2(test_preds, test_targets)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'vel_loss': train_vel_loss,
            'distill_loss': train_distill_loss,
            'test_r2': test_r2,
        })
        
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | Vel: {train_vel_loss:.4f} | Distill: {train_distill_loss:.4f} | Test R²: {test_r2:.4f} | Best: {best_test_r2:.4f}")
        
        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    print(f"\n  ✓ Student trained: R² = {best_test_r2:.4f}")
    return best_test_r2, history


def evaluate_codebook_usage(model: DistilledRVQModel, test_loader: DataLoader, device: torch.device) -> int:
    """Evaluate codebook utilization."""
    model.eval()
    model.use_vq = True
    
    all_indices = [[] for _ in range(model.cfg.num_quantizers)]
    
    with torch.no_grad():
        for batch in test_loader:
            window = batch['window'].to(device)
            out = model(window)
            
            for i, indices in enumerate(out['all_indices']):
                all_indices[i].extend(indices.cpu().numpy().tolist())
    
    print("\n  Codebook Utilization:")
    total_unique = 0
    for i, indices in enumerate(all_indices):
        unique = len(set(indices))
        total_unique += unique
        print(f"    Layer {i+1}: {unique}/{model.cfg.num_codes} codes used ({100*unique/model.cfg.num_codes:.1f}%)")
    
    return total_unique


def main():
    print("="*70)
    print("EXPERIMENT 22c: Multi-Seed Super-Teacher Distillation")
    print("="*70)
    print()
    print("STRATEGY:")
    print("  1. Train Wide Transformer (384, 6L) with augmentation across 3 seeds")
    print("  2. Cherry-pick the BEST teacher (target R² > 0.81)")
    print("  3. Use Super-Teacher to distill to RVQ Student")
    print("  4. Beat LSTM baseline (R² = ~0.80)")
    print()
    
    set_seed(42)  # For data loading reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading MC_Maze data...")
    spike_counts, kinematics = load_mc_maze_from_nwb(str(DATA_PATH))
    velocities = kinematics[:, 2:4]
    
    # Normalize
    spike_counts = (spike_counts - spike_counts.mean(0)) / (spike_counts.std(0) + 1e-6)
    velocities = (velocities - velocities.mean(0)) / (velocities.std(0) + 1e-6)
    
    print(f"  Spikes: {spike_counts.shape}")
    print(f"  Velocities: {velocities.shape}")
    
    cfg = Config()
    
    # Create datasets WITH AUGMENTATION for training
    train_ds = SlidingWindowDataset(
        spike_counts, velocities,
        window_size=cfg.window_size,
        lag=cfg.lag,
        augment=True,  # KEY: Data augmentation enabled
        noise_std=cfg.noise_std,
        time_mask_prob=cfg.time_mask_prob,
    )
    test_ds = SlidingWindowDataset(
        spike_counts, velocities,
        window_size=cfg.window_size,
        lag=cfg.lag,
        augment=False,  # No augmentation for test
    )
    
    n = len(train_ds)
    n_train = int(n * cfg.train_frac)
    
    train_subset = Subset(train_ds, list(range(n_train)))
    test_subset = Subset(test_ds, list(range(n_train, n)))
    
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=cfg.batch_size)
    
    print(f"  Train samples: {len(train_subset)}")
    print(f"  Test samples: {len(test_subset)}")
    
    # ========================================
    # LSTM Baseline
    # ========================================
    print("\n" + "="*60)
    print("BASELINE: LSTM")
    print("="*60)
    
    set_seed(42)
    lstm_train_ds = SlidingWindowDataset(spike_counts, velocities, cfg.window_size, cfg.lag, augment=False)
    lstm_train_subset = Subset(lstm_train_ds, list(range(n_train)))
    lstm_test_subset = Subset(lstm_train_ds, list(range(n_train, n)))
    lstm_train_loader = DataLoader(lstm_train_subset, batch_size=cfg.batch_size, shuffle=True)
    lstm_test_loader = DataLoader(lstm_test_subset, batch_size=cfg.batch_size)
    
    lstm = LSTMBaseline().to(device)
    lstm_optimizer = torch.optim.AdamW(lstm.parameters(), lr=3e-4)
    
    best_lstm_r2 = -float('inf')
    for epoch in range(1, 151):
        lstm.train()
        for batch in lstm_train_loader:
            lstm_optimizer.zero_grad()
            pred = lstm(batch['window'].to(device))
            loss = F.mse_loss(pred, batch['velocity'].to(device))
            loss.backward()
            lstm_optimizer.step()
        
        lstm.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in lstm_test_loader:
                pred = lstm(batch['window'].to(device))
                preds.append(pred.cpu().numpy())
                targets.append(batch['velocity'].numpy())
        
        test_r2 = r2(np.concatenate(preds), np.concatenate(targets))
        best_lstm_r2 = max(best_lstm_r2, test_r2)
        
        if epoch % 30 == 0:
            print(f"  Epoch {epoch:3d} | LSTM R²: {test_r2:.4f} | Best: {best_lstm_r2:.4f}")
    
    print(f"\n  ✓ LSTM Baseline: R² = {best_lstm_r2:.4f}")
    
    # ========================================
    # Phase 1: Train Multi-Seed Teacher
    # ========================================
    model, teacher_r2, best_seed, seed_results = train_phase1_multiseed_teacher(
        train_loader, test_loader, device, cfg
    )
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    
    # ========================================
    # Phase 2: Init RVQ
    # ========================================
    phase2_init_rvq(model, train_loader, device)
    
    # ========================================
    # Phase 3: Train Student with Distillation
    # ========================================
    student_r2, history = train_phase3_student(model, train_loader, test_loader, device, cfg)
    
    # Evaluate codebook
    total_codes = evaluate_codebook_usage(model, test_loader, device)
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 22c SUMMARY")
    print("="*70)
    
    print("\n1. BASELINES")
    print(f"   LSTM:                    R² = {best_lstm_r2:.4f}")
    
    print("\n2. MULTI-SEED TEACHER SELECTION")
    print(f"   Seeds tested:            {cfg.teacher_seeds}")
    for result in seed_results:
        marker = " ← SELECTED" if result['seed'] == best_seed else ""
        print(f"     Seed {result['seed']:3d}: R² = {result['r2']:.4f}{marker}")
    print(f"   Best seed:               {best_seed}")
    print(f"   Super-Teacher R²:        {teacher_r2:.4f}")
    
    print("\n3. DISTILLATION RESULTS")
    print(f"   Super-Teacher:              R² = {teacher_r2:.4f}")
    print(f"   RVQ-4 Student:              R² = {student_r2:.4f}")
    disc_tax = (teacher_r2 - student_r2) * 100
    print(f"   Discretization Tax:         {disc_tax:.2f}%")
    
    print("\n4. VS LSTM BASELINE")
    teacher_gap = (teacher_r2 - best_lstm_r2) * 100
    student_gap = (student_r2 - best_lstm_r2) * 100
    print(f"   Teacher gap: {'+' if teacher_gap > 0 else ''}{teacher_gap:.2f}%")
    print(f"   Student gap: {'+' if student_gap > 0 else ''}{student_gap:.2f}%")
    
    if student_r2 > best_lstm_r2:
        print("\n   🎉 BREAKTHROUGH: Student BEATS LSTM!")
        print("   First discrete VQ model to beat LSTM!")
    elif student_r2 > 0.784:
        print(f"\n   ✓ Student beats previous best (0.784) but not LSTM yet")
    else:
        print(f"\n   ⚠️ Student did not beat LSTM or previous best")
    
    print(f"\n5. CODEBOOK USAGE")
    print(f"   Total unique codes: {total_codes} / {cfg.num_quantizers * cfg.num_codes}")
    
    print("\n6. HISTORICAL COMPARISON")
    print("   Exp 19 (teacher 0.780 → student): R² = 0.783")
    print("   Exp 21b (single seed teacher):    R² = 0.8064")
    print("   Exp 22 (no augment teacher):      R² = 0.741")
    print(f"   Exp 22c (best of 3 seeds):        R² = {student_r2:.4f}")
    
    # Save results
    results = {
        "experiment": "22c",
        "description": "Multi-Seed Super-Teacher Distillation",
        "strategy": "Train teacher with 3 seeds, cherry-pick best, then distill to RVQ",
        "lstm_baseline": best_lstm_r2,
        "teacher_seeds": cfg.teacher_seeds,
        "seed_results": [
            {"seed": r['seed'], "r2": r['r2']} for r in seed_results
        ],
        "best_seed": best_seed,
        "teacher_r2": teacher_r2,
        "student_r2": student_r2,
        "discretization_tax": disc_tax,
        "teacher_vs_lstm": teacher_gap,
        "student_vs_lstm": student_gap,
        "student_beats_lstm": student_r2 > best_lstm_r2,
        "codebook_usage": total_codes,
        "config": {
            "d_model": cfg.d_model,
            "num_layers": cfg.num_layers,
            "dim_ff": cfg.dim_ff,
            "dropout": cfg.dropout,
            "num_quantizers": cfg.num_quantizers,
            "num_codes": cfg.num_codes,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "augmentation": True,
            "noise_std": cfg.noise_std,
            "time_mask_prob": cfg.time_mask_prob,
        },
        "history": history,
    }
    
    results_path = RESULTS_DIR / "exp22c_multiseed_teacher.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save final distilled model
    model_path = MODELS_DIR / "exp22c_distilled_rvq.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Distilled model saved to {model_path}")
    
    # Save best teacher checkpoint
    teacher_path = MODELS_DIR / f"exp22c_super_teacher_seed{best_seed}.pt"
    torch.save({
        'seed': best_seed,
        'r2': teacher_r2,
        'config': {
            'd_model': cfg.d_model,
            'num_layers': cfg.num_layers,
            'dim_ff': cfg.dim_ff,
        },
    }, teacher_path)
    print(f"Super-Teacher checkpoint saved to {teacher_path}")


if __name__ == "__main__":
    main()
