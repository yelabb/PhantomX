"""
Experiment 22: Distill Wide Transformer to RVQ

MOTIVATION:
===========
Exp 21b achieved a breakthrough: Wide Transformer (384, 6L) beat LSTM!
- Wide Transformer: R² = 0.8064
- LSTM Baseline:    R² = 0.8009

Now we need to distill this Super-Teacher to a discrete RVQ model.
If successful, this would be the FIRST discrete VQ model to beat LSTM.

STRATEGY:
=========
1. Use Wide Transformer (384, 6L) as teacher
2. Distill to RVQ-4 student using Exp 19 methodology
3. Key: Teacher is now 2.6% better than before (0.806 vs 0.784)

HYPOTHESIS:
===========
With a stronger teacher, the distilled student should:
- Exceed previous best discrete model (0.784)
- Potentially beat LSTM (0.800)

Expected improvement:
- Exp 19: Teacher 0.780 → Student 0.783 (surpassed teacher!)
- Exp 22: Teacher 0.806 → Student ~0.81? (if same pattern holds)

THREE-PHASE TRAINING:
  Phase 1: Train Wide Transformer teacher (or load from Exp 21b)
  Phase 2: K-Means init RVQ codebooks on teacher latents
  Phase 3: Fine-tune with VQ + distillation loss
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
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
    dim_ff: int = 768
    dropout: float = 0.1
    
    # Latent dimension (for RVQ)
    embedding_dim: int = 128
    
    # RVQ Student
    num_quantizers: int = 4
    num_codes: int = 128
    commitment_cost: float = 0.25
    
    # Training - Phase 1 (Teacher)
    pretrain_epochs: int = 150
    pretrain_lr: float = 3e-4
    
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
# Dataset
# ============================================================

class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        spike_counts: np.ndarray,
        velocities: np.ndarray,
        window_size: int = 10,
        lag: int = 0,
    ):
        self.window_size = window_size
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
        return {
            "window": torch.tensor(self.windows[idx], dtype=torch.float32),
            "velocity": torch.tensor(self.velocities[idx], dtype=torch.float32),
        }


# ============================================================
# Wide Transformer Encoder (from Exp 21b)
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
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


class WideTransformerEncoder(nn.Module):
    """Wide Transformer Encoder (384, 6L) — best from Exp 21b."""
    
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
    """
    Distilled RVQ Model with Wide Transformer encoder.
    
    Modes:
    - Teacher mode (use_vq=False): z_e → decoder
    - Student mode (use_vq=True): z_e → RVQ → z_q → decoder
    """
    
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
    model: DistilledRVQModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> float:
    """Phase 1: Train the Wide Transformer Teacher."""
    print("\n" + "="*60)
    print("PHASE 1: Training Wide Transformer Teacher")
    print("="*60)
    
    model.use_vq = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
                velocity = batch['velocity'].to(device)
                out = model(window)
                test_preds.append(out['velocity_pred'].cpu().numpy())
                test_targets.append(velocity.cpu().numpy())
        
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
            print(f"  Epoch {epoch:3d} | Loss: {train_loss:.4f} | Test R²: {test_r2:.4f} | Best: {best_test_r2:.4f}")
        
        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    print(f"\n  ✓ Teacher trained: R² = {best_test_r2:.4f}")
    return best_test_r2


def phase2_init_rvq(
    model: DistilledRVQModel,
    train_loader: DataLoader,
    device: torch.device,
):
    """Phase 2: Initialize RVQ codebooks with K-means."""
    print("\n" + "="*60)
    print("PHASE 2: K-Means Initialization of RVQ Codebooks")
    print("="*60)
    
    model.eval()
    model.use_vq = False
    
    # Collect all encoder outputs
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
    return best_test_r2, history


def evaluate_codebook_usage(model: DistilledRVQModel, test_loader: DataLoader, device: torch.device):
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
    print("EXPERIMENT 22: Distill Wide Transformer to RVQ")
    print("="*70)
    print()
    print("Goal: First discrete VQ model to beat LSTM!")
    print("Strategy: Distill Wide Transformer (R²=0.806) → RVQ Student")
    print()
    
    set_seed(42)
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
    
    # Create datasets
    cfg = Config()
    dataset = SlidingWindowDataset(spike_counts, velocities, cfg.window_size, cfg.lag)
    
    n = len(dataset)
    n_train = int(n * cfg.train_frac)
    train_dataset = Subset(dataset, list(range(n_train)))
    test_dataset = Subset(dataset, list(range(n_train, n)))
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # ========================================
    # LSTM Baseline
    # ========================================
    print("\n" + "="*60)
    print("BASELINE: LSTM")
    print("="*60)
    
    lstm = LSTMBaseline().to(device)
    lstm_optimizer = torch.optim.AdamW(lstm.parameters(), lr=3e-4)
    
    best_lstm_r2 = -float('inf')
    for epoch in range(1, 151):
        lstm.train()
        for batch in train_loader:
            lstm_optimizer.zero_grad()
            pred = lstm(batch['window'].to(device))
            loss = F.mse_loss(pred, batch['velocity'].to(device))
            loss.backward()
            lstm_optimizer.step()
        
        lstm.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                pred = lstm(batch['window'].to(device))
                preds.append(pred.cpu().numpy())
                targets.append(batch['velocity'].numpy())
        
        test_r2 = r2(np.concatenate(preds), np.concatenate(targets))
        best_lstm_r2 = max(best_lstm_r2, test_r2)
        
        if epoch % 30 == 0:
            print(f"  Epoch {epoch:3d} | LSTM R²: {test_r2:.4f} | Best: {best_lstm_r2:.4f}")
    
    print(f"\n  ✓ LSTM Baseline: R² = {best_lstm_r2:.4f}")
    
    # ========================================
    # Create and Train Distilled RVQ Model
    # ========================================
    model = DistilledRVQModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model parameters: {n_params:,}")
    
    # Phase 1: Train Teacher
    teacher_r2 = train_phase1_teacher(model, train_loader, test_loader, device, cfg)
    
    # Phase 2: Init RVQ
    phase2_init_rvq(model, train_loader, device)
    
    # Phase 3: Train Student with Distillation
    student_r2, history = train_phase3_student(model, train_loader, test_loader, device, cfg)
    
    # Evaluate codebook
    total_codes = evaluate_codebook_usage(model, test_loader, device)
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 22 SUMMARY")
    print("="*70)
    
    print("\n1. BASELINES")
    print(f"   LSTM:                    R² = {best_lstm_r2:.4f}")
    
    print("\n2. DISTILLATION RESULTS")
    print(f"   Wide Transformer (Teacher): R² = {teacher_r2:.4f}")
    print(f"   RVQ-4 Student:              R² = {student_r2:.4f}")
    
    # Discretization tax
    tax = (teacher_r2 - student_r2) / teacher_r2 * 100
    print(f"   Discretization Tax:         {tax:.2f}%")
    
    # Gap to LSTM
    student_gap = (student_r2 - best_lstm_r2) / best_lstm_r2 * 100
    teacher_gap = (teacher_r2 - best_lstm_r2) / best_lstm_r2 * 100
    
    print(f"\n3. VS LSTM BASELINE")
    print(f"   Teacher gap: {teacher_gap:+.2f}%")
    print(f"   Student gap: {student_gap:+.2f}%")
    
    if student_r2 > best_lstm_r2:
        print(f"\n   🎉 BREAKTHROUGH: Discrete RVQ model BEATS LSTM!")
        print(f"      Student R² = {student_r2:.4f} > LSTM R² = {best_lstm_r2:.4f}")
    elif student_r2 > 0.784:
        print(f"\n   ✓ New best discrete model! (beats Exp 19's 0.784)")
    else:
        print(f"\n   ⚠️ Student did not beat LSTM or previous best")
    
    print(f"\n4. CODEBOOK USAGE")
    print(f"   Total unique codes: {total_codes} / {cfg.num_quantizers * cfg.num_codes}")
    
    # Comparison to previous experiments
    print("\n5. HISTORICAL COMPARISON")
    print(f"   Exp 19 (old teacher 0.780 → student): R² = 0.783")
    print(f"   Exp 22 (new teacher {teacher_r2:.3f} → student): R² = {student_r2:.4f}")
    
    # Save results
    results = {
        "lstm_baseline": best_lstm_r2,
        "teacher_r2": teacher_r2,
        "student_r2": student_r2,
        "discretization_tax": tax,
        "vs_lstm": student_gap,
        "total_codes_used": total_codes,
        "config": {
            "d_model": cfg.d_model,
            "num_layers": cfg.num_layers,
            "num_quantizers": cfg.num_quantizers,
            "num_codes": cfg.num_codes,
            "beta": cfg.beta,
        },
    }
    
    results_path = RESULTS_DIR / "exp22_distill_wide_transformer.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save model
    model_path = MODELS_DIR / "exp22_distilled_rvq.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
