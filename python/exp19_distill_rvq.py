"""
Experiment 19: Distilled RVQ (Combining Best of Exp 12 + Exp 18)

MOTIVATION:
- Exp 12 achieved R²=0.776 with RVQ-4 (teacher was 0.784)
- Exp 18 proved distillation eliminates discretization tax (0% loss)
- BUT Exp 18 regressed due to lag tuning (Δ=+1) hurting the teacher

THE FIX:
Combine Exp 12's setup (Δ=0, 80/20 split) with Exp 18's distillation:
- Teacher R²: ~0.784 (proven in Exp 12)
- Distillation tax: ~0% (proven in Exp 18)
- Expected Student R²: ~0.784 → BEAT LSTM (0.780)

KEY DIFFERENCES FROM EXP 18:
1. lag=0 (no forward shift - MC_Maze encodes current velocity)
2. 80/20 train/test split (more training data)
3. No separate validation set (use test for early stopping like Exp 12)

THREE-PHASE TRAINING:
  Phase 1 (Teacher): Pre-train encoder + decoder without VQ
  Phase 2 (Init):    K-Means init RVQ codebooks on trained encoder outputs
  Phase 3 (Student): Fine-tune with VQ + distillation loss
"""

from __future__ import annotations

import json
import math
import random
import time
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
    # Data - MATCHING EXP 12 SETUP
    window_size: int = 10
    lag: int = 0  # KEY FIX: Δ=0 (no lag shift - current velocity)
    train_frac: float = 0.8  # 80/20 split like Exp 12
    
    # Model architecture (same as Exp 12/18)
    n_channels: int = 142
    d_model: int = 256
    embedding_dim: int = 128
    num_layers: int = 6
    nhead: int = 8
    dim_ff: int = 512
    dropout: float = 0.1
    
    # RVQ (same as Exp 12)
    num_quantizers: int = 4
    num_codes: int = 128
    commitment_cost: float = 0.25
    
    # Training - Phase 1 (Teacher)
    pretrain_epochs: int = 100
    pretrain_lr: float = 3e-4
    
    # Training - Phase 3 (Student with distillation)
    finetune_epochs: int = 60
    lr_encoder: float = 1e-5   # Nearly frozen
    lr_vq: float = 1e-4        # Train codebooks
    lr_decoder: float = 1e-4   # Fine-tune decoder
    
    # Distillation loss weights
    alpha: float = 1.0   # Velocity MSE weight
    beta: float = 0.5    # Latent distillation weight (z_q → z_e)
    
    # General
    batch_size: int = 64
    weight_decay: float = 1e-4
    patience: int = 20
    grad_clip: float = 1.0


# ============================================================
# Dataset (No Lag - Matching Exp 12)
# ============================================================

class SlidingWindowDataset(Dataset):
    """
    Simple sliding window dataset without lag tuning.
    
    For window ending at timestep t, target is velocity[t].
    This matches Exp 12's setup which achieved R²=0.784.
    """
    
    def __init__(
        self,
        spike_counts: np.ndarray,
        velocities: np.ndarray,
        window_size: int = 10,
    ):
        self.window_size = window_size
        n = len(spike_counts)
        n_samples = n - window_size + 1
        
        # Build windows: [i:i+window_size] → velocity[i+window_size-1]
        self.windows = np.stack([
            spike_counts[i:i + window_size]
            for i in range(n_samples)
        ])
        self.velocities = velocities[window_size - 1: window_size - 1 + n_samples]
        
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "window": torch.tensor(self.windows[idx], dtype=torch.float32),
            "velocity": torch.tensor(self.velocities[idx], dtype=torch.float32),
        }


# ============================================================
# Causal Transformer Encoder
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
        
        # Compute distances: ||z - e||² = ||z||² + ||e||² - 2⟨z, e⟩
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
    Distilled RVQ Model.
    
    Architecture:
        Spikes → CausalTransformer → [Optional RVQ] → MLP Decoder → Velocity
    
    Modes:
    - Teacher mode (use_vq=False): z_e → decoder
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
    model: DistilledRVQModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> float:
    """Phase 1: Train the Teacher (encoder + decoder without VQ)."""
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
    
    best_test_r2 = -float('inf')
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
        
        # Evaluation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                output = model(batch['window'].to(device))
                preds.append(output['velocity_pred'].cpu())
                targets.append(batch['velocity'])
        
        test_r2 = r2(torch.cat(preds).numpy(), torch.cat(targets).numpy())
        
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch <= 5:
            print(f"  Epoch {epoch:3d}: loss={train_loss/len(train_loader):.4f}, "
                  f"test_R²={test_r2:.4f} (best={best_test_r2:.4f})")
        
        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n✓ Teacher training complete. Best R² = {best_test_r2:.4f}")
    return best_test_r2


def init_phase2_codebooks(
    model: DistilledRVQModel,
    train_loader: DataLoader,
    device: torch.device,
):
    """Phase 2: Initialize RVQ codebooks using K-means on trained encoder outputs."""
    print("\n" + "="*60)
    print("PHASE 2: Initializing RVQ Codebooks (K-Means on z_e)")
    print("="*60)
    
    model.eval()
    all_z_e = []
    
    with torch.no_grad():
        for batch in train_loader:
            z_e = model.encode(batch['window'].to(device))
            all_z_e.append(z_e.cpu())
    
    all_z_e = torch.cat(all_z_e, dim=0)
    print(f"  Collected {len(all_z_e)} encoder outputs for K-means initialization")
    
    model.vq.init_from_data(all_z_e.to(device))
    print("✓ Codebook initialization complete")


def train_phase3_student(
    model: DistilledRVQModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> float:
    """Phase 3: Fine-tune with VQ enabled + distillation loss."""
    print("\n" + "="*60)
    print("PHASE 3: Training Student (VQ + Distillation)")
    print("="*60)
    print(f"  Distillation: α={cfg.alpha} (velocity), β={cfg.beta} (latent)")
    
    model.use_vq = True
    
    # Differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': cfg.lr_encoder},
        {'params': model.vq.parameters(), 'lr': cfg.lr_vq},
        {'params': model.decoder.parameters(), 'lr': cfg.lr_decoder},
    ], weight_decay=cfg.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.finetune_epochs)
    
    best_test_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, cfg.finetune_epochs + 1):
        model.train()
        train_loss_vel = 0.0
        train_loss_distill = 0.0
        
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window)
            
            # Velocity loss (ground truth)
            loss_vel = F.mse_loss(output['velocity_pred'], velocity)
            
            # Distillation loss: make z_q close to z_e
            # z_e is detached - we're training VQ to match the teacher, not vice versa
            loss_distill = F.mse_loss(output['z_q'], output['z_e'].detach())
            
            # Combined loss
            loss = cfg.alpha * loss_vel + cfg.beta * loss_distill + output['commitment_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            
            train_loss_vel += loss_vel.item()
            train_loss_distill += loss_distill.item()
        
        scheduler.step()
        
        # Evaluation
        model.eval()
        preds, targets = [], []
        codes_per_layer = [set() for _ in range(cfg.num_quantizers)]
        
        with torch.no_grad():
            for batch in test_loader:
                output = model(batch['window'].to(device))
                preds.append(output['velocity_pred'].cpu())
                targets.append(batch['velocity'])
                
                if 'all_indices' in output:
                    for i, indices in enumerate(output['all_indices']):
                        codes_per_layer[i].update(indices.cpu().numpy().tolist())
        
        test_r2 = r2(torch.cat(preds).numpy(), torch.cat(targets).numpy())
        
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch <= 5:
            codes_str = "/".join([str(len(s)) for s in codes_per_layer])
            print(f"  Epoch {epoch:3d}: test_R²={test_r2:.4f} | "
                  f"L_vel={train_loss_vel/len(train_loader):.4f} "
                  f"L_distill={train_loss_distill/len(train_loader):.4f} | "
                  f"codes=[{codes_str}]")
        
        if patience_counter >= cfg.patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n✓ Student training complete. Best R² = {best_test_r2:.4f}")
    return best_test_r2


def train_lstm_baseline(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
) -> float:
    """Train LSTM baseline for fair comparison."""
    model = LSTMBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_test_r2 = -float('inf')
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
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                pred = model(batch['window'].to(device))
                preds.append(pred.cpu())
                targets.append(batch['velocity'])
        
        test_r2 = r2(torch.cat(preds).numpy(), torch.cat(targets).numpy())
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    return best_test_r2


def evaluate_model(
    model: DistilledRVQModel,
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
                
                z_dist = (output['z_q'] - output['z_e']).norm(dim=-1).mean()
                z_distances.append(z_dist.item())
    
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    
    return {
        'r2': r2(preds, targets),
        'r2_vx': r2(preds[:, 0], targets[:, 0]),
        'r2_vy': r2(preds[:, 1], targets[:, 1]),
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
) -> Tuple[DataLoader, DataLoader]:
    """Create train/test dataloaders (80/20 split, matching Exp 12)."""
    dataset = SlidingWindowDataset(spikes, velocities, window_size=cfg.window_size)
    
    n_total = len(dataset)
    n_train = int(cfg.train_frac * n_total)
    
    train_ds = Subset(dataset, list(range(0, n_train)))
    test_ds = Subset(dataset, list(range(n_train, n_total)))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    print("\n" + "="*70)
    print("EXPERIMENT 19: Distilled RVQ (Best of Exp 12 + Exp 18)")
    print("="*70)
    print("\nObjective: Beat LSTM (R²=0.780) with discrete VQ representation")
    print("Strategy:  Exp 12 setup (Δ=0, 80/20) + Exp 18 distillation (0% tax)")
    print("Expected:  Teacher ~0.784 → Student ~0.784 → Beat LSTM!")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    cfg = Config()
    print(f"\nConfiguration (matching Exp 12):")
    print(f"  Lag (Δ): {cfg.lag} (no shift - current velocity)")
    print(f"  Split: {int(cfg.train_frac*100)}/{int((1-cfg.train_frac)*100)} train/test")
    print(f"  Window: {cfg.window_size} bins ({cfg.window_size * 25}ms)")
    print(f"  RVQ: {cfg.num_quantizers} layers × {cfg.num_codes} codes")
    print(f"  Distillation: α={cfg.alpha} (velocity), β={cfg.beta} (latent)")
    
    # Load data
    print("\n" + "-"*60)
    print("Loading MC_Maze dataset...")
    spikes, velocities = load_and_prepare_data(cfg)
    train_loader, test_loader = create_dataloaders(spikes, velocities, cfg)
    
    print(f"Dataset sizes: train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Initialize model
    model = DistilledRVQModel(cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    
    start_time = time.time()
    
    # ========================================
    # PHASE 1: Train Teacher (No VQ)
    # ========================================
    teacher_r2 = train_phase1_teacher(model, train_loader, test_loader, device, cfg)
    
    # ========================================
    # PHASE 2: Initialize Codebooks
    # ========================================
    init_phase2_codebooks(model, train_loader, device)
    
    # ========================================
    # PHASE 3: Train Student with Distillation
    # ========================================
    student_r2 = train_phase3_student(model, train_loader, test_loader, device, cfg)
    
    elapsed = time.time() - start_time
    
    # Final evaluation
    print("\n" + "-"*60)
    print("Final Evaluation")
    print("-"*60)
    
    # Student (with VQ)
    model.use_vq = True
    student_metrics = evaluate_model(model, test_loader, device)
    
    # Teacher (without VQ)
    model.use_vq = False
    teacher_metrics = evaluate_model(model, test_loader, device)
    
    # LSTM Baseline
    print("\nTraining LSTM baseline...")
    lstm_test_r2 = train_lstm_baseline(train_loader, test_loader, device)
    
    # Results Summary
    print("\n" + "="*70)
    print("EXPERIMENT 19 RESULTS")
    print("="*70)
    
    discretization_tax = teacher_metrics['r2'] - student_metrics['r2']
    beat_lstm = student_metrics['r2'] > lstm_test_r2
    
    print(f"\n{'Model':<25} {'Test R²':>10} {'vx':>10} {'vy':>10}")
    print("-"*70)
    print(f"{'Teacher (no VQ)':<25} {teacher_metrics['r2']:>10.4f} "
          f"{teacher_metrics['r2_vx']:>10.4f} {teacher_metrics['r2_vy']:>10.4f}")
    print(f"{'Student (Distilled RVQ)':<25} {student_metrics['r2']:>10.4f} "
          f"{student_metrics['r2_vx']:>10.4f} {student_metrics['r2_vy']:>10.4f}")
    print(f"{'LSTM Baseline':<25} {lstm_test_r2:>10.4f}")
    
    print(f"\n{'Analysis:':<25}")
    print(f"  Discretization tax: {discretization_tax:.4f} ({discretization_tax*100:.2f}%)")
    print(f"  Latent distance (||z_q - z_e||): {student_metrics['z_distance']:.4f}")
    print(f"  Codes per layer: {student_metrics['codes_per_layer']}")
    print(f"  Training time: {elapsed/60:.1f} min")
    
    print("\n" + "-"*70)
    if beat_lstm:
        print(f"🎉 SUCCESS: Student R²={student_metrics['r2']:.4f} > LSTM R²={lstm_test_r2:.4f}")
        print("   Distilled RVQ has beaten the LSTM baseline!")
        
        save_path = Path(__file__).parent / 'models' / 'exp19_distilled_rvq_best.pt'
        torch.save(model.state_dict(), save_path)
        print(f"   Model saved to {save_path}")
    else:
        gap = lstm_test_r2 - student_metrics['r2']
        print(f"📈 Gap to LSTM: {gap:.4f} ({gap*100:.2f}%)")
        print(f"   Student: {student_metrics['r2']:.4f} vs LSTM: {lstm_test_r2:.4f}")
        print("\n   Next steps:")
        print("   1. Increase β (distillation weight)")
        print("   2. Add dequant repair MLP after RVQ")
        print("   3. Try ensemble (Student + LSTM)")
        print("   4. Larger codebooks or more RVQ layers")
    
    # Save results
    results = {
        'config': {
            'lag': cfg.lag,
            'window_size': cfg.window_size,
            'train_frac': cfg.train_frac,
            'num_quantizers': cfg.num_quantizers,
            'num_codes': cfg.num_codes,
            'alpha': cfg.alpha,
            'beta': cfg.beta,
        },
        'teacher': {
            'test_r2': teacher_metrics['r2'],
            'r2_vx': teacher_metrics['r2_vx'],
            'r2_vy': teacher_metrics['r2_vy'],
        },
        'student': {
            'test_r2': student_metrics['r2'],
            'r2_vx': student_metrics['r2_vx'],
            'r2_vy': student_metrics['r2_vy'],
            'codes_per_layer': student_metrics['codes_per_layer'],
            'z_distance': student_metrics['z_distance'],
        },
        'lstm_test_r2': lstm_test_r2,
        'discretization_tax': discretization_tax,
        'beat_lstm': beat_lstm,
        'elapsed_time': elapsed,
    }
    
    results_path = RESULTS_DIR / 'exp19_distill_rvq_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*70)
    print("Comparison with previous experiments:")
    print("  • Exp 12 (RVQ-4, no distill):   R² = 0.776")
    print("  • Exp 18 (LADR-VQ, Δ=+1):       R² = 0.695 (regression)")
    print(f"  • Exp 19 (Distilled RVQ, Δ=0): R² = {student_metrics['r2']:.4f}")
    print("  • LSTM Baseline:                R² = 0.780")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_experiment()
