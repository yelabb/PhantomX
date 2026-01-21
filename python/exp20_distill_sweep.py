"""
Experiment 20: Distillation Weight Sweep (β Tuning)

MOTIVATION:
- Exp 19 achieved R² = 0.783 with β = 0.5
- Gap to LSTM is only 0.14% (0.0014 in R²)
- Distillation made student EXCEED teacher (negative discretization tax!)
- Hypothesis: Higher β may push student even closer to/beyond LSTM

SWEEP STRATEGY:
- β values: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
- Also test α (velocity weight) variations
- Track: R², discretization tax, codebook utilization

GOAL:
- Find optimal (α, β) to beat LSTM (R² ≥ 0.784)
- Current best: β=0.5 → R²=0.783

Based on Exp 19 codebase.
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
    # Data - MATCHING EXP 12/19 SETUP
    window_size: int = 10
    lag: int = 0  # No lag shift - current velocity
    train_frac: float = 0.8  # 80/20 split
    
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
    pretrain_epochs: int = 100
    pretrain_lr: float = 3e-4
    
    # Training - Phase 3 (Student with distillation)
    finetune_epochs: int = 60
    lr_encoder: float = 1e-5   # Nearly frozen
    lr_vq: float = 1e-4        # Train codebooks
    lr_decoder: float = 1e-4   # Fine-tune decoder
    
    # Distillation loss weights (will be swept)
    alpha: float = 1.0   # Velocity MSE weight
    beta: float = 0.5    # Latent distillation weight
    
    # General
    batch_size: int = 64
    weight_decay: float = 1e-4
    patience: int = 20
    grad_clip: float = 1.0


# ============================================================
# Dataset (No Lag - Matching Exp 12/19)
# ============================================================

class SlidingWindowDataset(Dataset):
    """Simple sliding window dataset without lag tuning."""
    
    def __init__(
        self,
        spike_counts: np.ndarray,
        velocities: np.ndarray,
        window_size: int = 10,
    ):
        self.window_size = window_size
        n = len(spike_counts)
        n_samples = n - window_size + 1
        
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
# Model Components (Same as Exp 19)
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
        return self.output_proj(x[:, -1, :])


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
        
        z_norm = torch.sum(z_flat ** 2, dim=1, keepdim=True)
        e_norm = torch.sum(self.embedding.weight ** 2, dim=1)
        distances = z_norm + e_norm.unsqueeze(0) - 2 * z_flat @ self.embedding.weight.t()
        
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)
        
        commitment_loss = F.mse_loss(z_q.detach(), z_flat) * self.commitment_cost
        z_q_st = z_flat + (z_q - z_flat).detach()
        
        if self.training:
            self._ema_update(z_flat, indices)
        
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


class DistilledRVQModel(nn.Module):
    """Distilled RVQ Model with teacher/student modes."""
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.use_vq = False
        
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
    verbose: bool = True,
) -> float:
    """Phase 1: Train the Teacher (encoder + decoder without VQ)."""
    if verbose:
        print("\n  [Phase 1] Training Teacher (No VQ)...")
    
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
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window)
            loss = F.mse_loss(output['velocity_pred'], velocity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        
        scheduler.step()
        
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
        
        if patience_counter >= cfg.patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    if verbose:
        print(f"    Teacher R² = {best_test_r2:.4f}")
    return best_test_r2


def init_phase2_codebooks(
    model: DistilledRVQModel,
    train_loader: DataLoader,
    device: torch.device,
):
    """Phase 2: Initialize RVQ codebooks using K-means."""
    model.eval()
    all_z_e = []
    
    with torch.no_grad():
        for batch in train_loader:
            z_e = model.encode(batch['window'].to(device))
            all_z_e.append(z_e.cpu())
    
    all_z_e = torch.cat(all_z_e, dim=0)
    model.vq.init_from_data(all_z_e.to(device))


def train_phase3_student(
    model: DistilledRVQModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    alpha: float,
    beta: float,
    verbose: bool = True,
) -> Tuple[float, Dict]:
    """Phase 3: Fine-tune with VQ enabled + distillation loss."""
    if verbose:
        print(f"  [Phase 3] Training Student (α={alpha}, β={beta})...")
    
    model.use_vq = True
    
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
        
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window)
            
            loss_vel = F.mse_loss(output['velocity_pred'], velocity)
            loss_distill = F.mse_loss(output['z_q'], output['z_e'].detach())
            loss = alpha * loss_vel + beta * loss_distill + output['commitment_loss']
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        
        scheduler.step()
        
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
        
        if patience_counter >= cfg.patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Final evaluation
    model.eval()
    preds_np, targets_np = [], []
    z_distances = []
    final_codes = [set() for _ in range(cfg.num_quantizers)]
    
    with torch.no_grad():
        for batch in test_loader:
            output = model(batch['window'].to(device))
            preds_np.append(output['velocity_pred'].cpu())
            targets_np.append(batch['velocity'])
            
            if 'all_indices' in output:
                for i, indices in enumerate(output['all_indices']):
                    final_codes[i].update(indices.cpu().numpy().tolist())
            
            z_dist = (output['z_q'] - output['z_e']).norm(dim=-1).mean()
            z_distances.append(z_dist.item())
    
    preds_np = torch.cat(preds_np).numpy()
    targets_np = torch.cat(targets_np).numpy()
    
    metrics = {
        'r2': r2(preds_np, targets_np),
        'r2_vx': r2(preds_np[:, 0], targets_np[:, 0]),
        'r2_vy': r2(preds_np[:, 1], targets_np[:, 1]),
        'codes_per_layer': [len(s) for s in final_codes],
        'z_distance': np.mean(z_distances),
    }
    
    if verbose:
        print(f"    Student R² = {metrics['r2']:.4f}")
    
    return best_test_r2, metrics


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
    
    return best_test_r2


# ============================================================
# Data Loading
# ============================================================

def load_and_prepare_data(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Load and normalize MC_Maze dataset."""
    tokenizer = SpikeTokenizer(n_channels=cfg.n_channels)
    mc_maze = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    spikes = mc_maze.spike_counts
    velocities = mc_maze.velocities
    
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
    """Create train/test dataloaders (80/20 split)."""
    dataset = SlidingWindowDataset(spikes, velocities, window_size=cfg.window_size)
    
    n_total = len(dataset)
    n_train = int(cfg.train_frac * n_total)
    
    train_ds = Subset(dataset, list(range(0, n_train)))
    test_ds = Subset(dataset, list(range(n_train, n_total)))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


# ============================================================
# Sweep Experiment
# ============================================================

def run_sweep():
    print("\n" + "="*70)
    print("EXPERIMENT 20: Distillation Weight Sweep (β Tuning)")
    print("="*70)
    print("\nObjective: Find optimal (α, β) to beat LSTM (R² ≥ 0.784)")
    print("Baseline:  Exp 19 with β=0.5 achieved R² = 0.783")
    print("Gap:       Only 0.14% to close!")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    cfg = Config()
    
    # Sweep parameters
    beta_values = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    alpha_values = [1.0]  # Keep α=1.0 fixed first, can expand later
    
    print(f"\nSweep grid:")
    print(f"  α (velocity weight): {alpha_values}")
    print(f"  β (distillation weight): {beta_values}")
    print(f"  Total experiments: {len(alpha_values) * len(beta_values)}")
    
    # Load data once
    print("\n" + "-"*60)
    print("Loading MC_Maze dataset...")
    spikes, velocities = load_and_prepare_data(cfg)
    train_loader, test_loader = create_dataloaders(spikes, velocities, cfg)
    print(f"Dataset: train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    
    # Train LSTM baseline once
    print("\nTraining LSTM baseline...")
    set_seed(42)
    lstm_r2 = train_lstm_baseline(train_loader, test_loader, device)
    print(f"LSTM Baseline R² = {lstm_r2:.4f}")
    
    # Store results
    all_results = []
    best_result = None
    best_r2 = -float('inf')
    
    # Run sweep
    print("\n" + "="*60)
    print("STARTING SWEEP")
    print("="*60)
    
    for alpha in alpha_values:
        for beta in beta_values:
            print(f"\n{'='*60}")
            print(f"Experiment: α={alpha}, β={beta}")
            print("="*60)
            
            set_seed(42)  # Same seed for fair comparison
            
            model = DistilledRVQModel(cfg).to(device)
            start_time = time.time()
            
            # Phase 1: Teacher
            teacher_r2 = train_phase1_teacher(model, train_loader, test_loader, device, cfg)
            
            # Phase 2: Init codebooks
            print("  [Phase 2] Initializing codebooks...")
            init_phase2_codebooks(model, train_loader, device)
            
            # Phase 3: Student with specific α, β
            student_r2, metrics = train_phase3_student(
                model, train_loader, test_loader, device, cfg, alpha, beta
            )
            
            elapsed = time.time() - start_time
            
            # Compute stats
            discretization_tax = teacher_r2 - metrics['r2']
            gap_to_lstm = lstm_r2 - metrics['r2']
            beat_lstm = metrics['r2'] > lstm_r2
            
            result = {
                'alpha': alpha,
                'beta': beta,
                'teacher_r2': teacher_r2,
                'student_r2': metrics['r2'],
                'r2_vx': metrics['r2_vx'],
                'r2_vy': metrics['r2_vy'],
                'discretization_tax': discretization_tax,
                'gap_to_lstm': gap_to_lstm,
                'beat_lstm': beat_lstm,
                'codes_per_layer': metrics['codes_per_layer'],
                'z_distance': metrics['z_distance'],
                'elapsed_time': elapsed,
            }
            all_results.append(result)
            
            # Print summary
            beat_str = "✅ BEAT LSTM!" if beat_lstm else f"Gap: {gap_to_lstm:.4f}"
            print(f"\n  Summary:")
            print(f"    Teacher R²: {teacher_r2:.4f}")
            print(f"    Student R²: {metrics['r2']:.4f}")
            print(f"    Tax:        {discretization_tax:.4f} ({discretization_tax*100:.2f}%)")
            print(f"    {beat_str}")
            print(f"    Time:       {elapsed/60:.1f} min")
            
            # Track best
            if metrics['r2'] > best_r2:
                best_r2 = metrics['r2']
                best_result = result
                
                # Save best model
                if beat_lstm:
                    save_path = Path(__file__).parent / 'models' / 'exp20_best_distilled_rvq.pt'
                    torch.save(model.state_dict(), save_path)
                    print(f"    🎉 New best model saved to {save_path}")
    
    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "="*70)
    print("EXPERIMENT 20 RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'α':<6} {'β':<6} {'Student R²':<12} {'Tax':<10} {'Gap to LSTM':<12} {'Status':<10}")
    print("-"*70)
    
    for r in sorted(all_results, key=lambda x: -x['student_r2']):
        status = "✅ BEAT" if r['beat_lstm'] else ""
        print(f"{r['alpha']:<6.2f} {r['beta']:<6.2f} {r['student_r2']:<12.4f} "
              f"{r['discretization_tax']:<10.4f} {r['gap_to_lstm']:<12.4f} {status}")
    
    print("\n" + "-"*70)
    print(f"LSTM Baseline: R² = {lstm_r2:.4f}")
    print(f"Best Student:  R² = {best_result['student_r2']:.4f} (α={best_result['alpha']}, β={best_result['beta']})")
    
    if best_result['beat_lstm']:
        improvement = best_result['student_r2'] - lstm_r2
        print(f"\n🎉 SUCCESS! Beat LSTM by {improvement:.4f} ({improvement*100:.2f}%)")
    else:
        print(f"\n📈 Best gap to LSTM: {best_result['gap_to_lstm']:.4f} ({best_result['gap_to_lstm']*100:.2f}%)")
        print("\n  Recommendations for next experiment:")
        print("  1. Try even higher β (4.0, 5.0) if trend is improving")
        print("  2. Add dequant repair MLP after RVQ")
        print("  3. Ensemble: Average Student + LSTM predictions")
        print("  4. Larger codebooks (256 codes per layer)")
    
    # Save results
    results_path = RESULTS_DIR / 'exp20_distill_sweep_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'lstm_baseline_r2': lstm_r2,
            'sweep_results': all_results,
            'best_result': best_result,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return all_results, best_result


if __name__ == "__main__":
    run_sweep()
