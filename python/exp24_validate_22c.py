"""
Experiment 24: Statistical Validation of Exp 22c (Multi-Seed Distillation)

MOTIVATION:
===========
Exp 22c trains a teacher with 3 seeds and cherry-picks the best, then distills to RVQ.
BUT: Cherry-picking introduces selection bias. To claim scientific validity, we need:

1. Statistical Significance: Run FULL pipeline (teacher → distill) with n=5 seeds
2. Fair Comparison: Compare distilled student vs LSTM with same conditions
3. Paired Statistical Tests: Wilcoxon signed-rank or paired t-test
4. Effect Size: Cohen's d to quantify practical significance
5. Verify the discretization tax is consistent across seeds

PROTOCOL:
=========
- Run full exp22c pipeline 5x with different random seeds
- Each run: train teacher → init RVQ → distill student
- Report: mean ± std, 95% CI, p-value, effect size
- Compare: Teacher R², Student R², Discretization Tax

EXPECTED OUTPUT:
================
If significant: "Distilled RVQ Student beats LSTM (p < 0.05, Cohen's d = X.XX)"
If not: "No significant difference (p = X.XX)"
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset, Subset
from scipy import stats

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent))
from phantomx.data.mc_maze_loader import load_mc_maze_from_nwb

DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Statistical Utilities
# ============================================================

def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for mean."""
    n = len(data)
    if n < 2:
        return float(data[0]), float(data[0])
    mean = np.mean(data)
    stderr = np.std(data, ddof=1) / np.sqrt(n)
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * stderr
    return mean - margin, mean + margin


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


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
# Configuration (from exp22c)
# ============================================================

@dataclass 
class Config:
    # Data
    n_channels: int = 142
    window_size: int = 10  # 250ms @ 40Hz
    lag: int = 0
    train_frac: float = 0.8
    
    # TEACHER: Wide Transformer (384, 6L)
    d_model: int = 384
    nhead: int = 8
    num_layers: int = 6
    dim_ff: int = 768
    dropout: float = 0.1
    
    # Data augmentation
    noise_std: float = 0.1
    time_mask_prob: float = 0.1
    
    # Latent dimension
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
    lr_encoder: float = 1e-5
    lr_vq: float = 1e-4
    lr_decoder: float = 1e-4
    
    # Distillation loss weights
    alpha: float = 1.0
    beta: float = 0.5
    
    # General
    batch_size: int = 64
    weight_decay: float = 1e-4
    patience: int = 25
    grad_clip: float = 1.0
    
    # Validation
    n_seeds: int = 5


# ============================================================
# Dataset
# ============================================================

class SlidingWindowDataset(Dataset):
    """Sliding window dataset with optional augmentation."""
    
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
        window = self.windows[idx].copy()
        velocity = self.velocities[idx]
        
        if self.augment:
            if self.noise_std > 0:
                noise = np.random.randn(*window.shape) * self.noise_std
                window = window + noise
            
            if self.time_mask_prob > 0:
                mask = np.random.random(window.shape[0]) > self.time_mask_prob
                window = window * mask[:, np.newaxis]
        
        return {
            "window": torch.tensor(window, dtype=torch.float32),
            "velocity": torch.tensor(velocity, dtype=torch.float32),
        }


# ============================================================
# Models (from exp22c)
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
    """Wide Transformer Encoder (384, 6L)."""
    
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


class VectorQuantizer(nn.Module):
    """Single VQ layer with EMA updates."""
    
    def __init__(self, num_codes: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.codebook = nn.Embedding(num_codes, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)
    
    def init_from_data(self, z: torch.Tensor):
        """Initialize codebook from data using K-means."""
        z_np = z.detach().cpu().numpy()
        n_samples = min(len(z_np), 10000)
        indices = np.random.choice(len(z_np), n_samples, replace=False)
        z_sample = z_np[indices]
        
        kmeans = KMeans(n_clusters=self.num_codes, n_init=10, random_state=42)
        kmeans.fit(z_sample)
        
        self.codebook.weight.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        ).to(self.codebook.weight.device)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        distances = torch.cdist(z, self.codebook.weight)
        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices)
        
        commitment_loss = F.mse_loss(z_q.detach(), z) * self.commitment_cost
        
        z_q = z + (z_q - z).detach()
        
        residual = z - z_q.detach()
        
        avg_probs = torch.bincount(indices.flatten(), minlength=self.num_codes).float()
        avg_probs = avg_probs / avg_probs.sum()
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, residual, {
            'indices': indices,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity,
        }


class ResidualVectorQuantizer(nn.Module):
    """Residual VQ with multiple layers."""
    
    def __init__(self, num_quantizers: int, num_codes: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(num_codes, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])
    
    def init_from_data(self, z_all: torch.Tensor):
        """Initialize all codebooks progressively."""
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
        
        for quantizer in self.quantizers:
            z_q_i, residual, info = quantizer(residual)
            z_q_sum = z_q_sum + z_q_i
            
            all_indices.append(info['indices'])
            total_commitment_loss = total_commitment_loss + info['commitment_loss']
            total_perplexity = total_perplexity + info['perplexity']
        
        return z_q_sum, {
            'indices': all_indices[0],
            'all_indices': all_indices,
            'commitment_loss': total_commitment_loss / self.num_quantizers,
            'perplexity': total_perplexity / self.num_quantizers,
        }


class DistilledRVQModel(nn.Module):
    """Distilled RVQ Model with Wide Transformer encoder."""
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.use_vq = False
        
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


class LSTMBaseline(nn.Module):
    """LSTM Baseline for comparison."""
    
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

def train_teacher(
    model: DistilledRVQModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> float:
    """Train teacher (Phase 1)."""
    model.use_vq = False
    
    optimizer = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=cfg.pretrain_lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.pretrain_epochs)
    
    best_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, cfg.pretrain_epochs + 1):
        model.train()
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            out = model(window)
            loss = F.mse_loss(out['velocity_pred'], velocity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        
        scheduler.step()
        
        # Eval
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch['window'].to(device))
                preds.append(out['velocity_pred'].cpu().numpy())
                targets.append(batch['velocity'].numpy())
        
        test_r2 = r2(np.concatenate(preds), np.concatenate(targets))
        
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.patience:
            break
    
    model.load_state_dict(best_state)
    return best_r2


def init_rvq(model: DistilledRVQModel, train_loader: DataLoader, device: torch.device):
    """Initialize RVQ codebooks (Phase 2)."""
    model.eval()
    z_list = []
    with torch.no_grad():
        for batch in train_loader:
            z_e = model.encode(batch['window'].to(device))
            z_list.append(z_e.cpu())
    
    z_all = torch.cat(z_list, dim=0).to(device)
    model.vq.init_from_data(z_all)


def train_student(
    model: DistilledRVQModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
) -> float:
    """Train student with distillation (Phase 3)."""
    # Cache teacher latents
    model.eval()
    model.use_vq = False
    teacher_z = {}
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            z_e = model.encode(batch['window'].to(device))
            teacher_z[i] = z_e.cpu()
    
    model.use_vq = True
    
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': cfg.lr_encoder},
        {'params': model.vq.parameters(), 'lr': cfg.lr_vq},
        {'params': model.decoder.parameters(), 'lr': cfg.lr_decoder},
    ], weight_decay=cfg.weight_decay)
    
    best_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, cfg.finetune_epochs + 1):
        model.train()
        for i, batch in enumerate(train_loader):
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            z_teacher = teacher_z[i].to(device)
            
            optimizer.zero_grad()
            out = model(window)
            
            vel_loss = F.mse_loss(out['velocity_pred'], velocity)
            latent_loss = F.mse_loss(out['z_q'], z_teacher)
            commit_loss = out['commitment_loss']
            
            loss = cfg.alpha * vel_loss + cfg.beta * latent_loss + commit_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        
        # Eval
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch['window'].to(device))
                preds.append(out['velocity_pred'].cpu().numpy())
                targets.append(batch['velocity'].numpy())
        
        test_r2 = r2(np.concatenate(preds), np.concatenate(targets))
        
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.patience:
            break
    
    model.load_state_dict(best_state)
    return best_r2


def run_full_pipeline(
    spike_counts: np.ndarray,
    velocities: np.ndarray,
    cfg: Config,
    device: torch.device,
    seed: int,
) -> Dict:
    """Run full exp22c pipeline: Teacher → Init RVQ → Distill Student."""
    set_seed(seed)
    
    # Create datasets
    train_ds = SlidingWindowDataset(
        spike_counts, velocities,
        window_size=cfg.window_size,
        augment=True,
        noise_std=cfg.noise_std,
        time_mask_prob=cfg.time_mask_prob,
    )
    test_ds = SlidingWindowDataset(
        spike_counts, velocities,
        window_size=cfg.window_size,
        augment=False,
    )
    
    n = len(train_ds)
    n_train = int(n * cfg.train_frac)
    train_subset = Subset(train_ds, list(range(n_train)))
    test_subset = Subset(test_ds, list(range(n_train, n)))
    
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=cfg.batch_size)
    
    # Phase 1: Train teacher
    model = DistilledRVQModel(cfg).to(device)
    teacher_r2 = train_teacher(model, train_loader, test_loader, device, cfg)
    
    # Phase 2: Init RVQ
    init_rvq(model, train_loader, device)
    
    # Phase 3: Distill student
    student_r2 = train_student(model, train_loader, test_loader, device, cfg)
    
    return {
        'seed': seed,
        'teacher_r2': teacher_r2,
        'student_r2': student_r2,
        'discretization_tax': (teacher_r2 - student_r2) * 100,
    }


def run_lstm_baseline(
    spike_counts: np.ndarray,
    velocities: np.ndarray,
    cfg: Config,
    device: torch.device,
    seed: int,
    use_augment: bool = True,
) -> float:
    """Run LSTM baseline."""
    set_seed(seed)
    
    train_ds = SlidingWindowDataset(
        spike_counts, velocities,
        window_size=cfg.window_size,
        augment=use_augment,
        noise_std=cfg.noise_std,
        time_mask_prob=cfg.time_mask_prob,
    )
    test_ds = SlidingWindowDataset(
        spike_counts, velocities,
        window_size=cfg.window_size,
        augment=False,
    )
    
    n = len(train_ds)
    n_train = int(n * cfg.train_frac)
    train_subset = Subset(train_ds, list(range(n_train)))
    test_subset = Subset(test_ds, list(range(n_train, n)))
    
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=cfg.batch_size)
    
    model = LSTMBaseline().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.pretrain_epochs)
    
    best_r2 = -float('inf')
    patience_counter = 0
    
    for epoch in range(1, cfg.pretrain_epochs + 1):
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
                preds.append(pred.cpu().numpy())
                targets.append(batch['velocity'].numpy())
        
        test_r2 = r2(np.concatenate(preds), np.concatenate(targets))
        
        if test_r2 > best_r2:
            best_r2 = test_r2
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.patience:
            break
    
    return best_r2


def main():
    print("=" * 70)
    print("EXPERIMENT 24: Statistical Validation of Exp 22c")
    print("              (Multi-Seed Super-Teacher Distillation)")
    print("=" * 70)
    print()
    print("Protocol:")
    print("  - Run FULL pipeline (teacher → RVQ init → distill) 5x")
    print("  - Report: mean ± std, 95% CI, p-value, effect size")
    print("  - Compare: Teacher, Student, LSTM")
    print()
    
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
    base_seeds = [42, 123, 456, 789, 1337]
    
    # ========================================
    # 1. Run Full Exp22c Pipeline (5 seeds)
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 1: Full Exp22c Pipeline (Teacher → RVQ → Student)")
    print("=" * 60)
    
    pipeline_results = []
    for i, seed in enumerate(base_seeds[:cfg.n_seeds]):
        print(f"\n  Run {i+1}/{cfg.n_seeds} (seed={seed})...")
        start_time = time.time()
        
        result = run_full_pipeline(spike_counts, velocities, cfg, device, seed)
        elapsed = time.time() - start_time
        
        print(f"    Teacher R²: {result['teacher_r2']:.4f}")
        print(f"    Student R²: {result['student_r2']:.4f}")
        print(f"    Disc. Tax:  {result['discretization_tax']:.2f}%")
        print(f"    Time: {elapsed:.1f}s")
        
        pipeline_results.append(result)
    
    teacher_scores = np.array([r['teacher_r2'] for r in pipeline_results])
    student_scores = np.array([r['student_r2'] for r in pipeline_results])
    disc_taxes = np.array([r['discretization_tax'] for r in pipeline_results])
    
    # ========================================
    # 2. LSTM Baseline (5 seeds, with augmentation)
    # ========================================
    print("\n" + "=" * 60)
    print("PHASE 2: LSTM Baseline (WITH Augmentation)")
    print("=" * 60)
    
    lstm_scores = []
    for i, seed in enumerate(base_seeds[:cfg.n_seeds]):
        print(f"  Run {i+1}/{cfg.n_seeds} (seed={seed})...", end=" ", flush=True)
        start_time = time.time()
        
        lstm_r2 = run_lstm_baseline(spike_counts, velocities, cfg, device, seed, use_augment=True)
        elapsed = time.time() - start_time
        
        print(f"R² = {lstm_r2:.4f} ({elapsed:.1f}s)")
        lstm_scores.append(lstm_r2)
    
    lstm_scores = np.array(lstm_scores)
    
    # ========================================
    # Statistical Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    # Teacher stats
    teacher_mean = np.mean(teacher_scores)
    teacher_std = np.std(teacher_scores, ddof=1)
    teacher_ci = compute_confidence_interval(teacher_scores)
    
    # Student stats
    student_mean = np.mean(student_scores)
    student_std = np.std(student_scores, ddof=1)
    student_ci = compute_confidence_interval(student_scores)
    
    # LSTM stats
    lstm_mean = np.mean(lstm_scores)
    lstm_std = np.std(lstm_scores, ddof=1)
    lstm_ci = compute_confidence_interval(lstm_scores)
    
    # Discretization tax stats
    tax_mean = np.mean(disc_taxes)
    tax_std = np.std(disc_taxes, ddof=1)
    
    print("\n1. Summary Statistics")
    print("-" * 60)
    print(f"  Teacher:   {teacher_mean:.4f} ± {teacher_std:.4f}  CI: [{teacher_ci[0]:.4f}, {teacher_ci[1]:.4f}]")
    print(f"  Student:   {student_mean:.4f} ± {student_std:.4f}  CI: [{student_ci[0]:.4f}, {student_ci[1]:.4f}]")
    print(f"  LSTM:      {lstm_mean:.4f} ± {lstm_std:.4f}  CI: [{lstm_ci[0]:.4f}, {lstm_ci[1]:.4f}]")
    print(f"  Disc. Tax: {tax_mean:.2f}% ± {tax_std:.2f}%")
    
    # Test 1: Student vs LSTM (paired t-test)
    print("\n2. Student vs LSTM (Paired t-test)")
    print("-" * 60)
    
    t_stat, p_value = stats.ttest_rel(student_scores, lstm_scores)
    d = cohens_d(student_scores, lstm_scores)
    
    try:
        _, p_wilcoxon = stats.wilcoxon(student_scores, lstm_scores)
    except ValueError:
        p_wilcoxon = float('nan')
    
    print(f"  Paired t-test:  t = {t_stat:.3f}, p = {p_value:.4f}")
    print(f"  Wilcoxon test:  p = {p_wilcoxon:.4f}")
    print(f"  Cohen's d:      {d:.3f} ({interpret_cohens_d(d)})")
    print(f"  Mean diff:      {student_mean - lstm_mean:.4f}")
    
    student_beats_lstm = student_mean > lstm_mean and p_value < 0.05
    
    if p_value < 0.05:
        if student_mean > lstm_mean:
            print(f"  → SIGNIFICANT: Student > LSTM (p < 0.05) ✓")
        else:
            print(f"  → SIGNIFICANT: LSTM > Student (p < 0.05)")
    else:
        print(f"  → NOT SIGNIFICANT (p >= 0.05)")
    
    # Test 2: Teacher vs LSTM
    print("\n3. Teacher vs LSTM (Paired t-test)")
    print("-" * 60)
    
    t_stat2, p_value2 = stats.ttest_rel(teacher_scores, lstm_scores)
    d2 = cohens_d(teacher_scores, lstm_scores)
    
    print(f"  Paired t-test:  t = {t_stat2:.3f}, p = {p_value2:.4f}")
    print(f"  Cohen's d:      {d2:.3f} ({interpret_cohens_d(d2)})")
    print(f"  Mean diff:      {teacher_mean - lstm_mean:.4f}")
    
    if p_value2 < 0.05:
        if teacher_mean > lstm_mean:
            print(f"  → SIGNIFICANT: Teacher > LSTM (p < 0.05) ✓")
        else:
            print(f"  → SIGNIFICANT: LSTM > Teacher (p < 0.05)")
    else:
        print(f"  → NOT SIGNIFICANT (p >= 0.05)")
    
    # Test 3: Teacher vs Student (discretization tax significance)
    print("\n4. Discretization Tax Consistency")
    print("-" * 60)
    
    t_stat3, p_value3 = stats.ttest_rel(teacher_scores, student_scores)
    print(f"  Tax range:      [{np.min(disc_taxes):.2f}%, {np.max(disc_taxes):.2f}%]")
    print(f"  Tax mean ± std: {tax_mean:.2f}% ± {tax_std:.2f}%")
    print(f"  Paired t-test:  t = {t_stat3:.3f}, p = {p_value3:.4f}")
    
    if p_value3 < 0.05:
        print(f"  → Discretization causes SIGNIFICANT performance drop")
    else:
        print(f"  → Discretization tax is NOT significant")
    
    # ========================================
    # Summary Table
    # ========================================
    print("\n" + "=" * 70)
    print("PUBLICATION-READY SUMMARY")
    print("=" * 70)
    
    print("\n┌───────────────────────────────────────────────────────────────────────┐")
    print("│              Exp 22c Validation: Neural Decoding Performance          │")
    print("├────────────────────────┬──────────────────┬────────────────────────────┤")
    print("│ Model                  │ R² (mean ± std)  │ 95% CI                     │")
    print("├────────────────────────┼──────────────────┼────────────────────────────┤")
    print(f"│ Teacher (Transformer)  │ {teacher_mean:.4f} ± {teacher_std:.4f} │ [{teacher_ci[0]:.4f}, {teacher_ci[1]:.4f}]           │")
    print(f"│ Student (RVQ-4)        │ {student_mean:.4f} ± {student_std:.4f} │ [{student_ci[0]:.4f}, {student_ci[1]:.4f}]           │")
    print(f"│ LSTM (augmented)       │ {lstm_mean:.4f} ± {lstm_std:.4f} │ [{lstm_ci[0]:.4f}, {lstm_ci[1]:.4f}]           │")
    print("└────────────────────────┴──────────────────┴────────────────────────────┘")
    
    print(f"\nDiscretization Tax: {tax_mean:.2f}% ± {tax_std:.2f}%")
    
    # ========================================
    # Verdict
    # ========================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if student_beats_lstm:
        print(f"\n✅ VALIDATED: RVQ Student SIGNIFICANTLY beats LSTM")
        print(f"   (p = {p_value:.4f}, Cohen's d = {d:.3f} [{interpret_cohens_d(d)}])")
        print(f"\n   Exp 22c claim 'Distilled Student beats LSTM' is CONFIRMED.")
    elif student_mean > lstm_mean:
        print(f"\n⚠️  TRENDING: Student > LSTM but NOT significant")
        print(f"   (p = {p_value:.4f}, need more seeds or larger effect)")
    else:
        print(f"\n❌ NOT VALIDATED: Student does NOT beat LSTM")
        print(f"   (p = {p_value:.4f})")
    
    if p_value2 < 0.05 and teacher_mean > lstm_mean:
        print(f"\n✅ Teacher SIGNIFICANTLY beats LSTM")
        print(f"   (p = {p_value2:.4f}, Cohen's d = {d2:.3f})")
    
    # ========================================
    # Save Results
    # ========================================
    results = {
        "experiment": "exp24_validate_22c",
        "date": datetime.now().isoformat(),
        "n_seeds": cfg.n_seeds,
        "seeds": base_seeds[:cfg.n_seeds],
        "pipeline_results": pipeline_results,
        "models": {
            "teacher": {
                "r2_scores": teacher_scores.tolist(),
                "mean": float(teacher_mean),
                "std": float(teacher_std),
                "ci_low": float(teacher_ci[0]),
                "ci_high": float(teacher_ci[1]),
            },
            "student": {
                "r2_scores": student_scores.tolist(),
                "mean": float(student_mean),
                "std": float(student_std),
                "ci_low": float(student_ci[0]),
                "ci_high": float(student_ci[1]),
            },
            "lstm": {
                "r2_scores": lstm_scores.tolist(),
                "mean": float(lstm_mean),
                "std": float(lstm_std),
                "ci_low": float(lstm_ci[0]),
                "ci_high": float(lstm_ci[1]),
            },
        },
        "discretization_tax": {
            "values": disc_taxes.tolist(),
            "mean": float(tax_mean),
            "std": float(tax_std),
        },
        "statistics": {
            "student_vs_lstm": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "p_wilcoxon": float(p_wilcoxon) if not np.isnan(p_wilcoxon) else None,
                "cohens_d": float(d),
                "effect_size": interpret_cohens_d(d),
                "significant": bool(p_value < 0.05),
            },
            "teacher_vs_lstm": {
                "t_statistic": float(t_stat2),
                "p_value": float(p_value2),
                "cohens_d": float(d2),
                "effect_size": interpret_cohens_d(d2),
                "significant": bool(p_value2 < 0.05),
            },
        },
        "verdict": {
            "student_beats_lstm": bool(student_beats_lstm),
            "teacher_beats_lstm": bool(p_value2 < 0.05 and teacher_mean > lstm_mean),
            "p_value_student": float(p_value),
            "p_value_teacher": float(p_value2),
        },
    }
    
    results_path = RESULTS_DIR / "exp24_validate_22c.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
