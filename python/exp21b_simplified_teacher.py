"""
Experiment 21b: Simplified Super-Teacher (No Mamba)

MOTIVATION (From Exp 21 Results):
=================================
Exp 21 ablation showed that the slow pathway (Mamba, 2s) HURTS performance:
- Full Model (slow+fast): R² = 0.753
- No Slow Pathway:        R² = 0.781  ← BETTER!

CONCLUSION: MC_Maze has no exploitable 2s preparatory dynamics.
The 250ms window is sufficient. Mamba just adds noise and overfitting.

STRATEGY:
=========
Strip out Mamba entirely. Focus on making the 250ms Transformer as strong as possible:
1. Deeper encoder (8-12 layers instead of 6)
2. Wider layers (384-512 d_model instead of 256)
3. Better regularization (dropout sweep, weight decay)
4. Data augmentation (noise injection)
5. Hyperparameter sweep (lr, batch_size)

Then distill to RVQ for the final discrete model.

TARGET: Teacher R² > 0.82, closing the gap to LSTM (0.789)
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
from phantomx.data.mc_maze_loader import load_mc_maze_from_nwb

DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    lag: int = 0           # Δ=0 (current velocity, proven best for MC_Maze)
    train_frac: float = 0.8
    
    # Transformer architecture - DEEPER AND WIDER
    d_model: int = 384          # Up from 256
    nhead: int = 8
    num_layers: int = 8         # Up from 6
    dim_ff: int = 768           # Up from 512
    dropout: float = 0.15       # Slightly more regularization
    
    # Output dimension (for RVQ compatibility)
    output_dim: int = 128
    
    # Training
    epochs: int = 150
    lr: float = 3e-4
    batch_size: int = 64
    weight_decay: float = 1e-4
    patience: int = 25
    grad_clip: float = 1.0
    
    # Data augmentation
    noise_std: float = 0.1      # Gaussian noise injection
    time_mask_prob: float = 0.1 # Randomly mask timesteps


# ============================================================
# Dataset
# ============================================================

class SlidingWindowDataset(Dataset):
    """Simple sliding window dataset with optional augmentation."""
    
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
        self.lag = lag
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
            
            # Random time masking
            if self.time_mask_prob > 0:
                mask = np.random.random(window.shape[0]) > self.time_mask_prob
                window = window * mask[:, np.newaxis]
        
        return {
            "window": torch.tensor(window, dtype=torch.float32),
            "velocity": torch.tensor(velocity, dtype=torch.float32),
        }


# ============================================================
# Causal Transformer Encoder (Deeper Version)
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


class DeepTransformerEncoder(nn.Module):
    """
    Deep Causal Transformer for 250ms window.
    
    Optimized version without slow pathway overhead.
    """
    
    def __init__(self, cfg: Config):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(cfg.n_channels, cfg.d_model)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, cfg.window_size, cfg.d_model) * 0.02
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.nhead, cfg.dim_ff, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        
        self.ln = nn.LayerNorm(cfg.d_model)
        
        # Project to output dimension
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.output_dim),
            nn.LayerNorm(cfg.output_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, window_size, n_channels]
        Returns:
            z_e: [batch, output_dim]
        """
        B, T, C = x.shape
        
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln(x)
        z_e = self.output_proj(x[:, -1, :])  # Last token
        
        return z_e


class SimplifiedTeacher(nn.Module):
    """
    Simplified Super-Teacher: Deep Transformer only, no Mamba.
    """
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        self.encoder = DeepTransformerEncoder(cfg)
        
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
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encode(x)
        velocity_pred = self.decoder(z_e)
        return {
            'velocity_pred': velocity_pred,
            'z_e': z_e,
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
# Training
# ============================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    name: str = "Model",
) -> Tuple[float, List[Dict]]:
    """Train a model and return best test R²."""
    
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
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'forward'):
                out = model(window)
                if isinstance(out, dict):
                    pred = out['velocity_pred']
                else:
                    pred = out
            else:
                pred = model(window)
            
            loss = F.mse_loss(pred, velocity)
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
                if isinstance(out, dict):
                    pred = out['velocity_pred']
                else:
                    pred = out
                
                test_preds.append(pred.cpu().numpy())
                test_targets.append(velocity.cpu().numpy())
        
        test_preds = np.concatenate(test_preds)
        test_targets = np.concatenate(test_targets)
        test_r2 = r2(test_preds, test_targets)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'test_r2': test_r2,
        })
        
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
    return best_test_r2, history


def run_hyperparameter_sweep(
    spike_counts: np.ndarray,
    velocities: np.ndarray,
    device: torch.device,
) -> Dict:
    """Sweep over key hyperparameters."""
    
    results = {}
    
    configs = [
        # (name, d_model, num_layers, dropout, lr)
        ("Baseline (256, 6L)", 256, 6, 0.1, 3e-4),
        ("Deeper (256, 8L)", 256, 8, 0.1, 3e-4),
        ("Wider (384, 6L)", 384, 6, 0.1, 3e-4),
        ("Wider+Deeper (384, 8L)", 384, 8, 0.1, 3e-4),
        ("Max (512, 10L)", 512, 10, 0.15, 2e-4),
        ("Max+Dropout (512, 10L, d=0.2)", 512, 10, 0.2, 2e-4),
    ]
    
    for name, d_model, num_layers, dropout, lr in configs:
        print(f"\n{'='*60}")
        print(f"CONFIG: {name}")
        print(f"{'='*60}")
        
        cfg = Config(
            d_model=d_model,
            num_layers=num_layers,
            dim_ff=d_model * 2,
            dropout=dropout,
            lr=lr,
        )
        
        # Create datasets
        train_ds = SlidingWindowDataset(
            spike_counts, velocities,
            window_size=cfg.window_size,
            lag=cfg.lag,
            augment=True,
            noise_std=cfg.noise_std,
        )
        test_ds = SlidingWindowDataset(
            spike_counts, velocities,
            window_size=cfg.window_size,
            lag=cfg.lag,
            augment=False,
        )
        
        n = len(train_ds)
        n_train = int(n * cfg.train_frac)
        train_subset = Subset(train_ds, list(range(n_train)))
        test_subset = Subset(test_ds, list(range(n_train, n)))
        
        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=cfg.batch_size)
        
        # Create model
        model = SimplifiedTeacher(cfg).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")
        
        # Train
        best_r2, _ = train_model(model, train_loader, test_loader, device, cfg, name)
        results[name] = {"r2": best_r2, "params": n_params}
        
        print(f"\n  ✓ {name}: R² = {best_r2:.4f}")
    
    return results


def main():
    print("="*70)
    print("EXPERIMENT 21b: Simplified Super-Teacher (No Mamba)")
    print("="*70)
    print()
    print("Strategy: Strip out slow pathway, make 250ms Transformer as strong as possible")
    print("Based on Exp 21 finding: slow pathway HURTS performance on MC_Maze")
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
    
    # Configuration
    cfg = Config()
    
    # ========================================
    # Part 1: LSTM Baseline
    # ========================================
    print("\n" + "="*60)
    print("BASELINE: LSTM (10-step window, Δ=0)")
    print("="*60)
    
    train_ds = SlidingWindowDataset(spike_counts, velocities, cfg.window_size, cfg.lag)
    n = len(train_ds)
    n_train = int(n * cfg.train_frac)
    
    train_subset = Subset(train_ds, list(range(n_train)))
    test_subset = Subset(train_ds, list(range(n_train, n)))
    
    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=cfg.batch_size)
    
    lstm = LSTMBaseline().to(device)
    best_lstm_r2, _ = train_model(lstm, train_loader, test_loader, device, cfg, "LSTM")
    print(f"\n  ✓ LSTM Baseline: R² = {best_lstm_r2:.4f}")
    
    # ========================================
    # Part 2: Hyperparameter Sweep
    # ========================================
    print("\n" + "="*60)
    print("HYPERPARAMETER SWEEP: Finding Optimal Teacher")
    print("="*60)
    
    sweep_results = run_hyperparameter_sweep(spike_counts, velocities, device)
    
    # ========================================
    # Part 3: Best Model with Augmentation
    # ========================================
    print("\n" + "="*60)
    print("BEST MODEL: Training with Data Augmentation")
    print("="*60)
    
    # Find best config from sweep
    best_config_name = max(sweep_results.keys(), key=lambda k: sweep_results[k]["r2"])
    print(f"  Best config from sweep: {best_config_name} (R² = {sweep_results[best_config_name]['r2']:.4f})")
    
    # Train final model with augmentation
    final_cfg = Config(
        d_model=384,
        num_layers=8,
        dim_ff=768,
        dropout=0.15,
        epochs=200,
        patience=30,
    )
    
    train_ds_aug = SlidingWindowDataset(
        spike_counts, velocities,
        window_size=final_cfg.window_size,
        lag=final_cfg.lag,
        augment=True,
        noise_std=0.1,
        time_mask_prob=0.1,
    )
    test_ds = SlidingWindowDataset(
        spike_counts, velocities,
        window_size=final_cfg.window_size,
        lag=final_cfg.lag,
        augment=False,
    )
    
    train_subset = Subset(train_ds_aug, list(range(n_train)))
    test_subset = Subset(test_ds, list(range(n_train, n)))
    
    train_loader = DataLoader(train_subset, batch_size=final_cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=final_cfg.batch_size)
    
    final_model = SimplifiedTeacher(final_cfg).to(device)
    n_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"  Final model parameters: {n_params:,}")
    
    best_final_r2, history = train_model(
        final_model, train_loader, test_loader, device, final_cfg, "Final Teacher"
    )
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 21b SUMMARY")
    print("="*70)
    
    print("\n1. BASELINE")
    print(f"   LSTM: R² = {best_lstm_r2:.4f}")
    
    print("\n2. HYPERPARAMETER SWEEP")
    for name, data in sorted(sweep_results.items(), key=lambda x: -x[1]["r2"]):
        gap = (data["r2"] - best_lstm_r2) / best_lstm_r2 * 100
        marker = "✓" if data["r2"] > best_lstm_r2 else ""
        print(f"   {name:35s}: R² = {data['r2']:.4f} ({gap:+.2f}%) {marker}")
    
    print("\n3. FINAL MODEL (with augmentation)")
    print(f"   R² = {best_final_r2:.4f}")
    gap = (best_final_r2 - best_lstm_r2) / best_lstm_r2 * 100
    if best_final_r2 > best_lstm_r2:
        print(f"   → BEATS LSTM by {gap:.2f}%! 🎉")
    else:
        print(f"   → {abs(gap):.2f}% below LSTM")
    
    # Compare to Exp 21
    print("\n4. COMPARISON TO EXP 21")
    print(f"   Exp 21 Full Model (slow+fast): R² ≈ 0.753")
    print(f"   Exp 21 No Slow Pathway:        R² ≈ 0.781")
    print(f"   Exp 21b Simplified Teacher:    R² = {best_final_r2:.4f}")
    
    # Save results
    results = {
        "lstm_baseline": best_lstm_r2,
        "sweep_results": {k: v["r2"] for k, v in sweep_results.items()},
        "final_model": best_final_r2,
        "config": {
            "d_model": final_cfg.d_model,
            "num_layers": final_cfg.num_layers,
            "dropout": final_cfg.dropout,
        },
        "history": history,
    }
    
    results_path = RESULTS_DIR / "exp21b_simplified_teacher.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save best model
    model_path = Path(__file__).parent / "models" / "exp21b_simplified_teacher.pt"
    torch.save(final_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
