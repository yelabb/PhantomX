"""
Experiment 23: Statistical Validation of Wide Transformer (384, 6L)

MOTIVATION:
===========
Exp 21b showed Wide Transformer (384, 6L) achieves R² = 0.8064, beating LSTM (0.8009).
BUT: This is a single run. To claim scientific victory, we need:

1. Statistical Significance: Multiple seeds (n=5) with confidence intervals
2. Fair Comparison: Run LSTM with SAME augmentation as Transformer
3. Paired Statistical Tests: Wilcoxon signed-rank or paired t-test
4. Effect Size: Cohen's d to quantify practical significance
5. Cross-Session Stability: Verify consistency across train/test splits

PROTOCOL:
=========
- Run each model 5x with different random seeds
- Use IDENTICAL data augmentation for fair comparison
- Report: mean ± std, 95% CI, p-value, effect size
- Generate publication-ready comparison table

EXPECTED OUTPUT:
================
If significant: "Wide Transformer beats LSTM (p < 0.05, Cohen's d = X.XX)"
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
# Configuration
# ============================================================

@dataclass 
class Config:
    # Data
    n_channels: int = 142
    window_size: int = 10  # 250ms @ 40Hz
    lag: int = 0
    train_frac: float = 0.8
    
    # Wide Transformer (384, 6L) - THE CHAMPION
    d_model: int = 384
    nhead: int = 8
    num_layers: int = 6  # 6 layers, not 8 (key finding from 21b)
    dim_ff: int = 768
    dropout: float = 0.15
    output_dim: int = 128
    
    # Training
    epochs: int = 150
    lr: float = 3e-4
    batch_size: int = 64
    weight_decay: float = 1e-4
    patience: int = 25
    grad_clip: float = 1.0
    
    # Data augmentation (CRITICAL for fair comparison)
    noise_std: float = 0.1
    time_mask_prob: float = 0.1
    
    # Validation
    n_seeds: int = 5  # Number of random seeds


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
# Models
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


class WideTransformer(nn.Module):
    """Wide Transformer (384, 6L) - The Champion from Exp 21b."""
    
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        
        # Encoder
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
            nn.Linear(cfg.d_model, cfg.output_dim),
            nn.LayerNorm(cfg.output_dim),
        )
        
        # Decoder
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln(x)
        z_e = self.output_proj(x[:, -1, :])
        return self.decoder(z_e)


class LSTMBaseline(nn.Module):
    """LSTM Baseline for fair comparison."""
    
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

def train_single_run(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    cfg: Config,
    verbose: bool = False,
) -> Tuple[float, float, float]:
    """Train model and return (best_r2, r2_vx, r2_vy)."""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
    
    best_test_r2 = -float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            pred = model(window)
            loss = F.mse_loss(pred, velocity)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                window = batch['window'].to(device)
                velocity = batch['velocity'].to(device)
                pred = model(window)
                preds.append(pred.cpu().numpy())
                targets.append(velocity.cpu().numpy())
        
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        test_r2 = r2(preds, targets)
        
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = preds.copy()
            best_targets = targets.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and epoch % 20 == 0:
            print(f"    Epoch {epoch:3d} | R²: {test_r2:.4f} | Best: {best_test_r2:.4f}")
        
        if patience_counter >= cfg.patience:
            break
    
    # Compute per-dimension R²
    r2_vx = r2_score(best_targets[:, 0], best_preds[:, 0])
    r2_vy = r2_score(best_targets[:, 1], best_preds[:, 1])
    
    return best_test_r2, r2_vx, r2_vy


def run_multi_seed_experiment(
    model_class,
    model_kwargs: Dict,
    spike_counts: np.ndarray,
    velocities: np.ndarray,
    cfg: Config,
    device: torch.device,
    name: str,
    use_augment: bool = True,
) -> Dict:
    """Run experiment with multiple seeds."""
    
    results = {
        "r2_scores": [],
        "r2_vx_scores": [],
        "r2_vy_scores": [],
        "seeds": [],
        "times": [],
    }
    
    base_seeds = [42, 123, 456, 789, 1337]
    
    for i, seed in enumerate(base_seeds[:cfg.n_seeds]):
        print(f"  Run {i+1}/{cfg.n_seeds} (seed={seed})...", end=" ", flush=True)
        start_time = time.time()
        
        set_seed(seed)
        
        # Create datasets
        train_ds = SlidingWindowDataset(
            spike_counts, velocities,
            window_size=cfg.window_size,
            lag=cfg.lag,
            augment=use_augment,
            noise_std=cfg.noise_std,
            time_mask_prob=cfg.time_mask_prob,
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
        
        # Create and train model
        model = model_class(**model_kwargs).to(device)
        r2_overall, r2_vx, r2_vy = train_single_run(
            model, train_loader, test_loader, device, cfg
        )
        
        elapsed = time.time() - start_time
        
        results["r2_scores"].append(r2_overall)
        results["r2_vx_scores"].append(r2_vx)
        results["r2_vy_scores"].append(r2_vy)
        results["seeds"].append(seed)
        results["times"].append(elapsed)
        
        print(f"R² = {r2_overall:.4f} ({elapsed:.1f}s)")
    
    # Compute statistics
    r2_arr = np.array(results["r2_scores"])
    results["mean"] = float(np.mean(r2_arr))
    results["std"] = float(np.std(r2_arr, ddof=1))
    results["ci_low"], results["ci_high"] = compute_confidence_interval(r2_arr)
    results["min"] = float(np.min(r2_arr))
    results["max"] = float(np.max(r2_arr))
    
    return results


def main():
    print("=" * 70)
    print("EXPERIMENT 23: Statistical Validation of Wide Transformer (384, 6L)")
    print("=" * 70)
    print()
    print("Protocol:")
    print(f"  - Run each model {5}x with different random seeds")
    print("  - IDENTICAL data augmentation for FAIR comparison")
    print("  - Report: mean ± std, 95% CI, p-value, effect size")
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
    
    # ========================================
    # 1. Wide Transformer WITH Augmentation
    # ========================================
    print("\n" + "=" * 60)
    print("MODEL 1: Wide Transformer (384, 6L) WITH Augmentation")
    print("=" * 60)
    
    transformer_results = run_multi_seed_experiment(
        model_class=WideTransformer,
        model_kwargs={"cfg": cfg},
        spike_counts=spike_counts,
        velocities=velocities,
        cfg=cfg,
        device=device,
        name="Wide Transformer",
        use_augment=True,
    )
    
    print(f"\n  Mean R²: {transformer_results['mean']:.4f} ± {transformer_results['std']:.4f}")
    print(f"  95% CI:  [{transformer_results['ci_low']:.4f}, {transformer_results['ci_high']:.4f}]")
    print(f"  Range:   [{transformer_results['min']:.4f}, {transformer_results['max']:.4f}]")
    
    # ========================================
    # 2. LSTM WITH Augmentation (FAIR comparison)
    # ========================================
    print("\n" + "=" * 60)
    print("MODEL 2: LSTM Baseline WITH Augmentation (FAIR)")
    print("=" * 60)
    
    lstm_aug_results = run_multi_seed_experiment(
        model_class=LSTMBaseline,
        model_kwargs={"n_channels": 142, "hidden_size": 256, "num_layers": 2},
        spike_counts=spike_counts,
        velocities=velocities,
        cfg=cfg,
        device=device,
        name="LSTM (augmented)",
        use_augment=True,
    )
    
    print(f"\n  Mean R²: {lstm_aug_results['mean']:.4f} ± {lstm_aug_results['std']:.4f}")
    print(f"  95% CI:  [{lstm_aug_results['ci_low']:.4f}, {lstm_aug_results['ci_high']:.4f}]")
    print(f"  Range:   [{lstm_aug_results['min']:.4f}, {lstm_aug_results['max']:.4f}]")
    
    # ========================================
    # 3. LSTM WITHOUT Augmentation (Original baseline)
    # ========================================
    print("\n" + "=" * 60)
    print("MODEL 3: LSTM Baseline WITHOUT Augmentation (Original)")
    print("=" * 60)
    
    lstm_noaug_results = run_multi_seed_experiment(
        model_class=LSTMBaseline,
        model_kwargs={"n_channels": 142, "hidden_size": 256, "num_layers": 2},
        spike_counts=spike_counts,
        velocities=velocities,
        cfg=cfg,
        device=device,
        name="LSTM (no augment)",
        use_augment=False,
    )
    
    print(f"\n  Mean R²: {lstm_noaug_results['mean']:.4f} ± {lstm_noaug_results['std']:.4f}")
    print(f"  95% CI:  [{lstm_noaug_results['ci_low']:.4f}, {lstm_noaug_results['ci_high']:.4f}]")
    print(f"  Range:   [{lstm_noaug_results['min']:.4f}, {lstm_noaug_results['max']:.4f}]")
    
    # ========================================
    # Statistical Tests
    # ========================================
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)
    
    transformer_scores = np.array(transformer_results["r2_scores"])
    lstm_aug_scores = np.array(lstm_aug_results["r2_scores"])
    lstm_noaug_scores = np.array(lstm_noaug_results["r2_scores"])
    
    # Test 1: Transformer vs LSTM (with augmentation) - FAIR COMPARISON
    print("\n1. Wide Transformer vs LSTM (both WITH augmentation) - FAIR COMPARISON")
    print("-" * 60)
    
    # Paired t-test (samples are matched by seed)
    t_stat, p_value_paired = stats.ttest_rel(transformer_scores, lstm_aug_scores)
    
    # Wilcoxon signed-rank (non-parametric alternative)
    try:
        w_stat, p_value_wilcoxon = stats.wilcoxon(transformer_scores, lstm_aug_scores)
    except ValueError:
        p_value_wilcoxon = float('nan')  # All differences are zero
    
    # Effect size
    d = cohens_d(transformer_scores, lstm_aug_scores)
    
    print(f"  Paired t-test:     t = {t_stat:.3f}, p = {p_value_paired:.4f}")
    print(f"  Wilcoxon test:     p = {p_value_wilcoxon:.4f}")
    print(f"  Cohen's d:         {d:.3f} ({interpret_cohens_d(d)})")
    print(f"  Mean difference:   {transformer_results['mean'] - lstm_aug_results['mean']:.4f}")
    
    if p_value_paired < 0.05:
        if transformer_results['mean'] > lstm_aug_results['mean']:
            print(f"  → SIGNIFICANT: Transformer > LSTM (p < 0.05) ✓")
        else:
            print(f"  → SIGNIFICANT: LSTM > Transformer (p < 0.05)")
    else:
        print(f"  → NOT SIGNIFICANT (p >= 0.05)")
    
    # Test 2: Transformer vs LSTM (without augmentation) - Original claim
    print("\n2. Wide Transformer (aug) vs LSTM (no aug) - ORIGINAL CLAIM")
    print("-" * 60)
    
    # Independent t-test (different conditions)
    t_stat2, p_value2 = stats.ttest_ind(transformer_scores, lstm_noaug_scores)
    d2 = cohens_d(transformer_scores, lstm_noaug_scores)
    
    print(f"  Independent t-test: t = {t_stat2:.3f}, p = {p_value2:.4f}")
    print(f"  Cohen's d:          {d2:.3f} ({interpret_cohens_d(d2)})")
    print(f"  Mean difference:    {transformer_results['mean'] - lstm_noaug_results['mean']:.4f}")
    
    if p_value2 < 0.05:
        if transformer_results['mean'] > lstm_noaug_results['mean']:
            print(f"  → SIGNIFICANT: Transformer > LSTM (p < 0.05) ✓")
        else:
            print(f"  → SIGNIFICANT: LSTM > Transformer (p < 0.05)")
    else:
        print(f"  → NOT SIGNIFICANT (p >= 0.05)")
    
    # ========================================
    # Summary Table
    # ========================================
    print("\n" + "=" * 70)
    print("PUBLICATION-READY SUMMARY")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    Neural Decoding Performance                      │")
    print("├──────────────────────────┬──────────────────┬───────────────────────┤")
    print("│ Model                    │ R² (mean ± std)  │ 95% CI                │")
    print("├──────────────────────────┼──────────────────┼───────────────────────┤")
    print(f"│ Wide Transformer (aug)   │ {transformer_results['mean']:.4f} ± {transformer_results['std']:.4f} │ [{transformer_results['ci_low']:.4f}, {transformer_results['ci_high']:.4f}]     │")
    print(f"│ LSTM (aug)               │ {lstm_aug_results['mean']:.4f} ± {lstm_aug_results['std']:.4f} │ [{lstm_aug_results['ci_low']:.4f}, {lstm_aug_results['ci_high']:.4f}]     │")
    print(f"│ LSTM (no aug)            │ {lstm_noaug_results['mean']:.4f} ± {lstm_noaug_results['std']:.4f} │ [{lstm_noaug_results['ci_low']:.4f}, {lstm_noaug_results['ci_high']:.4f}]     │")
    print("└──────────────────────────┴──────────────────┴───────────────────────┘")
    
    print(f"\nStatistical significance (Transformer vs LSTM with aug):")
    print(f"  p = {p_value_paired:.4f}, Cohen's d = {d:.3f} ({interpret_cohens_d(d)})")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    fair_winner = "Transformer" if transformer_results['mean'] > lstm_aug_results['mean'] else "LSTM"
    fair_significant = p_value_paired < 0.05
    
    if fair_significant and fair_winner == "Transformer":
        print(f"\n✅ VALIDATED: Wide Transformer SIGNIFICANTLY beats LSTM")
        print(f"   (p = {p_value_paired:.4f}, Cohen's d = {d:.3f})")
        print(f"\n   The claim 'Wide Transformer (384, 6L) beats LSTM' is CONFIRMED.")
    elif fair_significant and fair_winner == "LSTM":
        print(f"\n❌ REFUTED: LSTM actually beats Wide Transformer")
        print(f"   (p = {p_value_paired:.4f})")
    else:
        print(f"\n⚠️  INCONCLUSIVE: No significant difference detected")
        print(f"   (p = {p_value_paired:.4f})")
        print(f"\n   Need more runs or the difference is not practically meaningful.")
    
    # Check if augmentation matters
    print("\nAugmentation Effect on LSTM:")
    t_aug, p_aug = stats.ttest_ind(lstm_aug_scores, lstm_noaug_scores)
    d_aug = cohens_d(lstm_aug_scores, lstm_noaug_scores)
    print(f"  LSTM (aug) vs LSTM (no aug): p = {p_aug:.4f}, d = {d_aug:.3f}")
    if p_aug < 0.05:
        print(f"  → Augmentation has SIGNIFICANT effect on LSTM")
    
    # ========================================
    # Save Results
    # ========================================
    results = {
        "experiment": "exp23_statistical_validation",
        "date": datetime.now().isoformat(),
        "n_seeds": cfg.n_seeds,
        "models": {
            "transformer_aug": transformer_results,
            "lstm_aug": lstm_aug_results,
            "lstm_noaug": lstm_noaug_results,
        },
        "statistics": {
            "fair_comparison": {
                "t_statistic": float(t_stat),
                "p_value_paired": float(p_value_paired),
                "p_value_wilcoxon": float(p_value_wilcoxon) if not np.isnan(p_value_wilcoxon) else None,
                "cohens_d": float(d),
                "effect_size": interpret_cohens_d(d),
            },
            "original_claim": {
                "t_statistic": float(t_stat2),
                "p_value": float(p_value2),
                "cohens_d": float(d2),
                "effect_size": interpret_cohens_d(d2),
            },
        },
        "verdict": {
            "winner": fair_winner,
            "significant": fair_significant,
            "p_value": float(p_value_paired),
        }
    }
    
    results_path = RESULTS_DIR / "exp23_statistical_validation.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
