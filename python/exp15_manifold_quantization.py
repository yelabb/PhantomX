"""
Experiment 15: Manifold Quantization with Physics-Aligned Neural Tokenizer

THE ARCHITECTURE:
=================
    Spikes → CausalTransformerEncoder → z_e → FSQ[6,6,6,6] → z_q
                                                 ↓
                                         ┌──────┴──────┐
                                         ↓             ↓
                              StatelessMamba      SpikeDecoder
                              (Velocity)          (Reconstruction)
                                         ↓             ↓
                                      L_vel        L_recon
                                         
                              LatentPredictor: z_t → z_{t+1}
                                         ↓
                                     L_dynamics

LOSS FUNCTION:
==============
    L_total = L_velocity + 0.5 * L_reconstruction + 0.1 * L_dynamics

KEY INNOVATIONS:
================
1. FSQ[6,6,6,6] = 1296 codes with ordinal topology
2. StatelessMamba decoder (no shuffling issues)
3. LatentPredictor forces temporal structure in latent space
4. TTA via quantization error minimization

TARGET: R² > 0.78 (Beat LSTM baseline)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent))

from phantomx.vqvae.fsq import FiniteScalarQuantization
from phantomx.models_extended import (
    TransformerEncoder,
    StatelessMambaDecoder,
    LatentPredictor
)
from phantomx.tta.fsq_optimizer import FSQTestTimeAdapter

# Paths
DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"
SAVE_DIR = Path(__file__).parent / "models"


# ============================================================
# Configuration
# ============================================================

@dataclass
class ManifoldConfig:
    """Experiment 15 configuration."""
    
    # Data
    n_channels: int = 142
    n_bins: int = 10  # 250ms window at 40Hz
    bin_size_ms: float = 25.0
    
    # Encoder (CausalTransformer from Exp 11)
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 4
    encoder_dim: int = 128  # Output dimension
    
    # FSQ Bottleneck: 6^4 = 1296 codes
    fsq_levels: Tuple[int, ...] = (6, 6, 6, 6)
    d_fsq: int = 4  # len(fsq_levels)
    
    # Decoder A: StatelessMamba (Velocity)
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_layers: int = 2
    
    # Decoder B: SpikeReconstruction (MLP)
    spike_decoder_hidden: Tuple[int, ...] = (256, 256)
    
    # Dynamics: LatentPredictor
    dynamics_hidden: int = 64
    
    # Loss weights
    lambda_recon: float = 0.5
    lambda_dynamics: float = 0.1
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 150
    dropout: float = 0.1
    
    # Output
    output_dim: int = 2


# ============================================================
# Dataset
# ============================================================

class TemporalSlidingWindowDataset(Dataset):
    """
    Dataset providing temporal windows with:
    - Normalized spikes (encoder input)
    - Raw spikes (reconstruction target)
    - Velocities (kinematics target)
    - Temporal pairs for dynamics loss (z_t, z_{t+1})
    """
    
    def __init__(
        self,
        spike_counts: np.ndarray,
        velocities: np.ndarray,
        window_size: int = 10,
        normalize: bool = True
    ):
        self.window_size = window_size
        n_windows = len(spike_counts) - window_size
        
        # Store raw spikes for Poisson NLL
        self.spikes_raw = np.stack([
            spike_counts[i:i+window_size].T
            for i in range(n_windows)
        ]).astype(np.float32)
        
        # Normalize spikes for encoder
        if normalize:
            spike_sqrt = np.sqrt(spike_counts)
            mean = spike_sqrt.mean(axis=0, keepdims=True)
            std = spike_sqrt.std(axis=0, keepdims=True) + 1e-6
            spike_norm = (spike_sqrt - mean) / std
        else:
            spike_norm = spike_counts
        
        self.spikes_norm = np.stack([
            spike_norm[i:i+window_size].T
            for i in range(n_windows)
        ]).astype(np.float32)
        
        # Velocities at window end
        self.velocities = velocities[window_size:window_size+n_windows].astype(np.float32)
        
        # Valid indices for dynamics (need t and t+1)
        self.n_samples = n_windows - 1  # Last window has no "next"
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {
            'spikes': torch.tensor(self.spikes_norm[idx]),
            'spikes_raw': torch.tensor(self.spikes_raw[idx]),
            'kinematics': torch.tensor(self.velocities[idx]),
            # For dynamics loss: provide next window's data
            'spikes_next': torch.tensor(self.spikes_norm[idx + 1]),
            'idx': idx
        }


# ============================================================
# Main Model: ManifoldFSQVAE
# ============================================================

class CausalTransformerEncoder(nn.Module):
    """
    Causal Transformer Encoder (from Exp 11).
    
    Processes temporal spike windows with causal attention.
    """
    
    def __init__(self, config: ManifoldConfig):
        super().__init__()
        
        self.config = config
        
        # Input projection
        self.input_norm = nn.LayerNorm(config.n_channels)
        self.input_proj = nn.Linear(config.n_channels, config.d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.n_bins, config.d_model) * 0.02
        )
        
        # Causal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_encoder_layers
        )
        
        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(config.n_bins, config.n_bins), diagonal=1).bool()
        )
        
        # Aggregation to single vector
        self.aggregator = nn.Sequential(
            nn.Linear(config.d_model * config.n_bins, config.encoder_dim * 2),
            nn.LayerNorm(config.encoder_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.encoder_dim * 2, config.encoder_dim)
        )
    
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes: [batch, n_channels, n_bins]
        Returns:
            z_e: [batch, encoder_dim]
        """
        batch_size = spikes.shape[0]
        
        # [batch, n_bins, n_channels]
        x = spikes.permute(0, 2, 1)
        
        # Project
        x = self.input_norm(x)
        x = self.input_proj(x)
        x = x + self.pos_encoding
        
        # Causal attention
        x = self.transformer(x, mask=self.causal_mask)
        
        # Aggregate
        x = x.reshape(batch_size, -1)
        z_e = self.aggregator(x)
        
        return z_e


class SpikeReconstructionDecoder(nn.Module):
    """Simple MLP decoder for spike reconstruction."""
    
    def __init__(self, config: ManifoldConfig):
        super().__init__()
        
        input_dim = config.encoder_dim
        output_dim = config.n_channels * config.n_bins
        
        layers = []
        in_dim = input_dim
        for hidden_dim in config.spike_decoder_hidden:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        self.n_channels = config.n_channels
        self.n_bins = config.n_bins
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: [batch, encoder_dim]
        Returns:
            log_rates: [batch, n_channels, n_bins]
        """
        log_rates = self.decoder(z_q)
        return log_rates.view(-1, self.n_channels, self.n_bins)


class ManifoldFSQVAE(nn.Module):
    """
    Complete Manifold FSQ-VAE model.
    
    Components:
    - Encoder: CausalTransformerEncoder
    - Bottleneck: FiniteScalarQuantization [6,6,6,6]
    - Decoder A: StatelessMambaDecoder (velocity)
    - Decoder B: SpikeReconstructionDecoder
    - Dynamics: LatentPredictor
    """
    
    def __init__(self, config: ManifoldConfig):
        super().__init__()
        
        self.config = config
        
        # Encoder
        self.encoder = CausalTransformerEncoder(config)
        
        # FSQ Bottleneck
        self.fsq = FiniteScalarQuantization(
            levels=list(config.fsq_levels),
            input_dim=config.encoder_dim
        )
        
        # Decoder A: Velocity (using MLP instead of Mamba for simpler sequence handling)
        # Note: StatelessMamba expects sequence input, but we aggregate to single vector
        # Use MLP decoder for single-vector prediction
        self.velocity_decoder = nn.Sequential(
            nn.Linear(config.encoder_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, config.output_dim)
        )
        
        # Decoder B: Spike Reconstruction
        self.spike_decoder = SpikeReconstructionDecoder(config)
        
        # Dynamics Predictor (operates on FSQ latent space)
        self.dynamics = LatentPredictor(
            d_fsq=config.d_fsq,
            hidden_dim=config.dynamics_hidden,
            dropout=config.dropout
        )
        
        # Loss weights
        self.lambda_recon = config.lambda_recon
        self.lambda_dynamics = config.lambda_dynamics
    
    def forward(
        self,
        spikes: torch.Tensor,
        spikes_raw: torch.Tensor,
        kinematics_target: Optional[torch.Tensor] = None,
        spikes_next: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            spikes: [B, n_channels, n_bins] normalized spikes
            spikes_raw: [B, n_channels, n_bins] raw spike counts
            kinematics_target: [B, 2] velocity targets
            spikes_next: [B, n_channels, n_bins] next window (for dynamics)
        """
        batch_size = spikes.shape[0]
        device = spikes.device
        
        # Encode
        z_e = self.encoder(spikes)  # [B, encoder_dim]
        
        # Quantize
        z_q, fsq_info = self.fsq(z_e)
        
        # Decode velocity
        velocity_pred = self.velocity_decoder(z_q)  # [B, 2]
        
        # Decode spikes
        spike_logrates = self.spike_decoder(z_q)  # [B, n_channels, n_bins]
        
        # Output dict
        output = {
            'velocity_pred': velocity_pred,
            'spike_logrates': spike_logrates,
            'z_e': z_e,
            'z_q': z_q,
            'z_hat': fsq_info['z_hat'],  # Continuous pre-quantized
            'z_fsq': fsq_info['z_fsq'],  # Quantized FSQ codes
            'indices': fsq_info['indices'],
            'perplexity': fsq_info['perplexity'],
            'codebook_usage': fsq_info['codebook_usage'],
            'quant_error': fsq_info['quant_error'],
        }
        
        # Compute losses if targets provided
        if kinematics_target is not None:
            # Velocity loss (MSE)
            velocity_loss = F.mse_loss(velocity_pred, kinematics_target)
            output['velocity_loss'] = velocity_loss
            
            # Reconstruction loss (Poisson NLL)
            recon_loss = F.poisson_nll_loss(
                spike_logrates, spikes_raw,
                log_input=True, full=False, reduction='mean'
            )
            output['recon_loss'] = recon_loss
            
            # Dynamics loss (if next window provided)
            if spikes_next is not None:
                with torch.no_grad():
                    z_e_next = self.encoder(spikes_next)
                    _, fsq_info_next = self.fsq(z_e_next)
                    z_hat_next = fsq_info_next['z_hat']  # Target: continuous pre-quantized
                
                dynamics_loss = self.dynamics.compute_loss(
                    fsq_info['z_hat'],  # Current z_hat
                    z_hat_next  # Next z_hat (target)
                )
                output['dynamics_loss'] = dynamics_loss
            else:
                dynamics_loss = torch.tensor(0.0, device=device)
                output['dynamics_loss'] = dynamics_loss
            
            # Total loss
            total_loss = (
                velocity_loss +
                self.lambda_recon * recon_loss +
                self.lambda_dynamics * dynamics_loss
            )
            output['total_loss'] = total_loss
        
        return output
    
    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size


# ============================================================
# Training
# ============================================================

def train_manifold_fsq(
    model: ManifoldFSQVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ManifoldConfig,
    device: torch.device
) -> Dict:
    """Train ManifoldFSQVAE."""
    
    model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )
    
    history = {
        'train_loss': [], 'val_loss': [],
        'velocity_loss': [], 'recon_loss': [], 'dynamics_loss': [],
        'train_r2': [], 'val_r2': [],
        'perplexity': [], 'codebook_usage': [], 'quant_error': []
    }
    
    best_val_r2 = -float('inf')
    
    print("=" * 70)
    print("Experiment 15: Manifold FSQ-VAE")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"FSQ Codebook: {model.codebook_size} codes (levels: {config.fsq_levels})")
    print(f"Loss weights: λ_recon={config.lambda_recon}, λ_dynamics={config.lambda_dynamics}")
    print("=" * 70)
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        
        epoch_metrics = {
            'loss': 0, 'vel_loss': 0, 'recon_loss': 0, 'dyn_loss': 0,
            'perp': 0, 'usage': 0, 'qerr': 0
        }
        all_preds, all_targets = [], []
        n_batches = 0
        
        for batch in train_loader:
            spikes = batch['spikes'].to(device)
            spikes_raw = batch['spikes_raw'].to(device)
            targets = batch['kinematics'].to(device)
            spikes_next = batch['spikes_next'].to(device)
            
            # Forward
            output = model(spikes, spikes_raw, targets, spikes_next)
            loss = output['total_loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['vel_loss'] += output['velocity_loss'].item()
            epoch_metrics['recon_loss'] += output['recon_loss'].item()
            epoch_metrics['dyn_loss'] += output['dynamics_loss'].item()
            epoch_metrics['perp'] += output['perplexity'].item()
            epoch_metrics['usage'] += output['codebook_usage']
            epoch_metrics['qerr'] += output['quant_error'].item()
            
            all_preds.append(output['velocity_pred'].detach().cpu())
            all_targets.append(targets.cpu())
            n_batches += 1
        
        scheduler.step()
        
        # Epoch averages
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches
        
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        train_r2 = r2_score(all_targets, all_preds)
        
        # Validation
        val_metrics = evaluate(model, val_loader, device)
        
        # Store history
        history['train_loss'].append(epoch_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['velocity_loss'].append(epoch_metrics['vel_loss'])
        history['recon_loss'].append(epoch_metrics['recon_loss'])
        history['dynamics_loss'].append(epoch_metrics['dyn_loss'])
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_metrics['r2'])
        history['perplexity'].append(epoch_metrics['perp'])
        history['codebook_usage'].append(epoch_metrics['usage'])
        history['quant_error'].append(epoch_metrics['qerr'])
        
        # Print
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{config.epochs} | "
                  f"Loss: {epoch_metrics['loss']:.4f} | "
                  f"Vel: {epoch_metrics['vel_loss']:.4f} | "
                  f"Recon: {epoch_metrics['recon_loss']:.4f} | "
                  f"Dyn: {epoch_metrics['dyn_loss']:.4f} | "
                  f"Train R²: {train_r2:.4f} | "
                  f"Val R²: {val_metrics['r2']:.4f}")
        
        # Save best
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_r2': best_val_r2,
                'config': config
            }, SAVE_DIR / 'exp15_manifold_best.pt')
    
    # Final save
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': config
    }, SAVE_DIR / 'exp15_manifold_final.pt')
    
    with open(SAVE_DIR / 'exp15_manifold_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Training Complete!")
    print(f"Best Val R²: {best_val_r2:.4f}")
    print("=" * 70)
    
    # Baseline comparison
    lstm_r2 = 0.78
    if best_val_r2 > lstm_r2:
        print(f"✓ BEAT LSTM BASELINE! (+{(best_val_r2 - lstm_r2)*100:.2f}%)")
    else:
        gap = lstm_r2 - best_val_r2
        print(f"✗ Gap to LSTM: {gap:.4f} ({gap/lstm_r2*100:.1f}%)")
    
    return history, best_val_r2


def evaluate(model, data_loader, device):
    """Evaluate model."""
    model.eval()
    
    total_loss = 0
    all_preds, all_targets = [], []
    n_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            spikes = batch['spikes'].to(device)
            spikes_raw = batch['spikes_raw'].to(device)
            targets = batch['kinematics'].to(device)
            spikes_next = batch['spikes_next'].to(device)
            
            output = model(spikes, spikes_raw, targets, spikes_next)
            
            total_loss += output['total_loss'].item()
            all_preds.append(output['velocity_pred'].cpu())
            all_targets.append(targets.cpu())
            n_batches += 1
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    return {
        'loss': total_loss / n_batches,
        'r2': r2_score(all_targets, all_preds)
    }


# ============================================================
# Main
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading MC_Maze dataset...")
    try:
        from phantomx.data.mc_maze_loader import load_mc_maze_from_nwb
        spike_counts, kinematics = load_mc_maze_from_nwb(str(DATA_PATH))
        velocities = kinematics[:, 2:4]
        print(f"Loaded: {spike_counts.shape[0]} samples, {spike_counts.shape[1]} channels")
    except Exception as e:
        print(f"Could not load NWB: {e}")
        print("Generating synthetic data...")
        
        np.random.seed(42)
        n_samples = 10000
        n_channels = 142
        
        rates = np.random.uniform(0.5, 5, n_channels)
        spike_counts = np.random.poisson(rates, (n_samples, n_channels)).astype(np.float32)
        
        W = np.random.randn(n_channels, 2) * 0.1
        velocities = spike_counts @ W + np.random.randn(n_samples, 2) * 0.5
        velocities = velocities.astype(np.float32)
    
    # Train/val split
    n_train = int(len(spike_counts) * 0.8)
    
    spike_train = spike_counts[:n_train]
    spike_val = spike_counts[n_train:]
    vel_train = velocities[:n_train]
    vel_val = velocities[n_train:]
    
    print(f"Train: {len(spike_train)}, Val: {len(spike_val)}")
    
    # Config
    config = ManifoldConfig(n_channels=spike_counts.shape[1])
    
    # Datasets
    train_dataset = TemporalSlidingWindowDataset(
        spike_train, vel_train, window_size=config.n_bins
    )
    val_dataset = TemporalSlidingWindowDataset(
        spike_val, vel_val, window_size=config.n_bins
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    
    # Model
    model = ManifoldFSQVAE(config)
    
    # Train
    history, best_r2 = train_manifold_fsq(
        model, train_loader, val_loader, config, device
    )
    
    print(f"\n✓ Experiment 15 complete!")
    print(f"  Best model: {SAVE_DIR / 'exp15_manifold_best.pt'}")


if __name__ == "__main__":
    main()
