"""
Experiment 14: Finite Scalar Quantization (FSQ) with Dual-Head Decoder

THE PIVOT - Addressing Red Team Critique:
==========================================

1. Categorical vs. Ordinal Mismatch (Voronoi Ceiling):
   - VQ treats codes as orthogonal categories
   - Velocity is ordinal/continuous - neighboring values should have similar codes
   - FSQ creates topology-preserving discrete representations
   - Code [1,2,0] is CLOSE to [1,3,0], unlike VQ where Code 42 ≠ Code 43

2. Supervised Bottleneck Trap:
   - Previous experiments trained codebook only for velocity MSE
   - This discards neural variance not correlated with velocity
   - Dual-head decoder forces latent to be a TRUE neural summary
   - Head A: Kinematics (task loss)
   - Head B: Spike reconstruction (foundation model objective)

3. No Codebook Collapse:
   - FSQ uses entire hypercube volume by design
   - No EMA, k-means initialization, or commitment costs needed

ARCHITECTURE:
=============
    Spikes → CausalTransformer → z_e → FSQ[8,5,5,5] → z_q
                                                      ↓
                            [Decoder A: Velocity] ← z_q → [Decoder B: Spikes]
                                    ↓                            ↓
                                 MSE Loss                  Poisson NLL Loss

LOSS FUNCTION:
==============
    L = L_velocity + λ * L_reconstruction
    
    Information Bottleneck Interpretation:
    - Maximize I(Z; Y) via velocity prediction
    - Maintain I(Z; X) via spike reconstruction
    - Creates robust, generalizable neural representation

TARGET: R² > 0.78 (Beat LSTM baseline)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import time
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from phantomx.vqvae import (
    FSQ, 
    FSQVAE, 
    FSQVAEWithCausalTransformer,
    FSQVAETrainer,
    compute_baseline_comparison
)
from phantomx.data import MCMazeDataset

# Paths
DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"
SAVE_DIR = Path(__file__).parent / "models"


# ============================================================
# Experiment Configuration
# ============================================================

@dataclass
class FSQConfig:
    """FSQ-VAE hyperparameters based on theoretical analysis."""
    
    # Data settings
    n_channels: int = 142
    n_bins: int = 10  # 250ms window at 40Hz
    bin_size_ms: float = 25.0
    
    # FSQ quantizer (creates 8*5*5*5 = 1000 codes)
    # Chosen to match VQ codebook size but with ordinal structure
    fsq_levels: Tuple[int, ...] = (8, 5, 5, 5)
    
    # Encoder (Causal Transformer)
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    embedding_dim: int = 128
    
    # Decoders
    decoder_hidden_dims: Tuple[int, ...] = (256, 256, 128)
    spike_decoder_hidden_dims: Tuple[int, ...] = (256, 256, 256)
    
    # Loss weighting
    reconstruction_weight: float = 0.5  # λ in the loss function
    use_weight_schedule: bool = False  # Optionally decay λ over training
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 150
    dropout: float = 0.1
    
    # Output
    output_dim: int = 2  # (vx, vy)


# ============================================================
# Dataset
# ============================================================

class SlidingWindowDataset(Dataset):
    """
    Dataset providing windows of neural activity with:
    - Normalized spikes (for encoder input)
    - Raw spikes (for Poisson reconstruction target)
    - Velocities (for kinematics prediction)
    """
    
    def __init__(
        self,
        spike_counts: np.ndarray,
        velocities: np.ndarray,
        window_size: int = 10,
        normalize: bool = True
    ):
        self.window_size = window_size
        n_windows = len(spike_counts) - window_size + 1
        
        # Store raw spikes for Poisson NLL (reconstruction target)
        self.spikes_raw = np.stack([
            spike_counts[i:i+window_size].T  # [n_channels, n_bins]
            for i in range(n_windows)
        ])
        
        # Normalize spikes for encoder input
        if normalize:
            # Square root transform (variance stabilizing for Poisson)
            spike_sqrt = np.sqrt(spike_counts)
            # Z-score normalization per channel
            mean = spike_sqrt.mean(axis=0, keepdims=True)
            std = spike_sqrt.std(axis=0, keepdims=True) + 1e-6
            spike_norm = (spike_sqrt - mean) / std
        else:
            spike_norm = spike_counts
        
        # Create normalized windows
        self.spikes_norm = np.stack([
            spike_norm[i:i+window_size].T  # [n_channels, n_bins]
            for i in range(n_windows)
        ])
        
        # Velocities (target at window end)
        self.velocities = velocities[window_size-1:window_size-1+n_windows]
    
    def __len__(self):
        return len(self.spikes_norm)
    
    def __getitem__(self, idx):
        return {
            'spikes': torch.tensor(self.spikes_norm[idx], dtype=torch.float32),
            'spikes_raw': torch.tensor(self.spikes_raw[idx], dtype=torch.float32),
            'kinematics': torch.tensor(self.velocities[idx], dtype=torch.float32)
        }


# ============================================================
# FSQ-VAE Model (Causal Transformer + FSQ + Dual Head)
# ============================================================

class CausalTransformerFSQVAE(nn.Module):
    """
    Complete FSQ-VAE with:
    - Causal Transformer encoder (proven to work in previous experiments)
    - FSQ quantizer (topology-preserving discrete bottleneck)
    - Dual-head decoder (kinematics + spike reconstruction)
    """
    
    def __init__(self, config: FSQConfig):
        super().__init__()
        self.config = config
        
        # ======== INPUT PROJECTION ========
        self.input_norm = nn.LayerNorm(config.n_channels)
        self.input_proj = nn.Linear(config.n_channels, config.d_model)
        
        # ======== POSITIONAL ENCODING ========
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.n_bins, config.d_model) * 0.02
        )
        
        # ======== CAUSAL TRANSFORMER ENCODER ========
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers
        )
        
        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(config.n_bins, config.n_bins), diagonal=1).bool()
        )
        
        # ======== AGGREGATION → LATENT ========
        self.aggregator = nn.Sequential(
            nn.Linear(config.d_model * config.n_bins, config.embedding_dim * 2),
            nn.LayerNorm(config.embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim * 2, config.embedding_dim)
        )
        
        # ======== FSQ QUANTIZER ========
        self.fsq = FSQ(
            levels=list(config.fsq_levels),
            input_dim=config.embedding_dim
        )
        
        # ======== DECODER A: KINEMATICS ========
        self.kinematics_decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.decoder_hidden_dims[0]),
            nn.LayerNorm(config.decoder_hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_dims[0], config.decoder_hidden_dims[1]),
            nn.LayerNorm(config.decoder_hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.decoder_hidden_dims[1], config.decoder_hidden_dims[2]),
            nn.LayerNorm(config.decoder_hidden_dims[2]),
            nn.GELU(),
            nn.Linear(config.decoder_hidden_dims[2], config.output_dim)
        )
        
        # ======== DECODER B: SPIKE RECONSTRUCTION ========
        spike_output_dim = config.n_channels * config.n_bins
        self.spike_decoder = nn.Sequential(
            nn.Linear(config.embedding_dim, config.spike_decoder_hidden_dims[0]),
            nn.LayerNorm(config.spike_decoder_hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.spike_decoder_hidden_dims[0], config.spike_decoder_hidden_dims[1]),
            nn.LayerNorm(config.spike_decoder_hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.spike_decoder_hidden_dims[1], config.spike_decoder_hidden_dims[2]),
            nn.LayerNorm(config.spike_decoder_hidden_dims[2]),
            nn.GELU(),
            nn.Linear(config.spike_decoder_hidden_dims[2], spike_output_dim)
        )
        
        # Loss weight
        self.reconstruction_weight = config.reconstruction_weight
    
    def encode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Encode spikes to continuous latent.
        
        Args:
            spikes: [batch, n_channels, n_bins] normalized spike counts
            
        Returns:
            z_e: [batch, embedding_dim] continuous latent
        """
        batch_size = spikes.shape[0]
        
        # Reshape: [batch, n_bins, n_channels]
        x = spikes.permute(0, 2, 1)
        
        # Normalize and project
        x = self.input_norm(x)
        x = self.input_proj(x)  # [batch, n_bins, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Causal transformer
        x = self.transformer(x, mask=self.causal_mask)  # [batch, n_bins, d_model]
        
        # Flatten and aggregate
        x = x.reshape(batch_size, -1)  # [batch, n_bins * d_model]
        z_e = self.aggregator(x)  # [batch, embedding_dim]
        
        return z_e
    
    def forward(
        self,
        spikes: torch.Tensor,
        spikes_raw: torch.Tensor,
        kinematics_target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            spikes: [batch, n_channels, n_bins] normalized spikes (encoder input)
            spikes_raw: [batch, n_channels, n_bins] raw spike counts (reconstruction target)
            kinematics_target: [batch, 2] velocity targets
            
        Returns:
            Dictionary with predictions, losses, and statistics
        """
        batch_size = spikes.shape[0]
        device = spikes.device
        
        # ======== ENCODE ========
        z_e = self.encode(spikes)  # [batch, embedding_dim]
        
        # ======== QUANTIZE (FSQ) ========
        z_q, fsq_info = self.fsq(z_e)
        
        # ======== DECODE: KINEMATICS ========
        kinematics_pred = self.kinematics_decoder(z_q)  # [batch, 2]
        
        # ======== DECODE: SPIKES ========
        spike_logrates = self.spike_decoder(z_q)  # [batch, n_channels * n_bins]
        spike_logrates = spike_logrates.view(
            batch_size, self.config.n_channels, self.config.n_bins
        )
        
        # ======== OUTPUT ========
        output = {
            'kinematics_pred': kinematics_pred,
            'spike_logrates': spike_logrates,
            'z_e': z_e,
            'z_q': z_q,
            'indices': fsq_info['indices'],
            'perplexity': fsq_info['perplexity'],
            'codebook_usage': fsq_info['codebook_usage'],
        }
        
        if 'z_fsq' in fsq_info:
            output['z_fsq'] = fsq_info['z_fsq']
        
        # ======== COMPUTE LOSSES ========
        if kinematics_target is not None:
            # Kinematics loss (MSE)
            kinematics_loss = F.mse_loss(kinematics_pred, kinematics_target)
            output['kinematics_loss'] = kinematics_loss
            
            # Spike reconstruction loss (Poisson NLL)
            # Note: Poisson NLL expects log(rate) as input
            reconstruction_loss = F.poisson_nll_loss(
                spike_logrates,
                spikes_raw,
                log_input=True,
                full=False,
                reduction='mean'
            )
            output['reconstruction_loss'] = reconstruction_loss
            
            # Total loss
            total_loss = kinematics_loss + self.reconstruction_weight * reconstruction_loss
            output['total_loss'] = total_loss
        
        return output
    
    @property
    def codebook_size(self) -> int:
        return self.fsq.codebook_size


# ============================================================
# Training Function
# ============================================================

def train_fsq_vae(
    model: CausalTransformerFSQVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: FSQConfig,
    device: torch.device
) -> Dict:
    """
    Train FSQ-VAE with dual-head decoder.
    """
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Scheduler: Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )
    
    # History tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'kinematics_loss': [], 'reconstruction_loss': [],
        'train_r2': [], 'val_r2': [],
        'perplexity': [], 'codebook_usage': []
    }
    
    best_val_r2 = -float('inf')
    
    print("=" * 70)
    print("FSQ-VAE Training")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Codebook size: {model.codebook_size} (FSQ levels: {config.fsq_levels})")
    print(f"Reconstruction weight (λ): {config.reconstruction_weight}")
    print("=" * 70)
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_metrics = {
            'loss': 0, 'kin_loss': 0, 'recon_loss': 0,
            'perplexity': 0, 'codebook_usage': 0
        }
        all_preds, all_targets = [], []
        n_batches = 0
        
        for batch in train_loader:
            spikes = batch['spikes'].to(device)
            spikes_raw = batch['spikes_raw'].to(device)
            targets = batch['kinematics'].to(device)
            
            # Forward
            output = model(spikes, spikes_raw, targets)
            loss = output['total_loss']
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            epoch_metrics['loss'] += loss.item()
            epoch_metrics['kin_loss'] += output['kinematics_loss'].item()
            epoch_metrics['recon_loss'] += output['reconstruction_loss'].item()
            epoch_metrics['perplexity'] += output['perplexity'].item()
            epoch_metrics['codebook_usage'] += output['codebook_usage']
            
            all_preds.append(output['kinematics_pred'].detach().cpu())
            all_targets.append(targets.cpu())
            n_batches += 1
        
        scheduler.step()
        
        # Compute epoch metrics
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
        history['kinematics_loss'].append(epoch_metrics['kin_loss'])
        history['reconstruction_loss'].append(epoch_metrics['recon_loss'])
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_metrics['r2'])
        history['perplexity'].append(epoch_metrics['perplexity'])
        history['codebook_usage'].append(epoch_metrics['codebook_usage'])
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{config.epochs} | "
                  f"Loss: {epoch_metrics['loss']:.4f} | "
                  f"Kin: {epoch_metrics['kin_loss']:.4f} | "
                  f"Recon: {epoch_metrics['recon_loss']:.4f} | "
                  f"Train R²: {train_r2:.4f} | "
                  f"Val R²: {val_metrics['r2']:.4f} | "
                  f"Perp: {epoch_metrics['perplexity']:.0f}")
        
        # Save best model
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'config': config,
                'metrics': {'val_r2': best_val_r2}
            }, SAVE_DIR / 'exp14_fsq_best.pt')
    
    # Final save
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, SAVE_DIR / 'exp14_fsq_final.pt')
    
    # Save history
    with open(SAVE_DIR / 'exp14_fsq_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history, best_val_r2


def evaluate(model, data_loader, device):
    """Evaluate model on validation/test set."""
    model.eval()
    
    total_loss, total_kin, total_recon = 0, 0, 0
    all_preds, all_targets = [], []
    n_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            spikes = batch['spikes'].to(device)
            spikes_raw = batch['spikes_raw'].to(device)
            targets = batch['kinematics'].to(device)
            
            output = model(spikes, spikes_raw, targets)
            
            total_loss += output['total_loss'].item()
            total_kin += output['kinematics_loss'].item()
            total_recon += output['reconstruction_loss'].item()
            
            all_preds.append(output['kinematics_pred'].cpu())
            all_targets.append(targets.cpu())
            n_batches += 1
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    return {
        'loss': total_loss / n_batches,
        'kinematics_loss': total_kin / n_batches,
        'reconstruction_loss': total_recon / n_batches,
        'r2': r2_score(all_targets, all_preds)
    }


# ============================================================
# Ablation Studies
# ============================================================

def run_ablation_studies(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    device: torch.device
):
    """
    Run ablation studies to validate theoretical claims:
    1. FSQ vs VQ (topology preservation)
    2. λ sweep (reconstruction weight)
    3. FSQ levels configuration
    """
    spike_counts_train, velocities_train = train_data
    spike_counts_val, velocities_val = val_data
    
    results = {}
    
    # ============ ABLATION 1: Reconstruction Weight λ ============
    print("\n" + "=" * 70)
    print("ABLATION 1: Reconstruction Weight (λ)")
    print("=" * 70)
    
    lambdas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
    
    for lam in lambdas:
        print(f"\nλ = {lam}")
        
        config = FSQConfig(
            reconstruction_weight=lam,
            epochs=50,  # Quick ablation
            n_channels=spike_counts_train.shape[1]
        )
        
        train_dataset = SlidingWindowDataset(
            spike_counts_train, velocities_train,
            window_size=config.n_bins
        )
        val_dataset = SlidingWindowDataset(
            spike_counts_val, velocities_val,
            window_size=config.n_bins
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size,
            shuffle=False, num_workers=0
        )
        
        model = CausalTransformerFSQVAE(config)
        _, best_r2 = train_fsq_vae(model, train_loader, val_loader, config, device)
        
        results[f'lambda_{lam}'] = best_r2
        print(f"  → Best Val R²: {best_r2:.4f}")
    
    # ============ ABLATION 2: FSQ Levels ============
    print("\n" + "=" * 70)
    print("ABLATION 2: FSQ Levels Configuration")
    print("=" * 70)
    
    level_configs = [
        (8, 8, 8),           # 512 codes (coarse)
        (8, 5, 5, 5),        # 1000 codes (baseline)
        (8, 8, 8, 8),        # 4096 codes (fine)
        (5, 5, 5, 5, 5),     # 3125 codes (higher dim)
    ]
    
    for levels in level_configs:
        print(f"\nFSQ levels: {levels} ({np.prod(levels)} codes)")
        
        config = FSQConfig(
            fsq_levels=levels,
            epochs=50,
            n_channels=spike_counts_train.shape[1]
        )
        
        train_dataset = SlidingWindowDataset(
            spike_counts_train, velocities_train,
            window_size=config.n_bins
        )
        val_dataset = SlidingWindowDataset(
            spike_counts_val, velocities_val,
            window_size=config.n_bins
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size,
            shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size,
            shuffle=False, num_workers=0
        )
        
        model = CausalTransformerFSQVAE(config)
        _, best_r2 = train_fsq_vae(model, train_loader, val_loader, config, device)
        
        results[f'levels_{levels}'] = best_r2
        print(f"  → Best Val R²: {best_r2:.4f}")
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading MC_Maze dataset...")
    try:
        from phantomx.data.mc_maze_loader import load_mc_maze_from_nwb
        spike_counts, kinematics = load_mc_maze_from_nwb(str(DATA_PATH))
        velocities = kinematics[:, 2:4]  # Extract vx, vy
        print(f"Loaded: {spike_counts.shape[0]} samples, {spike_counts.shape[1]} channels")
    except Exception as e:
        print(f"Could not load NWB: {e}")
        print("Generating synthetic data for testing...")
        
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 10000
        n_channels = 142
        
        # Synthetic spike counts (Poisson)
        rates = np.random.uniform(0.5, 5, n_channels)
        spike_counts = np.random.poisson(rates, (n_samples, n_channels)).astype(np.float32)
        
        # Synthetic velocities (correlated with some channels)
        W = np.random.randn(n_channels, 2) * 0.1
        velocities = spike_counts @ W + np.random.randn(n_samples, 2) * 0.5
        velocities = velocities.astype(np.float32)
    
    # Train/val split (80/20)
    n_train = int(len(spike_counts) * 0.8)
    
    spike_train = spike_counts[:n_train]
    spike_val = spike_counts[n_train:]
    vel_train = velocities[:n_train]
    vel_val = velocities[n_train:]
    
    print(f"Train: {len(spike_train)}, Val: {len(spike_val)}")
    
    # Configuration
    config = FSQConfig(n_channels=spike_counts.shape[1])
    
    # Create datasets
    train_dataset = SlidingWindowDataset(
        spike_train, vel_train, window_size=config.n_bins
    )
    val_dataset = SlidingWindowDataset(
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
    
    # Create model
    model = CausalTransformerFSQVAE(config)
    
    # Train
    print("\n" + "=" * 70)
    print("MAIN EXPERIMENT: FSQ-VAE with Dual-Head Decoder")
    print("=" * 70)
    
    history, best_r2 = train_fsq_vae(
        model, train_loader, val_loader, config, device
    )
    
    # Print baseline comparison
    print(compute_baseline_comparison(best_r2, lstm_r2=0.78))
    
    # Optional: Run ablation studies
    run_ablations = False  # Set to True to run ablations
    
    if run_ablations:
        ablation_results = run_ablation_studies(
            (spike_train, vel_train),
            (spike_val, vel_val),
            device
        )
        
        print("\n" + "=" * 70)
        print("ABLATION SUMMARY")
        print("=" * 70)
        for key, r2 in ablation_results.items():
            print(f"  {key}: R² = {r2:.4f}")
    
    print("\n✓ Experiment 14 complete!")
    print(f"  Best model saved to: {SAVE_DIR / 'exp14_fsq_best.pt'}")


if __name__ == "__main__":
    main()
