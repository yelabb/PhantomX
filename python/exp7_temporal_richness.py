"""
Experiment 7: Richer Temporal Representations

Key insight: Our TemporalPattern tokenizer compresses a 10-step window into just 4 summary stats.
This loses detailed temporal dynamics. Let's try:

1. Full temporal flattening (keep all 10 timesteps)
2. Channel-specific temporal features (preserve channel identity better)
3. Larger embedding dimension for VQ
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import r2_score
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phantomx.data import MCMazeDataset
from phantomx.tokenizer import SpikeTokenizer


# ============================================================
# Tokenizers
# ============================================================

class FullTemporalTokenizer:
    """Keep all temporal information, just reshape."""
    def __init__(self, n_channels=142, window_size=10):
        self.n_channels = n_channels
        self.window_size = window_size
    
    def tokenize(self, spikes):
        # spikes: [T, C] -> [T, C] (identity, we'll flatten in model)
        return spikes.astype(np.float32)


class ChannelTemporalTokenizer:
    """Rich per-channel temporal features."""
    def __init__(self, n_channels=142, n_temporal_bins=5):
        self.n_channels = n_channels
        self.n_bins = n_temporal_bins
    
    def tokenize(self, spikes):
        T, C = spikes.shape
        bin_size = T // self.n_bins
        
        features = []
        for i in range(self.n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < self.n_bins - 1 else T
            segment = spikes[start:end]
            features.append(np.mean(segment, axis=0))  # Mean in each bin
        
        # Also add global stats
        features.append(np.std(spikes, axis=0))
        features.append(np.max(spikes, axis=0) - np.min(spikes, axis=0))
        
        return np.stack(features, axis=0).astype(np.float32)  # [n_bins+2, C]


# ============================================================
# VQ Layer
# ============================================================

class VectorQuantizer(nn.Module):
    def __init__(self, num_codes=256, embedding_dim=64, commitment_cost=0.1):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_codes, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_codes, 1/num_codes)
    
    def forward(self, z_e):
        distances = torch.cdist(z_e, self.embeddings.weight)
        indices = distances.argmin(dim=-1)
        z_q = self.embeddings(indices)
        
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        
        z_q = z_e + (z_q - z_e).detach()
        
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, {
            'indices': indices,
            'codebook_loss': codebook_loss,
            'commitment_loss': self.commitment_cost * commitment_loss,
            'perplexity': perplexity
        }


# ============================================================
# Models
# ============================================================

class FullTemporalVQVAE(nn.Module):
    """VQ-VAE that keeps all temporal information."""
    
    def __init__(self, n_channels=142, window_size=10, embedding_dim=128, num_codes=512):
        super().__init__()
        
        input_dim = n_channels * window_size  # Full flattening
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
        self.vq = VectorQuantizer(num_codes, embedding_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x, targets=None):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        z_e = self.encoder(x_flat)
        z_q, vq_info = self.vq(z_e)
        velocity_pred = self.decoder(z_q)
        
        output = {'velocity_pred': velocity_pred, 'z_e': z_e, 'z_q': z_q, **vq_info}
        
        if targets is not None:
            recon_loss = F.mse_loss(velocity_pred, targets)
            total_loss = recon_loss + vq_info['codebook_loss'] + vq_info['commitment_loss']
            output['recon_loss'] = recon_loss
            output['total_loss'] = total_loss
        
        return output


class ChannelTemporalVQVAE(nn.Module):
    """VQ-VAE with rich per-channel temporal features."""
    
    def __init__(self, n_channels=142, n_features=7, embedding_dim=128, num_codes=512):
        super().__init__()
        
        input_dim = n_features * n_channels
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )
        
        self.vq = VectorQuantizer(num_codes, embedding_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x, targets=None):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        z_e = self.encoder(x_flat)
        z_q, vq_info = self.vq(z_e)
        velocity_pred = self.decoder(z_q)
        
        output = {'velocity_pred': velocity_pred, 'z_e': z_e, 'z_q': z_q, **vq_info}
        
        if targets is not None:
            recon_loss = F.mse_loss(velocity_pred, targets)
            total_loss = recon_loss + vq_info['codebook_loss'] + vq_info['commitment_loss']
            output['recon_loss'] = recon_loss
            output['total_loss'] = total_loss
        
        return output


class TemporalConvVQVAE(nn.Module):
    """VQ-VAE with 1D conv encoder to capture temporal patterns."""
    
    def __init__(self, n_channels=142, window_size=10, embedding_dim=128, num_codes=512):
        super().__init__()
        
        # 1D conv treats time as sequence, channels as features
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(n_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # [B, 128, 1]
        )
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
        
        self.vq = VectorQuantizer(num_codes, embedding_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x, targets=None):
        # x: [B, T, C] -> [B, C, T] for Conv1d
        x = x.permute(0, 2, 1)
        
        h = self.conv_encoder(x).squeeze(-1)  # [B, 128]
        z_e = self.fc_encoder(h)
        z_q, vq_info = self.vq(z_e)
        velocity_pred = self.decoder(z_q)
        
        output = {'velocity_pred': velocity_pred, 'z_e': z_e, 'z_q': z_q, **vq_info}
        
        if targets is not None:
            recon_loss = F.mse_loss(velocity_pred, targets)
            total_loss = recon_loss + vq_info['codebook_loss'] + vq_info['commitment_loss']
            output['recon_loss'] = recon_loss
            output['total_loss'] = total_loss
        
        return output


# ============================================================
# Dataset
# ============================================================

class SlidingWindowDataset(Dataset):
    def __init__(self, spike_counts, velocities, window_size=10, tokenizer=None):
        self.window_size = window_size
        self.tokenizer = tokenizer
        
        n = len(spike_counts) - window_size + 1
        self.windows = []
        self.velocities = []
        
        for i in range(n):
            window = spike_counts[i:i + window_size]
            if tokenizer:
                window = tokenizer.tokenize(window)
            self.windows.append(window)
            self.velocities.append(velocities[i + window_size - 1])
        
        self.windows = np.stack(self.windows)
        self.velocities = np.stack(self.velocities)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return {
            'window': torch.tensor(self.windows[idx], dtype=torch.float32),
            'velocity': torch.tensor(self.velocities[idx], dtype=torch.float32)
        }


# ============================================================
# Training
# ============================================================

def train_and_evaluate(model, train_loader, val_loader, test_loader, device, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_state = None
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window, velocity)
            output['total_loss'].backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        perp_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                velocity = batch['velocity'].to(device)
                output = model(window, velocity)
                val_loss += output['total_loss'].item()
                perp_sum += output['perplexity'].item()
                val_preds.append(output['velocity_pred'].cpu().numpy())
                val_targets.append(velocity.cpu().numpy())
        
        val_loss /= len(val_loader)
        perp_avg = perp_sum / len(val_loader)
        val_r2 = r2_score(np.concatenate(val_targets), np.concatenate(val_preds))
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f}, val_r2={val_r2:.4f}, perp={perp_avg:.1f}")
        
        if patience >= 15:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    model.eval()
    
    all_preds, all_targets = [], []
    all_indices = []
    with torch.no_grad():
        for batch in test_loader:
            window = batch['window'].to(device)
            output = model(window)
            all_preds.append(output['velocity_pred'].cpu().numpy())
            all_targets.append(batch['velocity'].numpy())
            all_indices.append(output['indices'].cpu().numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    indices = np.concatenate(all_indices)
    
    return {
        'r2': r2_score(targets, preds),
        'r2_vx': r2_score(targets[:, 0], preds[:, 0]),
        'r2_vy': r2_score(targets[:, 1], preds[:, 1]),
        'n_codes_used': len(np.unique(indices))
    }


def main():
    DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"
    WINDOW_SIZE = 10
    
    print("=" * 60)
    print("Experiment 7: Richer Temporal Representations")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading MC_Maze data...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_dataset = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    
    # ========================================
    # Config 1: Full temporal (raw windows)
    # ========================================
    print("\n[2/5] Testing Full Temporal VQ-VAE (raw 10x142 windows)...")
    
    dataset1 = SlidingWindowDataset(
        mc_dataset.spike_counts,
        mc_dataset.velocities,
        window_size=WINDOW_SIZE,
        tokenizer=FullTemporalTokenizer()
    )
    
    n = len(dataset1)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_loader = DataLoader(Subset(dataset1, range(n_train)), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset1, range(n_train, n_train + n_val)), batch_size=64)
    test_loader = DataLoader(Subset(dataset1, range(n_train + n_val, n)), batch_size=64)
    
    model1 = FullTemporalVQVAE(n_channels=142, window_size=WINDOW_SIZE, num_codes=512).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    results['FullTemporal'] = train_and_evaluate(model1, train_loader, val_loader, test_loader, device)
    print(f"  Test R²: {results['FullTemporal']['r2']:.4f}")
    
    # ========================================
    # Config 2: Channel Temporal (binned features)
    # ========================================
    print("\n[3/5] Testing Channel Temporal VQ-VAE (5 bins + stats)...")
    
    dataset2 = SlidingWindowDataset(
        mc_dataset.spike_counts,
        mc_dataset.velocities,
        window_size=WINDOW_SIZE,
        tokenizer=ChannelTemporalTokenizer(n_temporal_bins=5)
    )
    
    train_loader = DataLoader(Subset(dataset2, range(n_train)), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset2, range(n_train, n_train + n_val)), batch_size=64)
    test_loader = DataLoader(Subset(dataset2, range(n_train + n_val, n)), batch_size=64)
    
    model2 = ChannelTemporalVQVAE(n_channels=142, n_features=7, num_codes=512).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    results['ChannelTemporal'] = train_and_evaluate(model2, train_loader, val_loader, test_loader, device)
    print(f"  Test R²: {results['ChannelTemporal']['r2']:.4f}")
    
    # ========================================
    # Config 3: Temporal Conv VQ-VAE
    # ========================================
    print("\n[4/5] Testing Temporal Conv VQ-VAE...")
    
    train_loader = DataLoader(Subset(dataset1, range(n_train)), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset1, range(n_train, n_train + n_val)), batch_size=64)
    test_loader = DataLoader(Subset(dataset1, range(n_train + n_val, n)), batch_size=64)
    
    model3 = TemporalConvVQVAE(n_channels=142, window_size=WINDOW_SIZE, num_codes=512).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model3.parameters()):,}")
    
    results['TemporalConv'] = train_and_evaluate(model3, train_loader, val_loader, test_loader, device)
    print(f"  Test R²: {results['TemporalConv']['r2']:.4f}")
    
    # ========================================
    # Config 4: Larger embedding + more codes
    # ========================================
    print("\n[5/5] Testing Full Temporal with 1024 codes, 256-dim embeddings...")
    
    model4 = FullTemporalVQVAE(n_channels=142, window_size=WINDOW_SIZE, embedding_dim=256, num_codes=1024).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model4.parameters()):,}")
    
    results['FullTemporal_Large'] = train_and_evaluate(model4, train_loader, val_loader, test_loader, device)
    print(f"  Test R²: {results['FullTemporal_Large']['r2']:.4f}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 7 RESULTS")
    print("=" * 60)
    print(f"{'Config':<22} | {'R²':>8} | {'R² vx':>8} | {'R² vy':>8} | {'Codes':>6}")
    print("-" * 60)
    
    for name, res in results.items():
        print(f"{name:<22} | {res['r2']:>8.4f} | {res['r2_vx']:>8.4f} | {res['r2_vy']:>8.4f} | {res['n_codes_used']:>6}")
    
    best_name = max(results, key=lambda k: results[k]['r2'])
    print("-" * 60)
    print(f"Best: {best_name} (R² = {results[best_name]['r2']:.4f})")
    print("=" * 60)
    
    # Save best model
    if results[best_name]['r2'] > 0.55:
        torch.save(model1.state_dict(), 'models/exp7_best_vqvae.pt')
        print(f"\nSaved best model to models/exp7_best_vqvae.pt")


if __name__ == '__main__':
    main()
