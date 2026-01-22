"""
Experiment 5: TemporalPattern VQ-VAE

Combine the best tokenization (TemporalPattern) with VQ-VAE for learned discrete representations.

Architecture:
    Spikes [T, 142] → TemporalPattern [4, 142] → Encoder → z_e → VQ → z_q → Decoder → Velocity [2]
                                                                    ↓
                                                               Codebook [256, 64]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import r2_score
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phantomx.data import MCMazeDataset
from phantomx.tokenizer import SpikeTokenizer


# ============================================================
# TemporalPattern Tokenizer
# ============================================================

class TemporalPatternTokenizer:
    """Extract temporal features per channel."""
    
    def __init__(self, n_channels=142):
        self.n_channels = n_channels
    
    def tokenize(self, spikes):
        """
        Extract temporal features.
        
        Args:
            spikes: [T, C] spike counts over window
            
        Returns:
            features: [4, C] temporal features per channel
        """
        T, C = spikes.shape
        
        features = np.zeros((4, C), dtype=np.float32)
        features[0] = np.mean(spikes, axis=0)  # Mean
        features[1] = np.mean(np.diff(spikes, axis=0), axis=0)  # Derivative
        features[2] = np.std(spikes, axis=0)  # Variability
        features[3] = np.max(spikes, axis=0) - np.min(spikes, axis=0)  # Range
        
        return features


# ============================================================
# VQ-VAE Components
# ============================================================

class VectorQuantizer(nn.Module):
    """Vector quantization with EMA updates."""
    
    def __init__(self, num_codes=256, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_codes, 1/num_codes)
    
    def forward(self, z_e):
        """
        Args:
            z_e: [batch, embedding_dim] encoded latents
            
        Returns:
            z_q: [batch, embedding_dim] quantized latents
            info: dict with indices, losses, perplexity
        """
        # Compute distances
        distances = torch.cdist(z_e, self.embedding.weight)  # [batch, num_codes]
        
        # Get nearest codes
        indices = distances.argmin(dim=-1)  # [batch]
        z_q = self.embedding(indices)  # [batch, embedding_dim]
        
        # Compute losses
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        
        # Straight-through gradient
        z_q = z_e + (z_q - z_e).detach()
        
        # Compute perplexity
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, {
            'indices': indices,
            'codebook_loss': codebook_loss,
            'commitment_loss': self.commitment_cost * commitment_loss,
            'perplexity': perplexity
        }


class TemporalPatternVQVAE(nn.Module):
    """
    VQ-VAE with TemporalPattern tokenization.
    
    Architecture:
        Input [4, 142] → Flatten → Encoder MLP → z_e [64] → VQ → z_q [64] → Decoder MLP → Velocity [2]
    """
    
    def __init__(
        self,
        n_features=4,
        n_channels=142,
        embedding_dim=64,
        num_codes=256,
        hidden_dims=[256, 128],
        output_dim=2
    ):
        super().__init__()
        
        input_dim = n_features * n_channels
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, embedding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_codes, embedding_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = embedding_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x, targets=None):
        """
        Args:
            x: [batch, 4, 142] temporal pattern tokens
            targets: [batch, 2] velocity targets (optional)
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # [batch, 568]
        
        # Encode
        z_e = self.encoder(x_flat)  # [batch, 64]
        
        # Quantize
        z_q, vq_info = self.vq(z_e)
        
        # Decode
        velocity_pred = self.decoder(z_q)  # [batch, 2]
        
        output = {
            'velocity_pred': velocity_pred,
            'z_e': z_e,
            'z_q': z_q,
            **vq_info
        }
        
        if targets is not None:
            recon_loss = F.mse_loss(velocity_pred, targets)
            total_loss = recon_loss + vq_info['codebook_loss'] + vq_info['commitment_loss']
            output['recon_loss'] = recon_loss
            output['total_loss'] = total_loss
        
        return output


# ============================================================
# Dataset
# ============================================================

class TemporalPatternDataset(Dataset):
    """Dataset with pre-computed temporal pattern tokens."""
    
    def __init__(self, spike_counts, velocities, window_size=10):
        self.tokenizer = TemporalPatternTokenizer()
        self.window_size = window_size
        
        # Pre-compute all tokens
        n = len(spike_counts) - window_size + 1
        self.tokens = []
        self.velocities = []
        
        for i in range(n):
            window = spike_counts[i:i + window_size]
            self.tokens.append(self.tokenizer.tokenize(window))
            self.velocities.append(velocities[i + window_size - 1])
        
        self.tokens = np.stack(self.tokens)
        self.velocities = np.stack(self.velocities)
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        return {
            'tokens': torch.tensor(self.tokens[idx], dtype=torch.float32),
            'velocity': torch.tensor(self.velocities[idx], dtype=torch.float32)
        }


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_recon = 0
    total_perp = 0
    
    for batch in loader:
        tokens = batch['tokens'].to(device)
        velocity = batch['velocity'].to(device)
        
        optimizer.zero_grad()
        output = model(tokens, velocity)
        loss = output['total_loss']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += output['recon_loss'].item()
        total_perp += output['perplexity'].item()
    
    n = len(loader)
    return total_loss / n, total_recon / n, total_perp / n


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in loader:
            tokens = batch['tokens'].to(device)
            velocity = batch['velocity'].to(device)
            
            output = model(tokens, velocity)
            total_loss += output['total_loss'].item()
            
            all_preds.append(output['velocity_pred'].cpu().numpy())
            all_targets.append(velocity.cpu().numpy())
    
    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    r2 = r2_score(targets, preds)
    r2_vx = r2_score(targets[:, 0], preds[:, 0])
    r2_vy = r2_score(targets[:, 1], preds[:, 1])
    
    return {
        'loss': total_loss / len(loader),
        'r2': r2,
        'r2_vx': r2_vx,
        'r2_vy': r2_vy
    }


def main():
    DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"
    WINDOW_SIZE = 10
    EPOCHS = 100
    
    print("=" * 60)
    print("Experiment 5: TemporalPattern VQ-VAE")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading MC_Maze data...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_dataset = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    # Create temporal pattern dataset
    print("[2/4] Creating temporal pattern tokens...")
    dataset = TemporalPatternDataset(
        mc_dataset.spike_counts,
        mc_dataset.velocities,
        window_size=WINDOW_SIZE
    )
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Token shape: {dataset.tokens.shape}")
    
    # Split
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_loader = DataLoader(Subset(dataset, range(n_train)), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(n_train, n_train + n_val)), batch_size=64)
    test_loader = DataLoader(Subset(dataset, range(n_train + n_val, n)), batch_size=64)
    
    # Create model
    print("[3/4] Creating VQ-VAE model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TemporalPatternVQVAE(
        n_features=4,
        n_channels=142,
        embedding_dim=64,
        num_codes=256
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Train
    print("[4/4] Training...")
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_recon, train_perp = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step(val_metrics['loss'])
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={train_loss:.4f}, recon={train_recon:.4f}, "
                  f"perp={train_perp:.1f}, val_r2={val_metrics['r2']:.4f}")
        
        if patience_counter >= 15:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"  Test R² (vx):      {test_metrics['r2_vx']:.4f}")
    print(f"  Test R² (vy):      {test_metrics['r2_vy']:.4f}")
    print(f"  Test R² (overall): {test_metrics['r2']:.4f}")
    
    # Codebook analysis
    model.eval()
    all_indices = []
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            output = model(tokens)
            all_indices.append(output['indices'].cpu().numpy())
    
    all_indices = np.concatenate(all_indices)
    unique_codes = len(np.unique(all_indices))
    
    print(f"\n  Codebook utilization: {unique_codes}/256 ({100*unique_codes/256:.1f}%)")
    
    print("=" * 60)
    
    if test_metrics['r2'] >= 0.5:
        print("\n✓ VQ-VAE with TemporalPattern achieves strong decoding!")
    else:
        print("\n⚠ VQ-VAE bottleneck may limit performance")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'n_features': 4,
            'n_channels': 142,
            'embedding_dim': 64,
            'num_codes': 256,
            'window_size': WINDOW_SIZE
        },
        'metrics': test_metrics
    }, 'models/temporal_vqvae.pt')
    print("\nModel saved to models/temporal_vqvae.pt")


if __name__ == '__main__':
    main()
