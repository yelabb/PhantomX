"""
Experiment 6: Improved VQ-VAE Architecture

Modifications to reduce VQ information bottleneck:
1. Residual VQ (z_q = z_e + VQ(z_e) to allow some continuous info to pass)
2. Product Quantization (multiple VQ heads)
3. Lower commitment cost
4. Larger codebook
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
# Tokenizer
# ============================================================

class TemporalPatternTokenizer:
    def __init__(self, n_channels=142):
        self.n_channels = n_channels
    
    def tokenize(self, spikes):
        T, C = spikes.shape
        features = np.zeros((4, C), dtype=np.float32)
        features[0] = np.mean(spikes, axis=0)
        features[1] = np.mean(np.diff(spikes, axis=0), axis=0)
        features[2] = np.std(spikes, axis=0)
        features[3] = np.max(spikes, axis=0) - np.min(spikes, axis=0)
        return features


# ============================================================
# Improved VQ Components
# ============================================================

class ProductVectorQuantizer(nn.Module):
    """
    Product Quantization: Split embedding into M groups, quantize each separately.
    
    This allows K^M combinations instead of just K codes.
    For K=64, M=4: 64^4 = 16.7M possible combinations.
    """
    
    def __init__(self, num_codes=64, embedding_dim=64, num_heads=4, commitment_cost=0.1):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.commitment_cost = commitment_cost
        
        assert embedding_dim % num_heads == 0
        
        # Each head has its own codebook
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_codes, self.head_dim)
            for _ in range(num_heads)
        ])
        
        for emb in self.embeddings:
            emb.weight.data.uniform_(-1/num_codes, 1/num_codes)
    
    def forward(self, z_e):
        batch_size = z_e.size(0)
        
        # Split into heads
        z_splits = z_e.view(batch_size, self.num_heads, self.head_dim)  # [B, M, D/M]
        
        z_q_list = []
        indices_list = []
        
        for i, emb in enumerate(self.embeddings):
            z_head = z_splits[:, i, :]  # [B, D/M]
            
            # Find nearest code
            distances = torch.cdist(z_head, emb.weight)
            idx = distances.argmin(dim=-1)
            z_q_head = emb(idx)
            
            z_q_list.append(z_q_head)
            indices_list.append(idx)
        
        z_q = torch.cat(z_q_list, dim=-1)  # [B, D]
        indices = torch.stack(indices_list, dim=-1)  # [B, M]
        
        # Losses
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        
        # Straight-through
        z_q = z_e + (z_q - z_e).detach()
        
        # Perplexity per head
        perplexities = []
        for i in range(self.num_heads):
            encodings = F.one_hot(indices_list[i], self.num_codes).float()
            avg_probs = encodings.mean(0)
            perp = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            perplexities.append(perp)
        
        return z_q, {
            'indices': indices,
            'codebook_loss': codebook_loss,
            'commitment_loss': self.commitment_cost * commitment_loss,
            'perplexity': torch.mean(torch.stack(perplexities))
        }


class ResidualVQVAE(nn.Module):
    """
    VQ-VAE with residual connection around VQ layer.
    
    z_out = α * z_q + (1 - α) * z_e
    
    This allows some continuous information to bypass VQ.
    """
    
    def __init__(
        self,
        n_features=4,
        n_channels=142,
        embedding_dim=64,
        num_codes=64,
        num_heads=4,
        vq_weight=0.8,  # How much of z_q to use (rest is z_e)
        output_dim=2
    ):
        super().__init__()
        
        self.vq_weight = vq_weight
        input_dim = n_features * n_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embedding_dim)
        )
        
        # Product VQ
        self.vq = ProductVectorQuantizer(num_codes, embedding_dim, num_heads)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x, targets=None):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        # Encode
        z_e = self.encoder(x_flat)
        
        # Quantize
        z_q, vq_info = self.vq(z_e)
        
        # Residual connection
        z_combined = self.vq_weight * z_q + (1 - self.vq_weight) * z_e
        
        # Decode
        velocity_pred = self.decoder(z_combined)
        
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
    def __init__(self, spike_counts, velocities, window_size=10):
        self.tokenizer = TemporalPatternTokenizer()
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

def train_and_evaluate(model, train_loader, val_loader, test_loader, device, epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_state = None
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            tokens = batch['tokens'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(tokens, velocity)
            output['total_loss'].backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(device)
                velocity = batch['velocity'].to(device)
                output = model(tokens, velocity)
                val_loss += output['total_loss'].item()
                val_preds.append(output['velocity_pred'].cpu().numpy())
                val_targets.append(velocity.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_r2 = r2_score(np.concatenate(val_targets), np.concatenate(val_preds))
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f}, val_r2={val_r2:.4f}")
        
        if patience >= 15:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    model.eval()
    
    all_preds, all_targets = [], []
    all_indices = []
    with torch.no_grad():
        for batch in test_loader:
            tokens = batch['tokens'].to(device)
            output = model(tokens)
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
        'indices': indices
    }


def main():
    DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"
    WINDOW_SIZE = 10
    
    print("=" * 60)
    print("Experiment 6: Improved VQ-VAE Architecture")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading MC_Maze data...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_dataset = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    dataset = TemporalPatternDataset(
        mc_dataset.spike_counts,
        mc_dataset.velocities,
        window_size=WINDOW_SIZE
    )
    
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_loader = DataLoader(Subset(dataset, range(n_train)), batch_size=64, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(n_train, n_train + n_val)), batch_size=64)
    test_loader = DataLoader(Subset(dataset, range(n_train + n_val, n)), batch_size=64)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = {}
    
    # ========================================
    # Config 1: Product VQ (4 heads, 64 codes each = 16M combinations)
    # ========================================
    print("\n[2/4] Testing Product VQ (4 heads × 64 codes)...")
    model1 = ResidualVQVAE(
        num_codes=64,
        num_heads=4,
        vq_weight=1.0  # Pure VQ
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    results['ProductVQ'] = train_and_evaluate(model1, train_loader, val_loader, test_loader, device)
    print(f"  Test R²: {results['ProductVQ']['r2']:.4f}")
    
    # ========================================
    # Config 2: Residual VQ (80% VQ, 20% continuous)
    # ========================================
    print("\n[3/4] Testing Residual VQ (80% discrete, 20% continuous)...")
    model2 = ResidualVQVAE(
        num_codes=64,
        num_heads=4,
        vq_weight=0.8
    ).to(device)
    
    results['ResidualVQ_0.8'] = train_and_evaluate(model2, train_loader, val_loader, test_loader, device)
    print(f"  Test R²: {results['ResidualVQ_0.8']['r2']:.4f}")
    
    # ========================================
    # Config 3: Mostly continuous (50% VQ, 50% continuous)
    # ========================================
    print("\n[4/4] Testing Residual VQ (50% discrete, 50% continuous)...")
    model3 = ResidualVQVAE(
        num_codes=64,
        num_heads=4,
        vq_weight=0.5
    ).to(device)
    
    results['ResidualVQ_0.5'] = train_and_evaluate(model3, train_loader, val_loader, test_loader, device)
    print(f"  Test R²: {results['ResidualVQ_0.5']['r2']:.4f}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 6 RESULTS")
    print("=" * 60)
    print(f"{'Config':<20} | {'R²':>10} | {'R² vx':>10} | {'R² vy':>10}")
    print("-" * 60)
    
    for name, res in results.items():
        print(f"{name:<20} | {res['r2']:>10.4f} | {res['r2_vx']:>10.4f} | {res['r2_vy']:>10.4f}")
    
    # Codebook analysis for best model
    best_name = max(results, key=lambda k: results[k]['r2'])
    best_indices = results[best_name]['indices']
    
    print("-" * 60)
    print(f"Best: {best_name} (R² = {results[best_name]['r2']:.4f})")
    
    # Count unique code combinations
    if best_indices.ndim == 2:
        unique_combos = len(set(map(tuple, best_indices)))
        print(f"  Unique code combinations: {unique_combos}")
        for h in range(best_indices.shape[1]):
            unique_per_head = len(np.unique(best_indices[:, h]))
            print(f"  Head {h}: {unique_per_head}/64 codes used")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
