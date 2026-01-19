"""
Experiment 8: Addressing Codebook Collapse

The VQ layer is collapsing to 7-8 codes out of 512+. This is codebook collapse.
Solutions:
1. EMA updates (smoother codebook training)
2. Random restart for dead codes  
3. KMeans initialization
4. Entropy regularization (encourage using all codes)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phantomx.data import MCMazeDataset
from phantomx.tokenizer import SpikeTokenizer


# ============================================================
# EMA Vector Quantizer with Dead Code Revival
# ============================================================

class EMAVectorQuantizer(nn.Module):
    """
    Exponential Moving Average VQ with dead code revival.
    Based on VQ-VAE-2 improvements.
    """
    
    def __init__(self, num_codes=256, embedding_dim=64, decay=0.99, 
                 commitment_cost=0.25, epsilon=1e-5, dead_threshold=0.01):
        super().__init__()
        
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        self.dead_threshold = dead_threshold
        
        # Initialize embeddings with small random values
        embeddings = torch.randn(num_codes, embedding_dim)
        self.register_buffer('embeddings', embeddings)
        self.register_buffer('cluster_size', torch.ones(num_codes))
        self.register_buffer('ema_w', embeddings.clone())
        
        self._initialized = False
    
    def _init_embeddings(self, z_e):
        """Initialize codebook with k-means on first batch."""
        print(f"    Initializing codebook with k-means on {z_e.shape[0]} samples...")
        z_np = z_e.detach().cpu().numpy()
        
        # Subsample if too large
        if len(z_np) > 10000:
            idx = np.random.choice(len(z_np), 10000, replace=False)
            z_np = z_np[idx]
        
        kmeans = KMeans(n_clusters=min(self.num_codes, len(z_np)), n_init=1, max_iter=50)
        kmeans.fit(z_np)
        
        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        if len(centers) < self.num_codes:
            # Pad with random perturbations
            extra = self.num_codes - len(centers)
            noise = torch.randn(extra, self.embedding_dim) * 0.01
            centers = torch.cat([centers, centers[:extra] + noise])
        
        self.embeddings.copy_(centers)
        self.ema_w.copy_(centers)
        self._initialized = True
    
    def forward(self, z_e):
        # Init codebook on first forward pass
        if not self._initialized and self.training:
            self._init_embeddings(z_e)
        
        # Find nearest code
        distances = torch.cdist(z_e, self.embeddings)
        indices = distances.argmin(dim=-1)
        
        # Get quantized vectors
        z_q = F.embedding(indices, self.embeddings)
        
        if self.training:
            # EMA update
            encodings = F.one_hot(indices, self.num_codes).float()
            n_j = encodings.sum(0)  # [K]
            
            # Update cluster sizes
            self.cluster_size.data = self.decay * self.cluster_size + (1 - self.decay) * n_j
            
            # Update embeddings
            dw = encodings.T @ z_e  # [K, D]
            self.ema_w.data = self.decay * self.ema_w + (1 - self.decay) * dw
            
            # Normalize
            n = self.cluster_size.clamp(min=self.epsilon)
            self.embeddings.data = self.ema_w / n.unsqueeze(-1)
            
            # Revive dead codes
            dead_mask = self.cluster_size < self.dead_threshold
            if dead_mask.any():
                n_dead = dead_mask.sum().item()
                if n_dead > 0:
                    # Sample random z_e to reinit dead codes
                    rand_idx = torch.randperm(z_e.size(0))[:n_dead]
                    self.embeddings.data[dead_mask] = z_e[rand_idx].detach()
                    self.cluster_size[dead_mask] = 1.0
        
        # Commitment loss (no codebook loss - handled by EMA)
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        
        # Straight-through
        z_q = z_e + (z_q - z_e).detach()
        
        # Compute metrics
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, {
            'indices': indices,
            'commitment_loss': self.commitment_cost * commitment_loss,
            'perplexity': perplexity,
            'n_active': (avg_probs > 0.001).sum()
        }


class EntropyRegularizedVQ(nn.Module):
    """VQ with entropy regularization to prevent collapse."""
    
    def __init__(self, num_codes=256, embedding_dim=64, commitment_cost=0.25, entropy_weight=0.1):
        super().__init__()
        
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.entropy_weight = entropy_weight
        
        self.embeddings = nn.Embedding(num_codes, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_codes, 1/num_codes)
    
    def forward(self, z_e):
        # Compute soft assignments (Gumbel-softmax style distances)
        distances = torch.cdist(z_e, self.embeddings.weight)
        
        # Hard assignment
        indices = distances.argmin(dim=-1)
        z_q = self.embeddings(indices)
        
        # Standard losses
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        
        # Entropy regularization: encourage uniform code usage
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(0) + 1e-10
        entropy = -torch.sum(avg_probs * torch.log(avg_probs))
        max_entropy = np.log(self.num_codes)
        
        # Negative entropy = loss (we want HIGH entropy = uniform usage)
        entropy_loss = -self.entropy_weight * (entropy / max_entropy)
        
        # Straight-through
        z_q = z_e + (z_q - z_e).detach()
        
        perplexity = torch.exp(entropy)
        
        return z_q, {
            'indices': indices,
            'codebook_loss': codebook_loss,
            'commitment_loss': self.commitment_cost * commitment_loss,
            'entropy_loss': entropy_loss,
            'perplexity': perplexity
        }


# ============================================================
# Model
# ============================================================

class RobustVQVAE(nn.Module):
    def __init__(self, n_channels=142, window_size=10, embedding_dim=128, 
                 num_codes=256, vq_type='ema'):
        super().__init__()
        
        input_dim = n_channels * window_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        )
        
        if vq_type == 'ema':
            self.vq = EMAVectorQuantizer(num_codes, embedding_dim)
        else:
            self.vq = EntropyRegularizedVQ(num_codes, embedding_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
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
            
            # Collect losses
            total_loss = recon_loss + vq_info['commitment_loss']
            if 'codebook_loss' in vq_info:
                total_loss += vq_info['codebook_loss']
            if 'entropy_loss' in vq_info:
                total_loss += vq_info['entropy_loss']
            
            output['recon_loss'] = recon_loss
            output['total_loss'] = total_loss
        
        return output


# ============================================================
# Dataset
# ============================================================

class SlidingWindowDataset(Dataset):
    def __init__(self, spike_counts, velocities, window_size=10):
        n = len(spike_counts) - window_size + 1
        self.windows = np.stack([spike_counts[i:i+window_size] for i in range(n)])
        self.velocities = velocities[window_size-1:window_size-1+n]
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
    
    best_val_r2 = -float('inf')
    best_state = None
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window, velocity)
            output['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += output['total_loss'].item()
        
        scheduler.step()
        train_loss /= len(train_loader)
        
        model.eval()
        val_preds, val_targets = [], []
        perp_sum = 0
        n_active_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                velocity = batch['velocity'].to(device)
                output = model(window, velocity)
                perp_sum += output['perplexity'].item()
                n_active_sum += output.get('n_active', output['perplexity']).item()
                val_preds.append(output['velocity_pred'].cpu().numpy())
                val_targets.append(velocity.cpu().numpy())
        
        perp_avg = perp_sum / len(val_loader)
        n_active_avg = n_active_sum / len(val_loader)
        val_r2 = r2_score(np.concatenate(val_targets), np.concatenate(val_preds))
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_r2={val_r2:.4f}, "
                  f"perp={perp_avg:.1f}, active={n_active_avg:.0f}")
        
        if patience >= 20:
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
        'n_codes_used': len(np.unique(indices)),
        'indices': indices
    }


def main():
    DATA_PATH = "c:/Users/guzzi/Desktop/Projects/DEV-ACTIF/NeuraLink/PhantomLink/data/raw/mc_maze.nwb"
    WINDOW_SIZE = 10
    
    print("=" * 60)
    print("Experiment 8: Addressing Codebook Collapse")
    print("=" * 60)
    
    # Load data
    print("\n[1/3] Loading MC_Maze data...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_dataset = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    dataset = SlidingWindowDataset(
        mc_dataset.spike_counts,
        mc_dataset.velocities,
        window_size=WINDOW_SIZE
    )
    
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    
    train_loader = DataLoader(Subset(dataset, range(n_train)), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(dataset, range(n_train, n_train + n_val)), batch_size=128)
    test_loader = DataLoader(Subset(dataset, range(n_train + n_val, n)), batch_size=128)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    
    # ========================================
    # Config 1: EMA VQ with k-means init
    # ========================================
    print("\n[2/3] Testing EMA VQ with k-means init and dead code revival...")
    
    model1 = RobustVQVAE(num_codes=256, vq_type='ema').to(device)
    print(f"  Parameters: {sum(p.numel() for p in model1.parameters()):,}")
    
    results['EMA_VQ'] = train_and_evaluate(model1, train_loader, val_loader, test_loader, device)
    print(f"  Test R²: {results['EMA_VQ']['r2']:.4f}, Codes used: {results['EMA_VQ']['n_codes_used']}")
    
    # ========================================
    # Config 2: Entropy-regularized VQ
    # ========================================
    print("\n[3/3] Testing Entropy-regularized VQ...")
    
    model2 = RobustVQVAE(num_codes=256, vq_type='entropy').to(device)
    print(f"  Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    
    results['Entropy_VQ'] = train_and_evaluate(model2, train_loader, val_loader, test_loader, device)
    print(f"  Test R²: {results['Entropy_VQ']['r2']:.4f}, Codes used: {results['Entropy_VQ']['n_codes_used']}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 8 RESULTS")
    print("=" * 60)
    print(f"{'Config':<20} | {'R²':>8} | {'R² vx':>8} | {'R² vy':>8} | {'Codes':>6}")
    print("-" * 60)
    
    for name, res in results.items():
        print(f"{name:<20} | {res['r2']:>8.4f} | {res['r2_vx']:>8.4f} | {res['r2_vy']:>8.4f} | {res['n_codes_used']:>6}")
    
    best_name = max(results, key=lambda k: results[k]['r2'])
    print("-" * 60)
    print(f"Best: {best_name} (R² = {results[best_name]['r2']:.4f})")
    
    # Analyze code distribution for best
    indices = results[best_name]['indices']
    unique, counts = np.unique(indices, return_counts=True)
    top10 = sorted(zip(counts, unique), reverse=True)[:10]
    print(f"\nTop 10 codes: {[f'{c}({cnt})' for cnt, c in top10]}")
    
    print("=" * 60)
    
    # Save best model
    if results[best_name]['r2'] > 0.55:
        torch.save(model1.state_dict() if best_name == 'EMA_VQ' else model2.state_dict(), 
                   'models/exp8_best_vqvae.pt')
        print(f"\nSaved best model to models/exp8_best_vqvae.pt")


if __name__ == '__main__':
    main()
