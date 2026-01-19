"""
Experiment 9: Push to R² ≥ 0.70

Current best: EMA VQ R² = 0.67
Strategies:
1. Larger batch size for better k-means init
2. Pre-train encoder without VQ, then finetune with VQ (progressive training)
3. Lower VQ commitment cost 
4. Deeper encoder
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
# EMA Vector Quantizer
# ============================================================

class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_codes=256, embedding_dim=128, decay=0.99, 
                 commitment_cost=0.1, epsilon=1e-5):
        super().__init__()
        
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.commitment_cost = commitment_cost
        self.epsilon = epsilon
        
        self.register_buffer('embeddings', torch.randn(num_codes, embedding_dim))
        self.register_buffer('cluster_size', torch.ones(num_codes))
        self.register_buffer('ema_w', self.embeddings.clone())
        self._initialized = False
    
    def init_from_data(self, z_all):
        """Initialize with k-means on full training data."""
        print(f"    K-means init with {len(z_all)} samples...")
        z_np = z_all.numpy()
        
        kmeans = KMeans(n_clusters=self.num_codes, n_init=10, max_iter=300)
        kmeans.fit(z_np)
        
        self.embeddings.copy_(torch.from_numpy(kmeans.cluster_centers_).float())
        self.ema_w.copy_(self.embeddings)
        self._initialized = True
        
        # Count how many samples per cluster
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        print(f"    Initial code usage: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    def forward(self, z_e):
        distances = torch.cdist(z_e, self.embeddings)
        indices = distances.argmin(dim=-1)
        z_q = F.embedding(indices, self.embeddings)
        
        if self.training and self._initialized:
            encodings = F.one_hot(indices, self.num_codes).float()
            n_j = encodings.sum(0)
            
            self.cluster_size.data = self.decay * self.cluster_size + (1 - self.decay) * n_j
            
            dw = encodings.T @ z_e
            self.ema_w.data = self.decay * self.ema_w + (1 - self.decay) * dw
            
            n = self.cluster_size.clamp(min=self.epsilon)
            self.embeddings.data = self.ema_w / n.unsqueeze(-1)
        
        commitment_loss = F.mse_loss(z_q.detach(), z_e)
        z_q = z_e + (z_q - z_e).detach()
        
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, {
            'indices': indices,
            'commitment_loss': self.commitment_cost * commitment_loss,
            'perplexity': perplexity
        }


# ============================================================
# Model with progressive training support
# ============================================================

class ProgressiveVQVAE(nn.Module):
    def __init__(self, n_channels=142, window_size=10, embedding_dim=128, num_codes=256):
        super().__init__()
        
        input_dim = n_channels * window_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, embedding_dim)
        )
        
        self.vq = EMAVectorQuantizer(num_codes, embedding_dim, commitment_cost=0.1)
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        
        self.use_vq = False  # Start without VQ
    
    def forward(self, x, targets=None):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        z_e = self.encoder(x_flat)
        
        if self.use_vq:
            z_q, vq_info = self.vq(z_e)
        else:
            z_q = z_e
            vq_info = {'commitment_loss': torch.tensor(0.0), 'perplexity': torch.tensor(0.0)}
        
        velocity_pred = self.decoder(z_q)
        
        output = {'velocity_pred': velocity_pred, 'z_e': z_e, 'z_q': z_q, **vq_info}
        
        if targets is not None:
            recon_loss = F.mse_loss(velocity_pred, targets)
            total_loss = recon_loss + vq_info['commitment_loss']
            output['recon_loss'] = recon_loss
            output['total_loss'] = total_loss
        
        return output
    
    def collect_embeddings(self, loader, device):
        """Collect all z_e for k-means init."""
        self.eval()
        embeddings = []
        with torch.no_grad():
            for batch in loader:
                window = batch['window'].to(device)
                z_e = self.encoder(window.view(window.size(0), -1))
                embeddings.append(z_e.cpu())
        return torch.cat(embeddings, dim=0)


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
# Progressive Training
# ============================================================

def train_progressive(model, train_loader, val_loader, device, pretrain_epochs=30, finetune_epochs=50):
    print("\n  Phase 1: Pre-training encoder without VQ...")
    model.use_vq = False
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_r2 = -float('inf')
    
    for epoch in range(pretrain_epochs):
        model.train()
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window, velocity)
            output['total_loss'].backward()
            optimizer.step()
        
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                output = model(window)
                val_preds.append(output['velocity_pred'].cpu().numpy())
                val_targets.append(batch['velocity'].numpy())
        
        val_r2 = r2_score(np.concatenate(val_targets), np.concatenate(val_preds))
        
        if (epoch + 1) % 10 == 0:
            print(f"    Pretrain Epoch {epoch+1}: val_r2={val_r2:.4f} (no VQ)")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
    
    print(f"  Best pretrain R²: {best_val_r2:.4f}")
    
    # Collect embeddings and init VQ
    print("\n  Phase 2: Initializing VQ codebook with k-means...")
    z_all = model.collect_embeddings(train_loader, device)
    model.vq.init_from_data(z_all)
    
    # Enable VQ and finetune
    print("\n  Phase 3: Finetuning with VQ...")
    model.use_vq = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)
    
    best_val_r2 = -float('inf')
    best_state = None
    patience = 0
    
    for epoch in range(finetune_epochs):
        model.train()
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window, velocity)
            output['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        val_preds, val_targets = [], []
        perp_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                output = model(window)
                perp_sum += output['perplexity'].item()
                val_preds.append(output['velocity_pred'].cpu().numpy())
                val_targets.append(batch['velocity'].numpy())
        
        perp_avg = perp_sum / len(val_loader)
        val_r2 = r2_score(np.concatenate(val_targets), np.concatenate(val_preds))
        
        if (epoch + 1) % 10 == 0:
            print(f"    Finetune Epoch {epoch+1}: val_r2={val_r2:.4f}, perp={perp_avg:.1f}")
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        
        if patience >= 15:
            print(f"    Early stopping at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model


def evaluate(model, test_loader, device):
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
    print("Experiment 9: Push to R² ≥ 0.70")
    print("=" * 60)
    
    # Load data
    print("\n[1/2] Loading MC_Maze data...")
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
    
    # ========================================
    # Progressive VQ-VAE
    # ========================================
    print("\n[2/2] Training Progressive VQ-VAE...")
    
    model = ProgressiveVQVAE(num_codes=256, embedding_dim=128).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_progressive(model, train_loader, val_loader, device)
    results = evaluate(model, test_loader, device)
    
    # ========================================
    # Results
    # ========================================
    print("\n" + "=" * 60)
    print("EXPERIMENT 9 RESULTS: Progressive VQ-VAE")
    print("=" * 60)
    print(f"Test R² (overall): {results['r2']:.4f}")
    print(f"Test R² (vx):      {results['r2_vx']:.4f}")
    print(f"Test R² (vy):      {results['r2_vy']:.4f}")
    print(f"Codes used:        {results['n_codes_used']}/256")
    
    # Code distribution
    indices = results['indices']
    unique, counts = np.unique(indices, return_counts=True)
    entropy = -np.sum((counts/counts.sum()) * np.log2(counts/counts.sum() + 1e-10))
    max_entropy = np.log2(len(unique))
    print(f"Code entropy:      {entropy:.2f} bits (max={max_entropy:.2f})")
    
    print("\nTop 10 codes:")
    top10 = sorted(zip(counts, unique), reverse=True)[:10]
    for cnt, code in top10:
        print(f"  Code {code}: {cnt} ({100*cnt/len(indices):.1f}%)")
    
    if results['r2'] >= 0.70:
        print("\n🎉 SUCCESS! R² ≥ 0.70 achieved!")
    else:
        print(f"\nR² = {results['r2']:.4f}, target was 0.70")
    
    print("=" * 60)
    
    # Save model
    torch.save(model.state_dict(), 'models/exp9_progressive_vqvae.pt')
    print(f"\nSaved model to models/exp9_progressive_vqvae.pt")


if __name__ == '__main__':
    main()
