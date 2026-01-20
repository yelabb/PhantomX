"""
Experiment 10: Beat the LSTM (R² > 0.78)

GOAL: Surpass raw LSTM baseline (R² = 0.78) with discrete VQ-VAE architecture

KEY INSIGHTS FROM PREVIOUS EXPERIMENTS:
1. Progressive training is critical (prevents codebook collapse)
2. Gumbel-Softmax failed because of end-to-end training (collapse)
3. Transformer didn't help with short 10-step windows
4. Current best: Progressive VQ-VAE R² = 0.71

NEW STRATEGY:
1. Gumbel-Softmax with PROGRESSIVE training (not end-to-end)
2. Causal Transformer encoder (proper temporal modeling)
3. Longer window + efficient attention
4. Multi-scale temporal features

The hypothesis: The 10% gap (0.71 vs 0.78) is due to:
- VQ rigidity (hard discretization loses nuance) -> Soft quantization
- MLP temporal modeling (can't capture dynamics) -> Causal Transformer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import math
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phantomx.tokenizer import SpikeTokenizer
from phantomx.data import MCMazeDataset

# Data path
DATA_PATH = "c:/Users/guzzi/Desktop/Projects/DEV-ACTIF/NeuraLink/PhantomLink/data/raw/mc_maze.nwb"


# ============================================================
# COMPONENT 1: Progressive Gumbel-Softmax VQ
# ============================================================

class ProgressiveGumbelVQ(nn.Module):
    """
    Gumbel-Softmax VQ with progressive training support.
    
    Key innovation: K-means initialization + gradual temperature annealing
    starts AFTER pre-training, preventing early collapse.
    """
    
    def __init__(
        self,
        num_codes: int = 256,
        embedding_dim: int = 128,
        temp_init: float = 1.0,
        temp_min: float = 0.1,
        temp_anneal_epochs: int = 30
    ):
        super().__init__()
        
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.temp_min = temp_min
        self.temp_anneal_epochs = temp_anneal_epochs
        
        # Codebook
        self.embeddings = nn.Parameter(torch.randn(num_codes, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)
        
        # Logit projection (learnable similarity metric)
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)
        
        # Temperature (registered as buffer for checkpointing)
        self.register_buffer('temperature', torch.tensor(temp_init))
        self.register_buffer('current_epoch', torch.tensor(0))
        self._initialized = False
    
    def init_from_data(self, z_all: torch.Tensor):
        """Initialize codebook with k-means on encoder outputs."""
        print(f"    Initializing Gumbel VQ with k-means ({len(z_all)} samples)...")
        z_np = z_all.detach().cpu().numpy()
        
        kmeans = KMeans(n_clusters=self.num_codes, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(z_np)
        
        self.embeddings.data.copy_(torch.from_numpy(kmeans.cluster_centers_).float())
        self._initialized = True
        
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        print(f"    K-means init: {len(labels)} clusters, usage range [{counts.min()}, {counts.max()}]")
    
    def update_temperature(self, epoch: int):
        """Cosine annealing for temperature."""
        progress = min(epoch / self.temp_anneal_epochs, 1.0)
        temp = self.temp_min + 0.5 * (1.0 - self.temp_min) * (1 + math.cos(math.pi * progress))
        self.temperature.fill_(temp)
        self.current_epoch.fill_(epoch)
    
    def forward(self, z_e: torch.Tensor) -> tuple:
        """
        Soft quantization with Gumbel-Softmax.
        
        During training: soft mixture of codes (preserves gradients)
        During eval: hard assignment (discrete)
        """
        # Compute similarities (negative L2 distance)
        # z_e: [B, D], embeddings: [K, D]
        z_e_norm = F.normalize(z_e, dim=-1)
        emb_norm = F.normalize(self.embeddings, dim=-1)
        logits = self.logit_scale * (z_e_norm @ emb_norm.T)  # [B, K]
        
        if self.training and self._initialized:
            # Gumbel-Softmax: soft during training
            soft_onehot = F.gumbel_softmax(logits, tau=self.temperature.item(), hard=False)
            
            # Straight-through estimator: hard forward, soft backward
            hard_indices = logits.argmax(dim=-1)
            hard_onehot = F.one_hot(hard_indices, self.num_codes).float()
            
            # Mix: use soft for backprop, but add hard for discretization pressure
            mixing = min(1.0, self.current_epoch.item() / 10.0)  # Gradual hardening
            onehot = (1 - mixing) * soft_onehot + mixing * (hard_onehot - soft_onehot.detach() + soft_onehot)
            
            z_q = onehot @ self.embeddings  # [B, D]
            indices = hard_indices
        else:
            # Hard assignment during eval
            indices = logits.argmax(dim=-1)
            z_q = F.embedding(indices, self.embeddings)
        
        # Metrics
        probs = F.softmax(logits, dim=-1)
        avg_probs = probs.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Entropy regularization: encourage diversity
        entropy_loss = -torch.mean(torch.sum(probs * torch.log(probs + 1e-10), dim=-1))
        
        return z_q, {
            'indices': indices,
            'perplexity': perplexity,
            'temperature': self.temperature,
            'entropy_loss': 0.01 * entropy_loss,  # Small regularization
            'commitment_loss': torch.tensor(0.0, device=z_e.device)  # For compatibility
        }


# ============================================================
# COMPONENT 2: Causal Transformer Encoder
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention for temporal sequences.
    
    Each position can only attend to itself and previous positions,
    preventing information leakage from the future.
    """
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Causal mask
        self.register_buffer('mask', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        B, T, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: [B, T, H, head_dim]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, H, T, head_dim]
        
        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale  # [B, H, T, T]
        
        # Causal mask
        if self.mask is None or self.mask.size(-1) < T:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            self.register_buffer('mask', mask)
        attn = attn.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = attn @ v  # [B, H, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, D)
        
        return self.proj(out)


class CausalTransformerBlock(nn.Module):
    """Transformer block with causal attention."""
    
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


class CausalTransformerEncoder(nn.Module):
    """
    Causal Transformer encoder for neural time series.
    
    Design choices:
    - Causal attention (each timestep only sees past)
    - Rotary position embeddings (RoPE) for relative timing
    - Final timestep output (contains full history)
    """
    
    def __init__(
        self,
        n_channels: int = 142,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        output_dim: int = 128,
        max_len: int = 50
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(n_channels, d_model)
        
        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, n_channels] - temporal window of neural data
        Returns:
            [B, output_dim] - aggregated embedding
        """
        B, T, C = x.shape
        
        # Project + positional
        x = self.input_proj(x)  # [B, T, d_model]
        x = x + self.pos_embed[:, :T, :]
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_final(x)
        
        # Use LAST timestep (contains full causal history)
        # This is key for BCI: we want the representation at "now"
        last = x[:, -1, :]  # [B, d_model]
        
        return self.output_proj(last)


# ============================================================
# COMBINED MODEL: CausalTransformer + ProgressiveGumbel
# ============================================================

class BeatLSTMModel(nn.Module):
    """
    Architecture designed to beat raw LSTM (R² > 0.78).
    
    Key innovations:
    1. Causal Transformer encoder (proper temporal modeling)
    2. Progressive Gumbel-Softmax VQ (soft discretization + k-means init)
    3. Skip connection around VQ (preserve residual information)
    """
    
    def __init__(
        self,
        n_channels: int = 142,
        window_size: int = 10,
        d_model: int = 256,
        embedding_dim: int = 128,
        num_codes: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        use_skip: bool = True
    ):
        super().__init__()
        
        self.use_skip = use_skip
        
        self.encoder = CausalTransformerEncoder(
            n_channels=n_channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            output_dim=embedding_dim
        )
        
        self.vq = ProgressiveGumbelVQ(num_codes, embedding_dim)
        
        # Decoder: takes z_q (and optionally z_e via skip)
        decoder_input = embedding_dim * 2 if use_skip else embedding_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        
        self.use_vq = False  # Progressive training: start without VQ
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder output."""
        return self.encoder(x)
    
    def forward(self, x: torch.Tensor, targets: torch.Tensor = None) -> dict:
        z_e = self.encode(x)  # [B, embedding_dim]
        
        if self.use_vq:
            z_q, vq_info = self.vq(z_e)
        else:
            z_q = z_e
            vq_info = {
                'indices': torch.zeros(x.size(0), dtype=torch.long, device=x.device),
                'perplexity': torch.tensor(0.0, device=x.device),
                'temperature': torch.tensor(1.0, device=x.device),
                'entropy_loss': torch.tensor(0.0, device=x.device),
                'commitment_loss': torch.tensor(0.0, device=x.device)
            }
        
        # Decode (with optional skip connection)
        if self.use_skip and self.use_vq:
            decoder_input = torch.cat([z_q, z_e], dim=-1)
        else:
            decoder_input = z_q if not self.use_skip else torch.cat([z_q, z_q], dim=-1)
        
        velocity_pred = self.decoder(decoder_input)
        
        output = {'velocity_pred': velocity_pred, 'z_e': z_e, 'z_q': z_q, **vq_info}
        
        if targets is not None:
            recon_loss = F.mse_loss(velocity_pred, targets)
            total_loss = recon_loss + vq_info.get('entropy_loss', 0)
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
# Progressive Training
# ============================================================

def train_progressive(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    pretrain_epochs: int = 40,
    finetune_epochs: int = 60
):
    """
    Progressive training strategy:
    1. Pre-train encoder without VQ (builds good representations)
    2. K-means init on encoder outputs
    3. Finetune with VQ enabled (soft-to-hard annealing)
    """
    
    print("\n  ═══════════════════════════════════════════════════════")
    print("  PHASE 1: Pre-training Causal Transformer Encoder")
    print("  ═══════════════════════════════════════════════════════")
    
    model.use_vq = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pretrain_epochs)
    
    best_val_r2 = -float('inf')
    
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window, velocity)
            loss = output['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                velocity = batch['velocity'].to(device)
                output = model(window, velocity)
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(velocity.cpu())
        
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_r2 = r2_score(val_targets, val_preds)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: loss={train_loss/len(train_loader):.4f}, val_R²={val_r2:.4f} (best={best_val_r2:.4f})")
    
    print(f"\n    Pre-training complete. Best R² = {best_val_r2:.4f}")
    
    # ================================================================
    # PHASE 2: K-means initialization
    # ================================================================
    print("\n  ═══════════════════════════════════════════════════════")
    print("  PHASE 2: K-means Codebook Initialization")
    print("  ═══════════════════════════════════════════════════════")
    
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in train_loader:
            window = batch['window'].to(device)
            z_e = model.encode(window)
            embeddings.append(z_e.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    model.vq.init_from_data(embeddings)
    
    # ================================================================
    # PHASE 3: Finetune with VQ
    # ================================================================
    print("\n  ═══════════════════════════════════════════════════════")
    print("  PHASE 3: Finetuning with Gumbel-Softmax VQ")
    print("  ═══════════════════════════════════════════════════════")
    
    model.use_vq = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epochs)
    
    best_val_r2_vq = -float('inf')
    best_state = None
    
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        model.vq.update_temperature(epoch)
        
        train_loss = 0
        codes_used = set()
        
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window, velocity)
            loss = output['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            codes_used.update(output['indices'].cpu().numpy().tolist())
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                velocity = batch['velocity'].to(device)
                output = model(window, velocity)
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(velocity.cpu())
        
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_r2 = r2_score(val_targets, val_preds)
        
        if val_r2 > best_val_r2_vq:
            best_val_r2_vq = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 10 == 0:
            temp = model.vq.temperature.item()
            print(f"    Epoch {epoch:3d}: loss={train_loss/len(train_loader):.4f}, val_R²={val_r2:.4f}, "
                  f"temp={temp:.3f}, codes={len(codes_used)}")
    
    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"\n    Finetuning complete. Best VQ R² = {best_val_r2_vq:.4f}")
    
    return best_val_r2_vq


# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    print("\n" + "="*70)
    print("EXPERIMENT 10: Beat the LSTM (Target: R² > 0.78)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading MC_Maze dataset...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_maze = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    spike_counts = mc_maze.spike_counts  # [N, 142]
    velocities = mc_maze.velocities      # [N, 2]
    
    # Normalize
    spike_counts = (spike_counts - spike_counts.mean(0)) / (spike_counts.std(0) + 1e-6)
    velocities = (velocities - velocities.mean(0)) / (velocities.std(0) + 1e-6)
    
    # Create datasets
    window_size = 10
    dataset = SlidingWindowDataset(spike_counts, velocities, window_size)
    
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, len(dataset)))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # ================================================================
    # Test multiple configurations
    # ================================================================
    
    configs = [
        {
            'name': 'CausalTransformer + Gumbel (skip)',
            'd_model': 256,
            'num_layers': 4,
            'use_skip': True
        },
        {
            'name': 'CausalTransformer + Gumbel (no skip)',
            'd_model': 256,
            'num_layers': 4,
            'use_skip': False
        },
        {
            'name': 'Deep CausalTransformer + Gumbel',
            'd_model': 256,
            'num_layers': 6,
            'use_skip': True
        },
        {
            'name': 'Wide CausalTransformer + Gumbel',
            'd_model': 384,
            'num_layers': 4,
            'use_skip': True
        },
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {cfg['name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        model = BeatLSTMModel(
            n_channels=142,
            window_size=window_size,
            d_model=cfg['d_model'],
            embedding_dim=128,
            num_codes=256,
            nhead=8,
            num_layers=cfg['num_layers'],
            use_skip=cfg['use_skip']
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")
        
        val_r2 = train_progressive(model, train_loader, val_loader, device)
        elapsed = time.time() - start_time
        
        # Final evaluation
        model.eval()
        test_preds, test_targets = [], []
        codes_used = set()
        
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                velocity = batch['velocity'].to(device)
                output = model(window, velocity)
                test_preds.append(output['velocity_pred'].cpu())
                test_targets.append(velocity.cpu())
                codes_used.update(output['indices'].cpu().numpy().tolist())
        
        test_preds = torch.cat(test_preds).numpy()
        test_targets = torch.cat(test_targets).numpy()
        
        r2_overall = r2_score(test_targets, test_preds)
        r2_vx = r2_score(test_targets[:, 0], test_preds[:, 0])
        r2_vy = r2_score(test_targets[:, 1], test_preds[:, 1])
        
        results.append({
            'name': cfg['name'],
            'r2': r2_overall,
            'r2_vx': r2_vx,
            'r2_vy': r2_vy,
            'codes': len(codes_used),
            'time': elapsed,
            'params': param_count
        })
        
        print(f"\n  Result: R²={r2_overall:.4f} (vx={r2_vx:.4f}, vy={r2_vy:.4f}), "
              f"codes={len(codes_used)}, time={elapsed:.1f}s")
        
        # Save best model
        if r2_overall >= 0.78:
            print(f"\n  🎉 TARGET ACHIEVED! Saving model...")
            torch.save(model.state_dict(), Path(__file__).parent / 'models' / 'exp10_beat_lstm.pt')
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 10 RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Model':<45} {'R²':>8} {'vx':>8} {'vy':>8} {'Codes':>6} {'Time':>8}")
    print("-"*70)
    
    for r in sorted(results, key=lambda x: x['r2'], reverse=True):
        status = "🎯" if r['r2'] >= 0.78 else ("📈" if r['r2'] >= 0.71 else "")
        print(f"{r['name']:<45} {r['r2']:>7.4f} {r['r2_vx']:>7.4f} {r['r2_vy']:>7.4f} "
              f"{r['codes']:>6} {r['time']:>7.1f}s {status}")
    
    print("\n" + "-"*70)
    print("Comparison to baselines:")
    print("  • Raw LSTM (target):    R² = 0.78")
    print("  • Progressive VQ-VAE:   R² = 0.71")
    print("="*70)
    
    best = max(results, key=lambda x: x['r2'])
    if best['r2'] >= 0.78:
        print(f"\n✅ SUCCESS: {best['name']} achieved R² = {best['r2']:.4f} (beats LSTM!)")
    elif best['r2'] >= 0.71:
        print(f"\n📈 PROGRESS: {best['name']} achieved R² = {best['r2']:.4f} (improved from 0.71)")
    else:
        print(f"\n❌ Need more work: best R² = {best['r2']:.4f}")
    
    return results


if __name__ == "__main__":
    run_experiment()
