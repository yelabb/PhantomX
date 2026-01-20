"""
Experiment 11: Close the Final Gap (R² > 0.78)

CURRENT STATE:
- Deep CausalTransformer + Gumbel: R² = 0.7727
- Raw LSTM target: R² = 0.78
- Gap: 0.7%

KEY INSIGHT FROM EXP 10:
The encoder ALONE reached R² = 0.78 during pre-training!
The gap is purely from VQ discretization bottleneck.

STRATEGIES TO CLOSE THE GAP:
1. Softer VQ: Higher minimum temperature (0.3 instead of 0.1)
2. Larger codebook: 512 codes for finer quantization
3. Residual VQ: Add continuous residual to preserve nuance
4. Longer pre-training: 60 epochs instead of 40
5. Multi-head VQ: Product quantization (4 heads × 64 codes)
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

# Use relative path from project root
DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"


# ============================================================
# Causal Transformer Components (from exp10)
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
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


class CausalTransformerBlock(nn.Module):
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
    def __init__(self, n_channels=142, d_model=256, nhead=8, num_layers=6,
                 dim_ff=512, dropout=0.1, output_dim=128, max_len=50):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.output_proj(x[:, -1, :])


# ============================================================
# Strategy 1: Softer Gumbel VQ (higher min temperature)
# ============================================================

class SoftGumbelVQ(nn.Module):
    """Gumbel VQ with softer annealing (temp stays at 0.3 minimum)."""
    
    def __init__(self, num_codes=256, embedding_dim=128, 
                 temp_init=1.0, temp_min=0.3, temp_anneal_epochs=40):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.temp_min = temp_min
        self.temp_anneal_epochs = temp_anneal_epochs
        
        self.embeddings = nn.Parameter(torch.randn(num_codes, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)
        self.register_buffer('temperature', torch.tensor(temp_init))
        self._initialized = False
    
    def init_from_data(self, z_all: torch.Tensor):
        z_np = z_all.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_codes, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(z_np)
        self.embeddings.data.copy_(torch.from_numpy(kmeans.cluster_centers_).float())
        self._initialized = True
    
    def update_temperature(self, epoch: int, max_epochs: int = None):
        # Use max_epochs if provided, otherwise use temp_anneal_epochs
        anneal_epochs = max_epochs if max_epochs else self.temp_anneal_epochs
        progress = min(epoch / anneal_epochs, 1.0)
        temp = self.temp_min + 0.5 * (1.0 - self.temp_min) * (1 + math.cos(math.pi * progress))
        self.temperature.fill_(temp)
    
    def forward(self, z_e: torch.Tensor) -> tuple:
        z_e_norm = F.normalize(z_e, dim=-1)
        emb_norm = F.normalize(self.embeddings, dim=-1)
        logits = self.logit_scale * (z_e_norm @ emb_norm.T)
        
        if self.training and self._initialized:
            soft_onehot = F.gumbel_softmax(logits, tau=self.temperature.item(), hard=False)
            z_q = soft_onehot @ self.embeddings
            indices = logits.argmax(dim=-1)
        else:
            indices = logits.argmax(dim=-1)
            z_q = F.embedding(indices, self.embeddings)
        
        probs = F.softmax(logits, dim=-1)
        avg_probs = probs.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, {
            'indices': indices,
            'perplexity': perplexity,
            'temperature': self.temperature,
            'commitment_loss': torch.tensor(0.0, device=z_e.device)
        }


# ============================================================
# Strategy 2: Residual VQ (continuous residual preserved)
# ============================================================

class ResidualGumbelVQ(nn.Module):
    """VQ with learnable residual: z_out = α*z_q + (1-α)*z_e"""
    
    def __init__(self, num_codes=256, embedding_dim=128, 
                 temp_min=0.2, residual_init=0.3):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.temp_min = temp_min
        
        self.embeddings = nn.Parameter(torch.randn(num_codes, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)
        
        # Learnable residual weight (how much continuous info to keep)
        self.residual_weight = nn.Parameter(torch.tensor(residual_init))
        
        self.register_buffer('temperature', torch.tensor(1.0))
        self._initialized = False
    
    def init_from_data(self, z_all: torch.Tensor):
        z_np = z_all.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_codes, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(z_np)
        self.embeddings.data.copy_(torch.from_numpy(kmeans.cluster_centers_).float())
        self._initialized = True
    
    def update_temperature(self, epoch: int, max_epochs: int = 60):
        progress = min(epoch / max_epochs, 1.0)
        temp = self.temp_min + 0.5 * (1.0 - self.temp_min) * (1 + math.cos(math.pi * progress))
        self.temperature.fill_(temp)
    
    def forward(self, z_e: torch.Tensor) -> tuple:
        z_e_norm = F.normalize(z_e, dim=-1)
        emb_norm = F.normalize(self.embeddings, dim=-1)
        logits = self.logit_scale * (z_e_norm @ emb_norm.T)
        
        if self.training and self._initialized:
            soft_onehot = F.gumbel_softmax(logits, tau=self.temperature.item(), hard=False)
            z_q_pure = soft_onehot @ self.embeddings
            indices = logits.argmax(dim=-1)
        else:
            indices = logits.argmax(dim=-1)
            z_q_pure = F.embedding(indices, self.embeddings)
        
        # Residual connection: blend quantized and continuous
        alpha = torch.sigmoid(self.residual_weight)  # Keep in [0, 1]
        z_q = alpha * z_q_pure + (1 - alpha) * z_e
        
        probs = F.softmax(logits, dim=-1)
        avg_probs = probs.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, {
            'indices': indices,
            'perplexity': perplexity,
            'temperature': self.temperature,
            'residual_alpha': alpha,
            'commitment_loss': torch.tensor(0.0, device=z_e.device)
        }


# ============================================================
# Strategy 3: Product VQ (multi-head quantization)
# ============================================================

class ProductGumbelVQ(nn.Module):
    """Product quantization: 4 heads × 64 codes each = 16M combinations."""
    
    def __init__(self, num_heads=4, codes_per_head=64, embedding_dim=128, temp_min=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.codes_per_head = codes_per_head
        self.head_dim = embedding_dim // num_heads
        self.temp_min = temp_min
        
        # Each head has its own codebook
        self.embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(codes_per_head, self.head_dim))
            for _ in range(num_heads)
        ])
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb)
        
        self.logit_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 10.0) for _ in range(num_heads)
        ])
        
        self.register_buffer('temperature', torch.tensor(1.0))
        self._initialized = False
    
    def init_from_data(self, z_all: torch.Tensor):
        z_np = z_all.detach().cpu().numpy()
        for h in range(self.num_heads):
            z_head = z_np[:, h*self.head_dim:(h+1)*self.head_dim]
            kmeans = KMeans(n_clusters=self.codes_per_head, n_init=10, max_iter=300, random_state=42+h)
            kmeans.fit(z_head)
            self.embeddings[h].data.copy_(torch.from_numpy(kmeans.cluster_centers_).float())
        self._initialized = True
    
    def update_temperature(self, epoch: int, max_epochs: int = 60):
        progress = min(epoch / max_epochs, 1.0)
        temp = self.temp_min + 0.5 * (1.0 - self.temp_min) * (1 + math.cos(math.pi * progress))
        self.temperature.fill_(temp)
    
    def forward(self, z_e: torch.Tensor) -> tuple:
        B = z_e.size(0)
        z_q_parts = []
        all_indices = []
        all_perplexities = []
        
        for h in range(self.num_heads):
            z_h = z_e[:, h*self.head_dim:(h+1)*self.head_dim]
            z_h_norm = F.normalize(z_h, dim=-1)
            emb_norm = F.normalize(self.embeddings[h], dim=-1)
            logits = self.logit_scales[h] * (z_h_norm @ emb_norm.T)
            
            if self.training and self._initialized:
                soft_onehot = F.gumbel_softmax(logits, tau=self.temperature.item(), hard=False)
                z_q_h = soft_onehot @ self.embeddings[h]
                indices_h = logits.argmax(dim=-1)
            else:
                indices_h = logits.argmax(dim=-1)
                z_q_h = F.embedding(indices_h, self.embeddings[h])
            
            z_q_parts.append(z_q_h)
            all_indices.append(indices_h)
            
            probs = F.softmax(logits, dim=-1)
            avg_probs = probs.mean(0)
            perp = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            all_perplexities.append(perp)
        
        z_q = torch.cat(z_q_parts, dim=-1)
        
        # Combine indices into unique code (for logging)
        combined_indices = all_indices[0].clone()
        for h in range(1, self.num_heads):
            combined_indices = combined_indices * self.codes_per_head + all_indices[h]
        
        avg_perplexity = sum(all_perplexities) / self.num_heads
        
        return z_q, {
            'indices': combined_indices,
            'perplexity': avg_perplexity,
            'temperature': self.temperature,
            'commitment_loss': torch.tensor(0.0, device=z_e.device)
        }


# ============================================================
# Unified Model
# ============================================================

class CloseGapModel(nn.Module):
    def __init__(self, n_channels=142, window_size=10, d_model=256, 
                 embedding_dim=128, num_codes=256, num_layers=6,
                 vq_type='soft'):
        super().__init__()
        
        self.encoder = CausalTransformerEncoder(
            n_channels=n_channels,
            d_model=d_model,
            nhead=8,
            num_layers=num_layers,
            output_dim=embedding_dim
        )
        
        if vq_type == 'soft':
            self.vq = SoftGumbelVQ(num_codes, embedding_dim, temp_min=0.3)
        elif vq_type == 'residual':
            self.vq = ResidualGumbelVQ(num_codes, embedding_dim, temp_min=0.2)
        elif vq_type == 'product':
            self.vq = ProductGumbelVQ(num_heads=4, codes_per_head=64, 
                                       embedding_dim=embedding_dim, temp_min=0.2)
        else:
            raise ValueError(f"Unknown vq_type: {vq_type}")
        
        self.vq_type = vq_type
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        
        self.use_vq = False
    
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x, targets=None):
        z_e = self.encode(x)
        
        if self.use_vq:
            z_q, vq_info = self.vq(z_e)
        else:
            z_q = z_e
            vq_info = {
                'indices': torch.zeros(x.size(0), dtype=torch.long, device=x.device),
                'perplexity': torch.tensor(0.0, device=x.device),
                'temperature': torch.tensor(1.0, device=x.device),
                'commitment_loss': torch.tensor(0.0, device=x.device)
            }
        
        velocity_pred = self.decoder(z_q)
        
        output = {'velocity_pred': velocity_pred, 'z_e': z_e, 'z_q': z_q, **vq_info}
        
        if targets is not None:
            recon_loss = F.mse_loss(velocity_pred, targets)
            output['recon_loss'] = recon_loss
            output['total_loss'] = recon_loss
        
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

def train_progressive(model, train_loader, val_loader, device,
                      pretrain_epochs=60, finetune_epochs=80):
    """Extended progressive training with longer pre-training."""
    
    print("\n  Phase 1: Pre-training Encoder (extended)...")
    model.use_vq = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pretrain_epochs)
    
    best_val_r2 = -float('inf')
    
    for epoch in range(1, pretrain_epochs + 1):
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
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['window'].to(device))
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(batch['velocity'])
        
        val_r2 = r2_score(torch.cat(val_targets).numpy(), torch.cat(val_preds).numpy())
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
        
        if epoch % 15 == 0:
            print(f"    Epoch {epoch:3d}: val_R²={val_r2:.4f} (best={best_val_r2:.4f})")
    
    print(f"\n    Pre-training complete. Best R² = {best_val_r2:.4f}")
    
    # Phase 2: K-means init
    print("\n  Phase 2: K-means Codebook Init...")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in train_loader:
            z_e = model.encode(batch['window'].to(device))
            embeddings.append(z_e.cpu())
    model.vq.init_from_data(torch.cat(embeddings))
    
    # Phase 3: Finetune with VQ
    print("\n  Phase 3: Finetuning with VQ...")
    model.use_vq = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epochs)
    
    best_val_r2_vq = -float('inf')
    best_state = None
    
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        model.vq.update_temperature(epoch, finetune_epochs)
        
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
        codes_used = set()
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['window'].to(device))
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(batch['velocity'])
                codes_used.update(output['indices'].cpu().numpy().tolist())
        
        val_r2 = r2_score(torch.cat(val_targets).numpy(), torch.cat(val_preds).numpy())
        
        if val_r2 > best_val_r2_vq:
            best_val_r2_vq = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 20 == 0:
            temp = model.vq.temperature.item()
            extra = ""
            if hasattr(model.vq, 'residual_weight'):
                alpha = torch.sigmoid(model.vq.residual_weight).item()
                extra = f", α={alpha:.2f}"
            print(f"    Epoch {epoch:3d}: val_R²={val_r2:.4f}, temp={temp:.3f}, codes={len(codes_used)}{extra}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n    Finetuning complete. Best VQ R² = {best_val_r2_vq:.4f}")
    return best_val_r2_vq


def run_experiment():
    print("\n" + "="*70)
    print("EXPERIMENT 11: Close the Final Gap (Target: R² > 0.78)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading MC_Maze dataset...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_maze = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    spike_counts = mc_maze.spike_counts
    velocities = mc_maze.velocities
    
    spike_counts = (spike_counts - spike_counts.mean(0)) / (spike_counts.std(0) + 1e-6)
    velocities = (velocities - velocities.mean(0)) / (velocities.std(0) + 1e-6)
    
    window_size = 10
    dataset = SlidingWindowDataset(spike_counts, velocities, window_size)
    
    n_train = int(0.8 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, list(range(n_train)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(n_train, len(dataset))))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Test configurations
    configs = [
        {'name': 'Soft Gumbel (temp_min=0.3)', 'vq_type': 'soft'},
        {'name': 'Residual Gumbel (learnable α)', 'vq_type': 'residual'},
        {'name': 'Product Gumbel (4×64)', 'vq_type': 'product'},
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {cfg['name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        model = CloseGapModel(
            n_channels=142,
            window_size=window_size,
            d_model=256,
            embedding_dim=128,
            num_codes=256,
            num_layers=6,
            vq_type=cfg['vq_type']
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")
        
        train_progressive(model, train_loader, val_loader, device)
        elapsed = time.time() - start_time
        
        # Final eval
        model.eval()
        test_preds, test_targets = [], []
        codes_used = set()
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['window'].to(device))
                test_preds.append(output['velocity_pred'].cpu())
                test_targets.append(batch['velocity'])
                codes_used.update(output['indices'].cpu().numpy().tolist())
        
        test_preds = torch.cat(test_preds).numpy()
        test_targets = torch.cat(test_targets).numpy()
        
        r2 = r2_score(test_targets, test_preds)
        r2_vx = r2_score(test_targets[:, 0], test_preds[:, 0])
        r2_vy = r2_score(test_targets[:, 1], test_preds[:, 1])
        
        results.append({
            'name': cfg['name'],
            'r2': r2,
            'r2_vx': r2_vx,
            'r2_vy': r2_vy,
            'codes': len(codes_used),
            'time': elapsed
        })
        
        status = "🎯" if r2 >= 0.78 else ("📈" if r2 >= 0.77 else "")
        print(f"\n  Result: R²={r2:.4f} (vx={r2_vx:.4f}, vy={r2_vy:.4f}), "
              f"codes={len(codes_used)}, time={elapsed/60:.1f}min {status}")
        
        if r2 >= 0.78:
            torch.save(model.state_dict(), Path(__file__).parent / 'models' / 'exp11_beat_lstm.pt')
            print("  🎉 TARGET ACHIEVED! Model saved.")
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT 11 RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Model':<40} {'R²':>8} {'vx':>8} {'vy':>8} {'Codes':>8}")
    print("-"*70)
    
    for r in sorted(results, key=lambda x: x['r2'], reverse=True):
        status = "🎯" if r['r2'] >= 0.78 else ("📈" if r['r2'] >= 0.77 else "")
        print(f"{r['name']:<40} {r['r2']:>7.4f} {r['r2_vx']:>7.4f} {r['r2_vy']:>7.4f} "
              f"{r['codes']:>8} {status}")
    
    print("\n" + "-"*70)
    print("Comparison:")
    print("  • Raw LSTM:                  R² = 0.78")
    print("  • Previous best (Exp 10):    R² = 0.7727")
    print("="*70)
    
    best = max(results, key=lambda x: x['r2'])
    if best['r2'] >= 0.78:
        print(f"\n✅ SUCCESS: {best['name']} achieved R² = {best['r2']:.4f} - LSTM BEATEN!")
    else:
        print(f"\n📈 Best: {best['name']} R² = {best['r2']:.4f}")
    
    return results


if __name__ == "__main__":
    run_experiment()
