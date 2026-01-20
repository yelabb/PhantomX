"""
Experiment 12: Residual Vector Quantization (RVQ)

MOTIVATION (from Red Team critique of Exp 11):
- Single-codebook VQ creates a "Voronoi Ceiling" - max R² is capped by codebook density
- Codebook collapse: only ~150/512 codes used = 85% dead weight
- Pre-training success (R²=0.76) hides that the VQ bottleneck is the chokepoint
- At low Gumbel temperatures, gradients approximate HardArgMax (zero Jacobian)
- You are trying to draw a smooth circle on a Minecraft grid

SOLUTION (Blue Team Pivot):
Implement Multi-Stage Residual Vector Quantization (RVQ):

    z_q1 = Q_1(z)           # Coarse trajectory (Layer 1: "intent")
    r_1 = z - z_q1          # Residual error
    z_q2 = Q_2(r_1)         # Medium detail (Layer 2: "speed")
    r_2 = r_1 - z_q2        # Residual error
    z_q3 = Q_3(r_2)         # Fine correction (Layer 3: "jitter")
    ...
    ẑ = z_q1 + z_q2 + ... + z_qN

KEY BENEFITS:
1. Exponential Variance Reduction: Error decays exponentially with layers
2. Decouples Semantics from Precision: Aligns with hierarchical motor control
3. Effective Vocabulary: K codes × N layers = K^N combinations
4. Gradient Flow: Sum of embeddings allows micro-adjustments through correction codebooks

IMPLEMENTATION:
- Discard Gumbel complexity
- Use Euclidean distance quantization with K-means init
- Apply standard commitment loss recursively
- 4-8 residual quantization stages with smaller codebooks (64-128 codes each)
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

DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"


# ============================================================
# Causal Transformer Encoder (from exp10/11)
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
# Residual Vector Quantization (RVQ) - Core Innovation
# ============================================================

class ResidualVQLayer(nn.Module):
    """
    Single layer of residual vector quantization.
    
    Uses standard Euclidean distance quantization with:
    - K-means initialization
    - Commitment loss
    - EMA codebook updates for stability
    """
    
    def __init__(self, num_codes: int = 128, embedding_dim: int = 128,
                 commitment_cost: float = 0.25, decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # Codebook embeddings
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        
        # EMA statistics
        self.register_buffer('ema_cluster_size', torch.zeros(num_codes))
        self.register_buffer('ema_embedding', self.embedding.weight.data.clone())
        self.register_buffer('initialized', torch.tensor(False))
    
    def init_from_data(self, z: torch.Tensor):
        """Initialize codebook using K-means clustering."""
        z_np = z.detach().cpu().numpy()
        n_samples = len(z_np)
        
        # Use fewer clusters if not enough samples
        n_clusters = min(self.num_codes, n_samples // 2)
        if n_clusters < self.num_codes:
            print(f"    Warning: Using {n_clusters} clusters (not enough samples for {self.num_codes})")
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(z_np)
        
        # Fill remaining codes with random perturbations of centroids
        centroids = torch.from_numpy(kmeans.cluster_centers_).float()
        if n_clusters < self.num_codes:
            remaining = self.num_codes - n_clusters
            random_indices = torch.randint(0, n_clusters, (remaining,))
            noise = torch.randn(remaining, self.embedding_dim) * 0.1
            extra_centroids = centroids[random_indices] + noise
            centroids = torch.cat([centroids, extra_centroids], dim=0)
        
        self.embedding.weight.data.copy_(centroids.to(self.embedding.weight.device))
        self.ema_embedding.copy_(self.embedding.weight.data)
        self.initialized.fill_(True)
    
    def forward(self, z: torch.Tensor) -> tuple:
        """
        Quantize input and return quantized output + residual.
        
        Args:
            z: [batch, embedding_dim] input to quantize
            
        Returns:
            z_q: [batch, embedding_dim] quantized output
            residual: [batch, embedding_dim] quantization error (z - z_q)
            info: dict with indices, commitment_loss, perplexity
        """
        # Compute distances: ||z - e||² = ||z||² + ||e||² - 2⟨z, e⟩
        z_flat = z.view(-1, self.embedding_dim)
        
        z_norm = torch.sum(z_flat ** 2, dim=1, keepdim=True)
        e_norm = torch.sum(self.embedding.weight ** 2, dim=1)
        distances = z_norm + e_norm.unsqueeze(0) - 2 * z_flat @ self.embedding.weight.t()
        
        # Find nearest codes
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)
        
        # Commitment loss: encourage encoder to commit to codebook
        commitment_loss = F.mse_loss(z_q.detach(), z_flat) * self.commitment_cost
        
        # Straight-through estimator
        z_q_st = z_flat + (z_q - z_flat).detach()
        
        # EMA codebook update
        if self.training:
            self._ema_update(z_flat, indices)
        
        # Compute perplexity (codebook utilization)
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Residual for next layer
        residual = z_flat - z_q_st
        
        return z_q_st.view_as(z), residual.view_as(z), {
            'indices': indices,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity
        }
    
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        """Update codebook using exponential moving average."""
        with torch.no_grad():
            encodings = F.one_hot(indices, self.num_codes).float()
            
            # Update cluster sizes
            self.ema_cluster_size = self.decay * self.ema_cluster_size + \
                                    (1 - self.decay) * encodings.sum(0)
            
            # Laplace smoothing
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon) /
                (n + self.num_codes * self.epsilon) * n
            )
            
            # Update embeddings
            dw = encodings.t() @ z
            self.ema_embedding = self.decay * self.ema_embedding + (1 - self.decay) * dw
            
            # Normalize
            self.embedding.weight.data = self.ema_embedding / self.ema_cluster_size.unsqueeze(1)


class ResidualVectorQuantizer(nn.Module):
    """
    Multi-stage Residual Vector Quantizer.
    
    Applies N quantization stages in cascade:
        z_q1 = Q1(z)
        r1 = z - z_q1
        z_q2 = Q2(r1)
        r2 = r1 - z_q2
        ...
        final = z_q1 + z_q2 + ... + z_qN
    
    This achieves exponential variance reduction and captures
    both coarse structure and fine details.
    """
    
    def __init__(self, num_quantizers: int = 4, num_codes: int = 128, 
                 embedding_dim: int = 128, commitment_cost: float = 0.25,
                 shared_codebook: bool = False):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        
        if shared_codebook:
            # All layers share the same codebook (parameter efficient)
            base_quantizer = ResidualVQLayer(num_codes, embedding_dim, commitment_cost)
            self.quantizers = nn.ModuleList([base_quantizer] * num_quantizers)
        else:
            # Each layer has its own codebook (more expressive)
            self.quantizers = nn.ModuleList([
                ResidualVQLayer(num_codes, embedding_dim, commitment_cost)
                for _ in range(num_quantizers)
            ])
    
    def init_from_data(self, z_all: torch.Tensor):
        """Initialize all codebooks from data using residual clustering."""
        residual = z_all.clone()
        for i, quantizer in enumerate(self.quantizers):
            print(f"    Initializing RVQ layer {i+1}/{self.num_quantizers}...")
            quantizer.init_from_data(residual)
            
            # Compute residual for next layer
            with torch.no_grad():
                z_q, new_residual, _ = quantizer(residual)
                residual = new_residual
    
    def forward(self, z: torch.Tensor) -> tuple:
        """
        Apply residual vector quantization.
        
        Args:
            z: [batch, embedding_dim] encoder output
            
        Returns:
            z_q: [batch, embedding_dim] sum of quantized outputs
            info: dict with aggregated metrics
        """
        z_q_sum = torch.zeros_like(z)
        residual = z
        
        all_indices = []
        total_commitment_loss = 0.0
        total_perplexity = 0.0
        layer_perplexities = []
        
        for i, quantizer in enumerate(self.quantizers):
            z_q_i, residual, info = quantizer(residual)
            z_q_sum = z_q_sum + z_q_i
            
            all_indices.append(info['indices'])
            total_commitment_loss = total_commitment_loss + info['commitment_loss']
            total_perplexity = total_perplexity + info['perplexity']
            layer_perplexities.append(info['perplexity'].item())
        
        # Compute unique code combinations
        # For logging: combine first 2-3 layers for unique ID
        combined_indices = all_indices[0].clone()
        for indices in all_indices[1:min(3, len(all_indices))]:
            combined_indices = combined_indices * self.num_codes + indices
        
        return z_q_sum, {
            'indices': combined_indices,
            'all_indices': all_indices,
            'commitment_loss': total_commitment_loss / self.num_quantizers,
            'perplexity': total_perplexity / self.num_quantizers,
            'layer_perplexities': layer_perplexities,
            'residual_norm': residual.norm(dim=-1).mean()
        }


# ============================================================
# Progressive RVQ (with layer dropout for robustness)
# ============================================================

class ProgressiveRVQ(nn.Module):
    """
    RVQ with progressive training and layer dropout.
    
    Features:
    - Gradual unfreezing of layers during training
    - Stochastic layer dropout for robustness
    - Residual scaling for stable training
    """
    
    def __init__(self, num_quantizers: int = 6, num_codes: int = 128,
                 embedding_dim: int = 128, commitment_cost: float = 0.25,
                 layer_dropout: float = 0.0, residual_scale: float = 1.0):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.layer_dropout = layer_dropout
        self.residual_scale = residual_scale
        
        self.quantizers = nn.ModuleList([
            ResidualVQLayer(num_codes, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])
        
        # Learnable scales for each layer
        self.layer_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1) * (0.5 ** i))  # Decreasing importance
            for i in range(num_quantizers)
        ])
        
        self.active_layers = num_quantizers  # Start with all layers
    
    def init_from_data(self, z_all: torch.Tensor):
        """Initialize all codebooks progressively."""
        residual = z_all.clone()
        for i, quantizer in enumerate(self.quantizers):
            print(f"    Initializing Progressive RVQ layer {i+1}/{self.num_quantizers}...")
            quantizer.init_from_data(residual)
            
            with torch.no_grad():
                z_q, new_residual, _ = quantizer(residual)
                residual = new_residual * self.residual_scale
    
    def set_active_layers(self, n: int):
        """Set number of active quantization layers."""
        self.active_layers = min(n, self.num_quantizers)
    
    def forward(self, z: torch.Tensor) -> tuple:
        z_q_sum = torch.zeros_like(z)
        residual = z
        
        all_indices = []
        total_commitment_loss = 0.0
        total_perplexity = 0.0
        layer_perplexities = []
        
        for i in range(self.active_layers):
            # Layer dropout during training
            if self.training and i > 0 and torch.rand(1).item() < self.layer_dropout:
                continue
            
            z_q_i, new_residual, info = self.quantizers[i](residual)
            
            # Scale contribution
            scale = torch.sigmoid(self.layer_scales[i])
            z_q_sum = z_q_sum + scale * z_q_i
            residual = new_residual * self.residual_scale
            
            all_indices.append(info['indices'])
            total_commitment_loss = total_commitment_loss + info['commitment_loss']
            total_perplexity = total_perplexity + info['perplexity']
            layer_perplexities.append(info['perplexity'].item())
        
        n_active = len(all_indices)
        combined_indices = all_indices[0] if all_indices else torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        
        return z_q_sum, {
            'indices': combined_indices,
            'all_indices': all_indices,
            'commitment_loss': total_commitment_loss / max(n_active, 1),
            'perplexity': total_perplexity / max(n_active, 1),
            'layer_perplexities': layer_perplexities,
            'residual_norm': residual.norm(dim=-1).mean(),
            'active_layers': n_active
        }


# ============================================================
# Main Model with RVQ
# ============================================================

class RVQModel(nn.Module):
    """
    Neural decoder with Residual Vector Quantization.
    
    Architecture:
        Spikes → CausalTransformer → RVQ → MLP Decoder → Velocity
    """
    
    def __init__(self, n_channels=142, window_size=10, d_model=256,
                 embedding_dim=128, num_quantizers=4, num_codes=128,
                 num_layers=6, rvq_type='standard', layer_dropout=0.0):
        super().__init__()
        
        self.encoder = CausalTransformerEncoder(
            n_channels=n_channels,
            d_model=d_model,
            nhead=8,
            num_layers=num_layers,
            output_dim=embedding_dim
        )
        
        if rvq_type == 'standard':
            self.vq = ResidualVectorQuantizer(
                num_quantizers=num_quantizers,
                num_codes=num_codes,
                embedding_dim=embedding_dim
            )
        elif rvq_type == 'progressive':
            self.vq = ProgressiveRVQ(
                num_quantizers=num_quantizers,
                num_codes=num_codes,
                embedding_dim=embedding_dim,
                layer_dropout=layer_dropout
            )
        else:
            raise ValueError(f"Unknown rvq_type: {rvq_type}")
        
        self.rvq_type = rvq_type
        
        # Decoder: richer network to utilize RVQ's expressiveness
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
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
                'commitment_loss': torch.tensor(0.0, device=x.device),
                'residual_norm': torch.tensor(0.0, device=x.device)
            }
        
        velocity_pred = self.decoder(z_q)
        
        output = {
            'velocity_pred': velocity_pred,
            'z_e': z_e,
            'z_q': z_q,
            **vq_info
        }
        
        if targets is not None:
            recon_loss = F.mse_loss(velocity_pred, targets)
            commitment_loss = vq_info.get('commitment_loss', 0.0)
            if isinstance(commitment_loss, torch.Tensor):
                total_loss = recon_loss + commitment_loss
            else:
                total_loss = recon_loss
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
# Training with RVQ-specific phases
# ============================================================

def train_rvq_progressive(model, train_loader, val_loader, device,
                          pretrain_epochs=50, finetune_epochs=100):
    """
    Training procedure for RVQ model.
    
    Phase 1: Pre-train encoder without VQ
    Phase 2: Initialize RVQ codebooks with residual K-means
    Phase 3: Fine-tune full model with RVQ
    """
    
    print("\n  Phase 1: Pre-training Encoder...")
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
        
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: val_R²={val_r2:.4f} (best={best_val_r2:.4f})")
    
    print(f"\n    Pre-training complete. Best R² = {best_val_r2:.4f}")
    
    # Phase 2: Initialize RVQ codebooks
    print("\n  Phase 2: Residual K-means Codebook Init...")
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in train_loader:
            z_e = model.encode(batch['window'].to(device))
            embeddings.append(z_e.cpu())
    model.vq.init_from_data(torch.cat(embeddings))
    
    # Phase 3: Fine-tune with RVQ
    print("\n  Phase 3: Fine-tuning with RVQ...")
    model.use_vq = True
    
    # Lower LR for stable RVQ training
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': model.vq.parameters(), 'lr': 5e-5},
        {'params': model.decoder.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epochs)
    
    best_val_r2_vq = -float('inf')
    best_state = None
    patience = 30
    no_improve = 0
    
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        
        epoch_loss = 0.0
        epoch_commitment = 0.0
        
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window, velocity)
            output['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += output['recon_loss'].item()
            if 'commitment_loss' in output and isinstance(output['commitment_loss'], torch.Tensor):
                epoch_commitment += output['commitment_loss'].item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        all_codes_per_layer = [set() for _ in range(len(model.vq.quantizers))]
        residual_norms = []
        
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['window'].to(device))
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(batch['velocity'])
                
                if 'all_indices' in output:
                    for i, indices in enumerate(output['all_indices']):
                        all_codes_per_layer[i].update(indices.cpu().numpy().tolist())
                
                if 'residual_norm' in output:
                    residual_norms.append(output['residual_norm'].item())
        
        val_r2 = r2_score(torch.cat(val_targets).numpy(), torch.cat(val_preds).numpy())
        
        if val_r2 > best_val_r2_vq:
            best_val_r2_vq = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if epoch % 10 == 0:
            codes_info = "/".join([str(len(s)) for s in all_codes_per_layer])
            res_norm = np.mean(residual_norms) if residual_norms else 0.0
            print(f"    Epoch {epoch:3d}: val_R²={val_r2:.4f} (best={best_val_r2_vq:.4f}), "
                  f"codes=[{codes_info}], res_norm={res_norm:.4f}")
        
        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n    Fine-tuning complete. Best VQ R² = {best_val_r2_vq:.4f}")
    return best_val_r2_vq


# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    print("\n" + "="*70)
    print("EXPERIMENT 12: Residual Vector Quantization (RVQ)")
    print("="*70)
    print("\nObjective: Break through Voronoi Ceiling with multi-stage quantization")
    print("Target: R² > 0.78 (beat raw LSTM baseline)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading MC_Maze dataset...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_maze = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    spike_counts = mc_maze.spike_counts
    velocities = mc_maze.velocities
    
    # Normalize
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
    
    # Configurations to test
    configs = [
        {
            'name': 'RVQ-4 (4 layers × 128 codes)',
            'num_quantizers': 4,
            'num_codes': 128,
            'rvq_type': 'standard',
            'layer_dropout': 0.0
        },
        {
            'name': 'RVQ-6 (6 layers × 128 codes)',
            'num_quantizers': 6,
            'num_codes': 128,
            'rvq_type': 'standard',
            'layer_dropout': 0.0
        },
        {
            'name': 'RVQ-8 (8 layers × 64 codes)',
            'num_quantizers': 8,
            'num_codes': 64,
            'rvq_type': 'standard',
            'layer_dropout': 0.0
        },
        {
            'name': 'Progressive RVQ-6 (with dropout)',
            'num_quantizers': 6,
            'num_codes': 128,
            'rvq_type': 'progressive',
            'layer_dropout': 0.1
        },
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {cfg['name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        model = RVQModel(
            n_channels=142,
            window_size=window_size,
            d_model=256,
            embedding_dim=128,
            num_quantizers=cfg['num_quantizers'],
            num_codes=cfg['num_codes'],
            num_layers=6,
            rvq_type=cfg['rvq_type'],
            layer_dropout=cfg['layer_dropout']
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")
        
        # Calculate effective vocabulary
        effective_vocab = cfg['num_codes'] ** min(cfg['num_quantizers'], 3)  # Conservative estimate
        print(f"Effective vocabulary (first 3 layers): {effective_vocab:,}")
        
        train_rvq_progressive(model, train_loader, val_loader, device,
                              pretrain_epochs=50, finetune_epochs=100)
        
        elapsed = time.time() - start_time
        
        # Final evaluation
        model.eval()
        test_preds, test_targets = [], []
        all_codes_per_layer = [set() for _ in range(cfg['num_quantizers'])]
        residual_norms = []
        
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch['window'].to(device))
                test_preds.append(output['velocity_pred'].cpu())
                test_targets.append(batch['velocity'])
                
                if 'all_indices' in output:
                    for i, indices in enumerate(output['all_indices']):
                        if i < len(all_codes_per_layer):
                            all_codes_per_layer[i].update(indices.cpu().numpy().tolist())
                
                if 'residual_norm' in output:
                    residual_norms.append(output['residual_norm'].item())
        
        test_preds = torch.cat(test_preds).numpy()
        test_targets = torch.cat(test_targets).numpy()
        
        r2 = r2_score(test_targets, test_preds)
        r2_vx = r2_score(test_targets[:, 0], test_preds[:, 0])
        r2_vy = r2_score(test_targets[:, 1], test_preds[:, 1])
        
        codes_per_layer = [len(s) for s in all_codes_per_layer]
        total_codes = sum(codes_per_layer)
        avg_residual = np.mean(residual_norms) if residual_norms else 0.0
        
        results.append({
            'name': cfg['name'],
            'r2': r2,
            'r2_vx': r2_vx,
            'r2_vy': r2_vy,
            'codes_per_layer': codes_per_layer,
            'total_codes': total_codes,
            'residual_norm': avg_residual,
            'time': elapsed
        })
        
        status = "🎯" if r2 >= 0.78 else ("📈" if r2 >= 0.77 else "")
        print(f"\n  Result: R²={r2:.4f} (vx={r2_vx:.4f}, vy={r2_vy:.4f})")
        print(f"  Codes per layer: {codes_per_layer}")
        print(f"  Final residual norm: {avg_residual:.4f}")
        print(f"  Time: {elapsed/60:.1f}min {status}")
        
        if r2 >= 0.78:
            save_path = Path(__file__).parent / 'models' / 'exp12_rvq_best.pt'
            torch.save(model.state_dict(), save_path)
            print(f"  🎉 TARGET ACHIEVED! Model saved to {save_path}")
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT 12 RESULTS SUMMARY: Residual Vector Quantization")
    print("="*70)
    
    print(f"\n{'Model':<40} {'R²':>8} {'vx':>8} {'vy':>8} {'Codes':>10}")
    print("-"*70)
    
    for r in sorted(results, key=lambda x: x['r2'], reverse=True):
        status = "🎯" if r['r2'] >= 0.78 else ("📈" if r['r2'] >= 0.77 else "")
        codes_str = "/".join([str(c) for c in r['codes_per_layer'][:3]]) + "..."
        print(f"{r['name']:<40} {r['r2']:>7.4f} {r['r2_vx']:>7.4f} {r['r2_vy']:>7.4f} "
              f"{codes_str:>10} {status}")
    
    print("\n" + "-"*70)
    print("Comparison to previous approaches:")
    print("  • Raw LSTM baseline:        R² = 0.78")
    print("  • Exp 10 (Deep Causal):     R² = 0.7727")
    print("  • Exp 11 (Gumbel VQ):       R² = 0.7127")
    print("="*70)
    
    # Analysis
    best = max(results, key=lambda x: x['r2'])
    print("\n📊 RVQ Analysis:")
    print(f"  • Best config: {best['name']}")
    print(f"  • Total unique codes used: {best['total_codes']}")
    print(f"  • Final residual norm: {best['residual_norm']:.4f}")
    print(f"  • Per-layer utilization: {best['codes_per_layer']}")
    
    # Check if RVQ fixed the Voronoi ceiling
    if best['r2'] > 0.77:
        improvement = best['r2'] - 0.7127
        print(f"\n✅ RVQ improved over single VQ by {improvement:.4f} R² points")
        print("   The residual quantization successfully broke through the Voronoi ceiling!")
    
    if best['r2'] >= 0.78:
        print(f"\n🎉 SUCCESS: RVQ achieved R² = {best['r2']:.4f} - LSTM BEATEN!")
    else:
        gap = 0.78 - best['r2']
        print(f"\n📈 Best: {best['name']} R² = {best['r2']:.4f} (gap to LSTM: {gap:.4f})")
        print("\nNext steps to consider:")
        print("  1. Increase number of RVQ layers (try 10-12)")
        print("  2. Use larger codebooks (256 codes)")
        print("  3. Add temporal modeling in decoder")
        print("  4. Combine with Test-Time Adaptation (TTA)")
    
    return results


if __name__ == "__main__":
    run_experiment()
