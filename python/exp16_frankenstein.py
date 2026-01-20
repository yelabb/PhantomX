"""
Experiment 16: The Frankenstein Pivot
=====================================

RED TEAM CRITIQUE (What's Wrong):
---------------------------------
1. FSQ (Exp 14-15) was a theoretical overreach - too brittle for high-variance
   kinematics regression. It collapsed immediately (Perplexity=5).
   
2. Returning to Exp 12 (Transformer + RVQ) will reproduce R²=0.776 - same ceiling.
   The Transformer only sees 250ms (10 bins). LSTM baseline sees seconds of history.

3. The 0.4% gap requires BOTH: precision (RVQ) AND temporal context (2s window).

BLUE TEAM SOLUTION (The Frankenstein Pivot):
-------------------------------------------
Combine the best components from each experiment:

From Exp 12: RVQ-4 (4 layers × 128 codes) - provides precision
From Exp 13: Mamba Backbone (80 bins = 2.0s) - provides temporal context
From Exp 14: Stateless Training (reset state every batch) - fixes shuffle bug
From Red Team: Huber Loss - prevents gradient explosions that killed Exp 14

Architecture:
    Input (2s window) → Mamba Encoder → RVQ-4 → MLP Decoder → Velocity

TARGET: R² > 0.78 (Beat LSTM baseline)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from einops import rearrange, repeat
import math
import time
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent))
from phantomx.tokenizer import SpikeTokenizer
from phantomx.data import MCMazeDataset

DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"


# ============================================================
# Mamba (S6) Core Components (Stateless Version)
# ============================================================

class S6Layer(nn.Module):
    """
    Simplified S6 (Selective State Space) Layer.
    
    STATELESS version - state is always initialized to zero at the start
    of each forward pass. This eliminates the "Shuffled State Suicide" bug.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        
        # Input projection: x -> (z, x_proj)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # 1D convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=conv_bias
        )
        
        # S6 projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_min)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias.data = inv_dt
        
        # State matrix A (diagonal, negative for stability)
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        
        # Skip connection D
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """STATELESS forward pass - always starts with h_0 = 0."""
        batch, seq_len, _ = x.shape
        
        # Project input
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Apply 1D convolution
        x_conv = rearrange(x_proj, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = rearrange(x_conv, "b d l -> b l d")
        x_conv = F.silu(x_conv)
        
        # S6: data-dependent Δ, B, C
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(
            x_dbl, 
            [self.dt_rank, self.d_state, self.d_state], 
            dim=-1
        )
        
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)
        
        A = -torch.exp(self.A_log)
        
        # Selective scan (STATELESS - h_0 = 0)
        y = self._selective_scan_stateless(x_conv, dt, A, B, C, self.D)
        
        # Apply gate and output projection
        y = y * F.silu(z)
        output = self.out_proj(y)
        
        return output
    
    def _selective_scan_stateless(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Stateless selective scan - h_0 = 0."""
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize hidden state to zero
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            dt_t = dt[:, t, :]
            B_t = B[:, t, :]
            C_t = C[:, t, :]
            
            # Discretize
            dt_A = dt_t.unsqueeze(-1) * A.unsqueeze(0)
            A_bar = torch.exp(dt_A)
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            
            # State update
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            
            # Output
            y_t = torch.einsum("bdn,bn->bd", h, C_t) + D * x_t
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    """Mamba block with residual connection and layer norm."""
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s6 = S6Layer(d_model, d_state=d_state, expand=expand)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.s6(x)
        x = self.dropout(x)
        return x + residual


class MambaEncoder(nn.Module):
    """
    Stateless Mamba encoder for wide-window processing.
    
    Key design: Process 2-second windows (80 bins) in a single forward pass.
    No state is passed between batches - eliminates shuffle bugs.
    """
    
    def __init__(
        self,
        n_channels: int = 142,
        d_model: int = 256,
        d_state: int = 16,
        num_layers: int = 6,
        expand: int = 2,
        dropout: float = 0.1,
        output_dim: int = 128,
        max_len: int = 100
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(n_channels, d_model)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, expand, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_channels] spike windows
            
        Returns:
            z: [batch, output_dim] encoded representation (from last timestep)
        """
        B, T, C = x.shape
        
        # Input projection + positional embedding
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        
        # Process through Mamba layers (stateless)
        for layer in self.layers:
            x = layer(x)
        
        # Output: take last timestep
        x = self.ln_final(x)
        z = self.output_proj(x[:, -1, :])
        
        return z


# ============================================================
# Residual Vector Quantization (RVQ) - From Exp 12
# ============================================================

class ResidualVQLayer(nn.Module):
    """Single layer of residual vector quantization with EMA updates."""
    
    def __init__(self, num_codes: int = 128, embedding_dim: int = 128,
                 commitment_cost: float = 0.25, decay: float = 0.99, epsilon: float = 1e-5):
        super().__init__()
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        self.embedding = nn.Embedding(num_codes, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        
        self.register_buffer('ema_cluster_size', torch.zeros(num_codes))
        self.register_buffer('ema_embedding', self.embedding.weight.data.clone())
        self.register_buffer('initialized', torch.tensor(False))
    
    def init_from_data(self, z: torch.Tensor):
        """Initialize codebook using K-means clustering."""
        z_np = z.detach().cpu().numpy()
        n_samples = len(z_np)
        
        n_clusters = min(self.num_codes, n_samples // 2)
        if n_clusters < self.num_codes:
            print(f"    Warning: Using {n_clusters} clusters (not enough samples)")
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(z_np)
        
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
        z_flat = z.view(-1, self.embedding_dim)
        
        z_norm = torch.sum(z_flat ** 2, dim=1, keepdim=True)
        e_norm = torch.sum(self.embedding.weight ** 2, dim=1)
        distances = z_norm + e_norm.unsqueeze(0) - 2 * z_flat @ self.embedding.weight.t()
        
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices)
        
        commitment_loss = F.mse_loss(z_q.detach(), z_flat) * self.commitment_cost
        
        z_q_st = z_flat + (z_q - z_flat).detach()
        
        if self.training:
            self._ema_update(z_flat, indices)
        
        encodings = F.one_hot(indices, self.num_codes).float()
        avg_probs = encodings.mean(0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        residual = z_flat - z_q_st
        
        return z_q_st.view_as(z), residual.view_as(z), {
            'indices': indices,
            'commitment_loss': commitment_loss,
            'perplexity': perplexity
        }
    
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        with torch.no_grad():
            encodings = F.one_hot(indices, self.num_codes).float()
            
            self.ema_cluster_size = self.decay * self.ema_cluster_size + \
                                    (1 - self.decay) * encodings.sum(0)
            
            n = self.ema_cluster_size.sum()
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon) /
                (n + self.num_codes * self.epsilon) * n
            )
            
            dw = encodings.t() @ z
            self.ema_embedding = self.decay * self.ema_embedding + (1 - self.decay) * dw
            
            self.embedding.weight.data = self.ema_embedding / self.ema_cluster_size.unsqueeze(1)


class ResidualVectorQuantizer(nn.Module):
    """
    Multi-stage Residual Vector Quantizer (RVQ-4).
    
    Applies N quantization stages in cascade:
        z_q1 = Q1(z), r1 = z - z_q1
        z_q2 = Q2(r1), r2 = r1 - z_q2
        ...
        final = z_q1 + z_q2 + ... + z_qN
    """
    
    def __init__(self, num_quantizers: int = 4, num_codes: int = 128, 
                 embedding_dim: int = 128, commitment_cost: float = 0.25):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        
        self.quantizers = nn.ModuleList([
            ResidualVQLayer(num_codes, embedding_dim, commitment_cost)
            for _ in range(num_quantizers)
        ])
    
    def init_from_data(self, z_all: torch.Tensor):
        """Initialize all codebooks from data using residual clustering."""
        device = z_all.device
        residual = z_all.clone()
        for i, quantizer in enumerate(self.quantizers):
            print(f"    Initializing RVQ layer {i+1}/{self.num_quantizers}...")
            quantizer.init_from_data(residual)
            
            with torch.no_grad():
                residual = residual.to(device)
                z_q, new_residual, _ = quantizer(residual)
                residual = new_residual.to(device)
    
    def forward(self, z: torch.Tensor) -> tuple:
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
# The Frankenstein Model: Mamba + RVQ-4
# ============================================================

class FrankensteinModel(nn.Module):
    """
    The "Best of Both Worlds" architecture.
    
    Components:
    - Mamba Encoder (2s window, stateless) from Exp 13
    - RVQ-4 (4 layers × 128 codes) from Exp 12
    - MLP Decoder
    - Huber Loss for gradient stability
    """
    
    def __init__(
        self,
        n_channels: int = 142,
        window_size: int = 80,  # 2 seconds at 40Hz
        d_model: int = 256,
        d_state: int = 16,
        num_layers: int = 6,
        embedding_dim: int = 128,
        num_quantizers: int = 4,
        num_codes: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.window_size = window_size
        
        # Mamba Encoder (from Exp 13)
        self.encoder = MambaEncoder(
            n_channels=n_channels,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            output_dim=embedding_dim,
            dropout=dropout,
            max_len=window_size + 10
        )
        
        # RVQ-4 (from Exp 12)
        self.vq = ResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_codes=num_codes,
            embedding_dim=embedding_dim
        )
        
        # MLP Decoder
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        
        self.use_vq = False
        
        # Huber loss (from Red Team recommendation)
        self.huber_loss = nn.HuberLoss(delta=1.0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Dict:
        batch_size = x.size(0)
        device = x.device
        
        # Encode
        z_e = self.encode(x)
        
        # Vector quantization
        if self.use_vq:
            z_q, vq_info = self.vq(z_e)
        else:
            z_q = z_e
            vq_info = {
                'indices': torch.zeros(batch_size, dtype=torch.long, device=device),
                'perplexity': torch.tensor(0.0, device=device),
                'commitment_loss': torch.tensor(0.0, device=device),
                'residual_norm': torch.tensor(0.0, device=device)
            }
        
        # Decode
        velocity_pred = self.decoder(z_q)
        
        output = {
            'velocity_pred': velocity_pred,
            'z_e': z_e,
            'z_q': z_q,
            **vq_info
        }
        
        if targets is not None:
            # Use Huber Loss for gradient stability
            recon_loss = self.huber_loss(velocity_pred, targets)
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

class WideWindowDataset(Dataset):
    """Dataset for wide-window processing (80 bins = 2 seconds)."""
    
    def __init__(
        self,
        spike_counts: np.ndarray,
        velocities: np.ndarray,
        window_size: int = 80
    ):
        n = len(spike_counts) - window_size + 1
        
        # Pre-compute all windows
        self.windows = np.stack([spike_counts[i:i+window_size] for i in range(n)])
        
        # Velocity at the END of each window
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

def train_frankenstein(
    model: FrankensteinModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 100,
    lr: float = 3e-4
):
    """
    Progressive training for Frankenstein model.
    
    Phase 1: Pre-train Mamba encoder + decoder (no VQ)
    Phase 2: Initialize RVQ codebooks with residual K-means
    Phase 3: Fine-tune full model with RVQ
    """
    
    print("\n" + "="*60)
    print("Training Frankenstein Model (Mamba + RVQ-4)")
    print("="*60)
    
    # ========================================
    # Phase 1: Pre-training (no VQ)
    # ========================================
    print("\n[Phase 1] Pre-training Mamba encoder + decoder (no VQ)...")
    
    model.use_vq = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pretrain_epochs)
    
    best_val_r2 = -float('inf')
    
    for epoch in range(1, pretrain_epochs + 1):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window, velocity)
            loss = output['total_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                output = model(window)
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(batch['velocity'])
        
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_r2 = r2_score(val_targets, val_preds)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
        
        if epoch % 10 == 0 or epoch == 1:
            avg_loss = np.mean(train_losses)
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, val_R²={val_r2:.4f} (best={best_val_r2:.4f})")
    
    print(f"\n  Pre-training complete. Best R² = {best_val_r2:.4f}")
    
    # ========================================
    # Phase 2: K-means initialization
    # ========================================
    print("\n[Phase 2] Initializing RVQ codebooks with residual K-means...")
    
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch in train_loader:
            window = batch['window'].to(device)
            z_e = model.encode(window)
            embeddings.append(z_e)  # Keep on GPU
    
    all_embeddings = torch.cat(embeddings)
    model.vq.init_from_data(all_embeddings)
    
    # ========================================
    # Phase 3: Finetuning with RVQ
    # ========================================
    print("\n[Phase 3] Finetuning with RVQ-4...")
    
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
        
        if epoch % 10 == 0 or epoch == 1:
            codes_info = "/".join([str(len(s)) for s in all_codes_per_layer])
            res_norm = np.mean(residual_norms) if residual_norms else 0.0
            print(f"  Epoch {epoch:3d}: val_R²={val_r2:.4f} (best={best_val_r2_vq:.4f}), "
                  f"codes=[{codes_info}], res_norm={res_norm:.4f}")
        
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n  Finetuning complete. Best VQ R² = {best_val_r2_vq:.4f}")
    return best_val_r2_vq, best_val_r2


# ============================================================
# LSTM Baseline (for comparison)
# ============================================================

class LSTMBaseline(nn.Module):
    """Simple LSTM baseline for velocity decoding."""
    
    def __init__(self, n_channels=142, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.decoder(out[:, -1, :])


def train_lstm_baseline(train_loader, val_loader, device, epochs=50, window_size=80):
    """Train LSTM baseline with same window size for fair comparison."""
    model = LSTMBaseline(n_channels=142).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_r2 = -float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch['window'].to(device))
            loss = F.mse_loss(pred, batch['velocity'].to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['window'].to(device))
                preds.append(pred.cpu())
                targets.append(batch['velocity'])
        
        r2 = r2_score(torch.cat(targets).numpy(), torch.cat(preds).numpy())
        if r2 > best_r2:
            best_r2 = r2
    
    return best_r2


# ============================================================
# Main Experiment
# ============================================================

def run_experiment():
    print("\n" + "="*70)
    print("EXPERIMENT 16: The Frankenstein Pivot")
    print("="*70)
    print("\nArchitecture: Mamba Encoder (2s) + RVQ-4 + MLP Decoder + Huber Loss")
    print("\nComponents:")
    print("  • From Exp 12: RVQ-4 (4 layers × 128 codes) - precision")
    print("  • From Exp 13: Mamba Backbone (80 bins = 2.0s) - temporal context")
    print("  • From Exp 14: Stateless Training (no shuffle bug)")
    print("  • From Red Team: Huber Loss (gradient stability)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # ========================================
    # Load data
    # ========================================
    print("\nLoading MC_Maze dataset...")
    tokenizer = SpikeTokenizer(n_channels=142)
    mc_maze = MCMazeDataset(data_path=DATA_PATH, tokenizer=tokenizer)
    
    spike_counts = mc_maze.spike_counts
    velocities = mc_maze.velocities
    
    # Normalize
    spike_counts = (spike_counts - spike_counts.mean(0)) / (spike_counts.std(0) + 1e-6)
    velocities = (velocities - velocities.mean(0)) / (velocities.std(0) + 1e-6)
    
    print(f"  Spike counts: {spike_counts.shape}")
    print(f"  Velocities: {velocities.shape}")
    
    # ========================================
    # Configuration
    # ========================================
    config = {
        'window_size': 80,  # 2 seconds at 40Hz
        'd_model': 256,
        'd_state': 16,
        'num_layers': 6,
        'embedding_dim': 128,
        'num_quantizers': 4,
        'num_codes': 128,
        'dropout': 0.1,
        'pretrain_epochs': 50,
        'finetune_epochs': 100
    }
    
    print(f"\nConfiguration:")
    print(f"  Window: {config['window_size']} bins ({config['window_size'] * 25}ms)")
    print(f"  Mamba: {config['num_layers']} layers, d_model={config['d_model']}")
    print(f"  RVQ: {config['num_quantizers']} layers × {config['num_codes']} codes")
    
    # Create dataset
    dataset = WideWindowDataset(spike_counts, velocities, config['window_size'])
    
    n_train = int(0.8 * len(dataset))
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, len(dataset)))
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"\nData: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # ========================================
    # Create and train model
    # ========================================
    model = FrankensteinModel(
        n_channels=142,
        window_size=config['window_size'],
        d_model=config['d_model'],
        d_state=config['d_state'],
        num_layers=config['num_layers'],
        embedding_dim=config['embedding_dim'],
        num_quantizers=config['num_quantizers'],
        num_codes=config['num_codes'],
        dropout=config['dropout']
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    start_time = time.time()
    
    best_r2_vq, best_r2_pretrain = train_frankenstein(
        model, train_loader, val_loader, device,
        pretrain_epochs=config['pretrain_epochs'],
        finetune_epochs=config['finetune_epochs']
    )
    
    elapsed = time.time() - start_time
    
    # ========================================
    # Final evaluation
    # ========================================
    model.eval()
    test_preds, test_targets = [], []
    all_codes_per_layer = [set() for _ in range(config['num_quantizers'])]
    
    with torch.no_grad():
        for batch in val_loader:
            output = model(batch['window'].to(device))
            test_preds.append(output['velocity_pred'].cpu())
            test_targets.append(batch['velocity'])
            
            if 'all_indices' in output:
                for i, indices in enumerate(output['all_indices']):
                    all_codes_per_layer[i].update(indices.cpu().numpy().tolist())
    
    test_preds = torch.cat(test_preds).numpy()
    test_targets = torch.cat(test_targets).numpy()
    
    final_r2 = r2_score(test_targets, test_preds)
    r2_vx = r2_score(test_targets[:, 0], test_preds[:, 0])
    r2_vy = r2_score(test_targets[:, 1], test_preds[:, 1])
    
    codes_per_layer = [len(s) for s in all_codes_per_layer]
    total_codes = sum(codes_per_layer)
    
    # ========================================
    # Train LSTM baseline with same window for fair comparison
    # ========================================
    print("\n" + "="*60)
    print("Training LSTM Baseline (same 2s window for fair comparison)...")
    print("="*60)
    
    lstm_r2 = train_lstm_baseline(train_loader, val_loader, device, epochs=50, 
                                   window_size=config['window_size'])
    
    # ========================================
    # Results
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 16 RESULTS")
    print("="*70)
    
    print(f"\nFrankenstein Model (Mamba + RVQ-4):")
    print(f"  Pre-training R² (encoder only): {best_r2_pretrain:.4f}")
    print(f"  Final R² (with RVQ):            {final_r2:.4f}")
    print(f"  R² vx:                          {r2_vx:.4f}")
    print(f"  R² vy:                          {r2_vy:.4f}")
    print(f"  Codes per layer:                {codes_per_layer}")
    print(f"  Total codes used:               {total_codes}")
    print(f"  Training time:                  {elapsed/60:.1f} min")
    
    print(f"\nBaselines:")
    print(f"  LSTM (same 2s window):          {lstm_r2:.4f}")
    print(f"  LSTM (10-step, 250ms):          0.7800 (historical)")
    print(f"  RVQ-4 Transformer (Exp 12):     0.7760")
    
    print("\n" + "-"*70)
    
    if final_r2 >= 0.78:
        gap_pct = (final_r2 - 0.78) / 0.78 * 100
        print(f"✅ SUCCESS! R² = {final_r2:.4f} - LSTM BEATEN by {gap_pct:.2f}%!")
        print("\nKey insight: Wide context (2s) + RVQ precision = breakthrough")
        
        # Save model
        save_path = Path(__file__).parent / 'models' / 'exp16_frankenstein_best.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'r2': final_r2,
            'r2_vx': r2_vx,
            'r2_vy': r2_vy,
            'pretrain_r2': best_r2_pretrain,
            'codes_per_layer': codes_per_layer
        }, save_path)
        print(f"Model saved to {save_path}")
    else:
        gap = 0.78 - final_r2
        gap_pct = gap / 0.78 * 100
        print(f"❌ Gap to LSTM: {gap:.4f} ({gap_pct:.1f}%)")
        
        if final_r2 > 0.776:
            print(f"📈 But improved over Exp 12 (RVQ-4 Transformer): {final_r2:.4f} > 0.776")
    
    print("="*70)
    
    print("\n✓ Experiment 16 complete!")
    
    return {
        'final_r2': final_r2,
        'r2_vx': r2_vx,
        'r2_vy': r2_vy,
        'pretrain_r2': best_r2_pretrain,
        'lstm_r2': lstm_r2,
        'codes_per_layer': codes_per_layer,
        'elapsed': elapsed
    }


if __name__ == "__main__":
    run_experiment()
