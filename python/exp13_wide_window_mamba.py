"""
Experiment 13: Wide-Window Mamba ("The Context Hammer")

RED TEAM CRITIQUE OF EXP 12:
============================
1. "Shuffled State Suicide": Passing Mamba hidden state across shuffled batches
   causes causal hallucination. The model learns to ignore the broken state and
   falls back to being a feed-forward network (R² = 0.68 = MLP ceiling).

2. "Short Sequence Tax": Mamba shines on LONG sequences. Using it on 10 steps
   is like buying a Ferrari to drive 5 meters.

BLUE TEAM PIVOT:
================
Don't try to remember history; just SEE it.

1. Increase Window Size: 10 bins (250ms) → 80 bins (2000ms)
   - A 2-second window captures the entire reaching movement
   - Mamba processes 80 steps as efficiently as 10 (linear scaling)
   - Transformers would struggle with quadratic attention costs

2. Stateless Training: Reset Mamba state at start of every batch (h_0 = 0)
   - Eliminates "Shuffled State" bug instantly
   - Model relies purely on 2 seconds of context within the window
   - Window is guaranteed to be contiguous and valid

3. Theory:
   - Trajectory Completeness: 250ms is a glimpse; 2000ms is the full trajectory
   - Vanishing Gradients: LSTMs struggle with 80 steps; Mamba handles it naturally
   - Computational Efficiency: Mamba's linear scaling makes this practical

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

# Use relative path from project root
DATA_PATH = Path(__file__).parent.parent / "data" / "mc_maze.nwb"


# ============================================================
# Mamba (S6) Core Components (Stateless Version)
# ============================================================

class S6Layer(nn.Module):
    """
    Simplified S6 (Selective State Space) Layer.
    
    This is a STATELESS version - state is always initialized to zero
    at the start of each forward pass.
    
    SSM dynamics:
        h'(t) = A * h(t) + B * x(t)
        y(t) = C * h(t) + D * x(t)
    
    S6 makes B, C, Δ (discretization step) data-dependent.
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
        """
        STATELESS forward pass - always starts with h_0 = 0.
        
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            output: [batch, seq_len, d_model]
        """
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
        """
        Stateless selective scan - h_0 = 0.
        
        For long sequences, this is where Mamba shines:
        - Linear complexity O(L) instead of O(L²) for attention
        - Natural handling of long-range dependencies via SSM
        """
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


class WideWindowMambaEncoder(nn.Module):
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
# Vector Quantizer with Gumbel-Softmax
# ============================================================

class GumbelVQ(nn.Module):
    """Gumbel-Softmax Vector Quantizer with temperature annealing."""
    
    def __init__(
        self,
        num_codes: int = 256,
        embedding_dim: int = 128,
        temp_init: float = 1.0,
        temp_min: float = 0.3,
        commitment_cost: float = 0.1
    ):
        super().__init__()
        
        self.num_codes = num_codes
        self.embedding_dim = embedding_dim
        self.temp_min = temp_min
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Parameter(torch.randn(num_codes, embedding_dim))
        nn.init.xavier_uniform_(self.embeddings)
        
        self.logit_scale = nn.Parameter(torch.ones(1) * 10.0)
        self.register_buffer('temperature', torch.tensor(temp_init))
        self._initialized = False
    
    def init_from_data(self, z_all: torch.Tensor):
        """Initialize codebook with K-means."""
        z_np = z_all.detach().cpu().numpy()
        if len(z_np) > 10000:
            idx = np.random.choice(len(z_np), 10000, replace=False)
            z_np = z_np[idx]
        
        kmeans = KMeans(n_clusters=self.num_codes, n_init=10, max_iter=300, random_state=42)
        kmeans.fit(z_np)
        
        self.embeddings.data.copy_(torch.from_numpy(kmeans.cluster_centers_).float())
        self._initialized = True
        print(f"  VQ initialized with K-means: {self.num_codes} codes")
    
    def update_temperature(self, epoch: int, max_epochs: int):
        """Cosine annealing of temperature."""
        progress = min(epoch / max_epochs, 1.0)
        temp = self.temp_min + 0.5 * (1.0 - self.temp_min) * (1 + math.cos(math.pi * progress))
        self.temperature.fill_(temp)
    
    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
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
        
        commitment_loss = self.commitment_cost * F.mse_loss(z_q.detach(), z_e)
        
        return z_q, {
            'indices': indices,
            'perplexity': perplexity,
            'temperature': self.temperature.clone(),
            'commitment_loss': commitment_loss
        }


# ============================================================
# Full Model: Wide-Window Mamba VQ-VAE
# ============================================================

class WideWindowMambaVQVAE(nn.Module):
    """
    Wide-Window Mamba VQ-VAE.
    
    Key innovation: Process 2-second windows (80 bins) in a single stateless pass.
    This eliminates shuffle bugs and leverages Mamba's linear scaling.
    """
    
    def __init__(
        self,
        n_channels: int = 142,
        window_size: int = 80,  # 2 seconds at 40Hz
        d_model: int = 256,
        d_state: int = 16,
        num_layers: int = 6,
        embedding_dim: int = 128,
        num_codes: int = 256,
        output_dim: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.window_size = window_size
        
        # Encoder
        self.encoder = WideWindowMambaEncoder(
            n_channels=n_channels,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        # Vector Quantizer
        self.vq = GumbelVQ(
            num_codes=num_codes,
            embedding_dim=embedding_dim
        )
        
        # Decoder (simple velocity head)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        
        self.use_vq = False
    
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
                'perplexity': torch.tensor(256.0, device=device),
                'temperature': torch.tensor(1.0, device=device),
                'commitment_loss': torch.tensor(0.0, device=device)
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
            recon_loss = F.mse_loss(velocity_pred, targets)
            output['recon_loss'] = recon_loss
            output['total_loss'] = recon_loss + vq_info['commitment_loss']
        
        return output


# ============================================================
# Dataset
# ============================================================

class WideWindowDataset(Dataset):
    """
    Dataset for wide-window processing.
    
    Uses 80 bins (2 seconds) to capture full reaching trajectories.
    """
    
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
# Training Loop
# ============================================================

def train_wide_window_mamba(
    model: WideWindowMambaVQVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 60,
    lr: float = 3e-4
):
    """Progressive training: pretrain encoder, then add VQ."""
    
    print("\n" + "="*60)
    print("Training Wide-Window Mamba VQ-VAE")
    print("="*60)
    
    # ========================================
    # Phase 1: Pre-training (no VQ)
    # ========================================
    print("\n[Phase 1] Pre-training encoder + decoder (no VQ)...")
    
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
    print("\n[Phase 2] Initializing VQ codebook with K-means...")
    
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch in train_loader:
            window = batch['window'].to(device)
            z_e = model.encode(window)
            embeddings.append(z_e.cpu())
    
    all_embeddings = torch.cat(embeddings)
    model.vq.init_from_data(all_embeddings)
    
    # ========================================
    # Phase 3: Finetuning with VQ
    # ========================================
    print("\n[Phase 3] Finetuning with VQ...")
    
    model.use_vq = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, finetune_epochs)
    
    best_val_r2_vq = -float('inf')
    best_state = None
    
    for epoch in range(1, finetune_epochs + 1):
        model.train()
        model.vq.update_temperature(epoch, finetune_epochs)
        
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
        codes_used = set()
        
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                output = model(window)
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(batch['velocity'])
                codes_used.update(output['indices'].cpu().numpy().tolist())
        
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_r2 = r2_score(val_targets, val_preds)
        
        if val_r2 > best_val_r2_vq:
            best_val_r2_vq = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if epoch % 15 == 0 or epoch == 1:
            temp = model.vq.temperature.item()
            avg_loss = np.mean(train_losses)
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, val_R²={val_r2:.4f}, "
                  f"temp={temp:.3f}, codes={len(codes_used)}")
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n  Finetuning complete. Best VQ R² = {best_val_r2_vq:.4f}")
    
    return best_val_r2_vq


def run_experiment():
    """Run the Wide-Window Mamba experiment."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 13: Wide-Window Mamba ('The Context Hammer')")
    print("="*70)
    print("\nKey Innovation:")
    print("  • Window: 10 bins (250ms) → 80 bins (2000ms)")
    print("  • Stateless: Reset Mamba state each batch (no shuffle bugs)")
    print("  • Mamba processes 80 steps as efficiently as 10 (linear scaling)")
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
    
    print(f"  Spike counts: {spike_counts.shape}")
    print(f"  Velocities: {velocities.shape}")
    
    # ========================================
    # Experiment configurations
    # ========================================
    configs = [
        {
            'name': 'Mamba-6L Wide-80 (2s)',
            'window_size': 80,
            'num_layers': 6,
            'd_model': 256,
            'd_state': 16
        },
        {
            'name': 'Mamba-4L Wide-80 (2s)',
            'window_size': 80,
            'num_layers': 4,
            'd_model': 256,
            'd_state': 16
        },
        {
            'name': 'Mamba-6L Wide-40 (1s)',
            'window_size': 40,
            'num_layers': 6,
            'd_model': 256,
            'd_state': 16
        },
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {cfg['name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create dataset with this window size
        window_size = cfg['window_size']
        dataset = WideWindowDataset(spike_counts, velocities, window_size)
        
        # Train/val split
        n_train = int(0.8 * len(dataset))
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, len(dataset)))
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        print(f"  Window size: {window_size} bins ({window_size * 25}ms)")
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        # Create model
        model = WideWindowMambaVQVAE(
            n_channels=142,
            window_size=window_size,
            d_model=cfg['d_model'],
            d_state=cfg['d_state'],
            num_layers=cfg['num_layers'],
            embedding_dim=128,
            num_codes=256,
            dropout=0.1
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
        
        # Train
        best_r2 = train_wide_window_mamba(
            model, train_loader, val_loader, device,
            pretrain_epochs=50,
            finetune_epochs=60
        )
        
        elapsed = time.time() - start_time
        
        # Final evaluation
        model.eval()
        test_preds, test_targets = [], []
        codes_used = set()
        
        with torch.no_grad():
            for batch in val_loader:
                window = batch['window'].to(device)
                output = model(window)
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
            'window_size': window_size,
            'r2': r2,
            'r2_vx': r2_vx,
            'r2_vy': r2_vy,
            'codes': len(codes_used),
            'params': param_count,
            'time': elapsed
        })
        
        status = "🎯" if r2 >= 0.78 else ("📈" if r2 >= 0.77 else "")
        print(f"\n  Result: R²={r2:.4f} (vx={r2_vx:.4f}, vy={r2_vy:.4f}), "
              f"codes={len(codes_used)}, time={elapsed/60:.1f}min {status}")
        
        if r2 >= 0.78:
            save_path = Path(__file__).parent / 'models' / 'exp13_wide_mamba.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'r2': r2,
            }, save_path)
            print(f"  🎉 TARGET ACHIEVED! Model saved to {save_path}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 13 RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Model':<30} {'Window':>8} {'R²':>8} {'vx':>8} {'vy':>8} {'Codes':>8}")
    print("-"*75)
    
    for r in sorted(results, key=lambda x: x['r2'], reverse=True):
        status = "🎯" if r['r2'] >= 0.78 else ("📈" if r['r2'] >= 0.77 else "")
        print(f"{r['name']:<30} {r['window_size']:>8} {r['r2']:>7.4f} {r['r2_vx']:>7.4f} "
              f"{r['r2_vy']:>7.4f} {r['codes']:>8} {status}")
    
    print("\n" + "-"*70)
    print("Baselines:")
    print("  • Raw LSTM (10-step window): R² = 0.78")
    print("  • Previous best (RVQ-4):     R² = 0.7757")
    print("  • Exp 12 Mamba (broken):     R² = 0.68 (shuffle bug)")
    print("="*70)
    
    best = max(results, key=lambda x: x['r2'])
    if best['r2'] >= 0.78:
        print(f"\n✅ SUCCESS: {best['name']} achieved R² = {best['r2']:.4f} - LSTM BEATEN!")
        print("\nKey insight: Wide context window (2s) + stateless Mamba")
        print("gives the model enough trajectory information to decode accurately.")
    else:
        gap = 0.78 - best['r2']
        print(f"\n📈 Best: {best['name']} R² = {best['r2']:.4f} (gap to LSTM: {gap:.4f})")
        print("\nNext steps:")
        print("  • Try even wider windows (120 bins = 3 seconds)")
        print("  • Add auxiliary losses (spike reconstruction)")
        print("  • Experiment with bidirectional Mamba")
    
    return results


if __name__ == "__main__":
    run_experiment()
