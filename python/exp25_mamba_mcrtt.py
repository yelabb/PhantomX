"""
Experiment 25: Mamba on MC_RTT ("The Navigation Filter")

HYPOTHESIS:
===========
MC_RTT is fundamentally different from MC_Maze:
- MC_Maze: Discrete reaching movements with pauses → context is noise
- MC_RTT: Continuous random target tracking → context IS the trajectory

What failed on MC_Maze becomes STRENGTH on MC_RTT:

1. L'Architecture: Mamba (State Space Model)
   - From Exp 13, but NOW the long context matters
   - Window: 2 seconds (80 bins at 40Hz)
   - Mamba acts as a "Neural Kalman Filter" - smoothing trajectory over time

2. Stateful Training (CRITICAL):
   - NO SHUFFLE - batches are consecutive in time
   - Model maintains hidden state h_t across the session
   - This is what makes Mamba shine on continuous data

3. Why this will work:
   - MC_Maze: "Where am I going?" (instant decision)
   - MC_RTT: "Where have I been + where am I going?" (trajectory integration)
   - The 2-second context captures the target trajectory being tracked

TARGET: R² > 0.70 (establish baseline on new dataset)
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

# Data path for MC_RTT
DATA_PATH = Path(__file__).parent.parent / "data" / "mc_rtt.nwb"


# ============================================================
# MC_RTT Data Loader
# ============================================================

def load_mc_rtt_from_nwb(file_path: str, bin_size_ms: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spike counts and kinematics from MC_RTT NWB file.
    
    MC_RTT specifics:
    - 130 neural units (vs 142 in MC_Maze)
    - ~649 seconds of continuous recording at 1000Hz
    - finger_vel instead of hand_vel
    - Continuous target tracking task (not discrete reaches)
    
    Args:
        file_path: Path to the NWB file
        bin_size_ms: Time bin size in milliseconds (default 25ms = 40Hz)
        
    Returns:
        spike_counts: [T, n_channels] binned spike counts
        velocities: [T, 2] finger velocity (vx, vy)
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError:
        raise ImportError("PyNWB is required. Install with: pip install pynwb")
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NWB file not found: {file_path}")
    
    print(f"Loading MC_RTT from: {file_path}")
    
    with NWBHDF5IO(str(path), mode='r', load_namespaces=True) as io:
        nwb = io.read()
        
        # Get neural units
        units = nwb.units
        n_channels = len(units.id[:])
        print(f"  Found {n_channels} neural units")
        
        # Get behavior data
        behavior = nwb.processing.get('behavior')
        if behavior is None:
            raise RuntimeError("No 'behavior' processing module found")
        
        # Get finger velocity (MC_RTT uses finger_vel, not hand_vel)
        finger_vel_container = behavior.data_interfaces.get('finger_vel')
        if finger_vel_container is None:
            raise RuntimeError("No finger_vel found in behavior")
        
        finger_vel = finger_vel_container.data[:]  # [T, 2] (vx, vy)
        rate = finger_vel_container.rate  # 1000 Hz
        n_samples = len(finger_vel)
        duration = n_samples / rate
        
        print(f"  Raw data: {n_samples} samples at {rate}Hz = {duration:.1f}s")
        
        # Bin size in samples
        bin_samples = int(bin_size_ms * rate / 1000)
        n_bins = n_samples // bin_samples
        
        print(f"  Binning: {bin_samples} samples/bin → {n_bins} bins at {1000/bin_size_ms:.0f}Hz")
        
        # Bin spikes
        spike_counts = np.zeros((n_bins, n_channels), dtype=np.float32)
        
        for unit_idx in range(n_channels):
            spike_times = units.get_unit_spike_times(unit_idx)
            if spike_times is not None and len(spike_times) > 0:
                # Convert spike times to bin indices
                bin_indices = (spike_times * 1000 / bin_size_ms).astype(np.int32)
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                
                # Count spikes per bin
                for bin_idx in bin_indices:
                    spike_counts[bin_idx, unit_idx] += 1
        
        # Bin velocities (average within each bin)
        velocities = np.zeros((n_bins, 2), dtype=np.float32)
        for i in range(n_bins):
            start_idx = i * bin_samples
            end_idx = start_idx + bin_samples
            velocities[i] = finger_vel[start_idx:end_idx].mean(axis=0)
        
        print(f"  Loaded: spike_counts {spike_counts.shape}, velocities {velocities.shape}")
        
        # Basic stats
        mean_rate = spike_counts.mean() * (1000 / bin_size_ms)
        print(f"  Mean firing rate: {mean_rate:.2f} spikes/s")
        
        return spike_counts, velocities


# ============================================================
# Mamba (S6) Core Components - STATEFUL Version
# ============================================================

class S6LayerStateful(nn.Module):
    """
    Stateful S6 (Selective State Space) Layer.
    
    CRITICAL DIFFERENCE from Exp 13:
    - Maintains hidden state h_t between forward calls
    - For MC_RTT's continuous navigation, this state IS the trajectory memory
    
    SSM dynamics:
        h'(t) = A * h(t) + B * x(t)
        y(t) = C * h(t) + D * x(t)
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
        
        # Hidden state buffer (for stateful mode)
        self.register_buffer('h_state', None)
    
    def reset_state(self, batch_size: int, device: torch.device):
        """Reset hidden state to zeros."""
        self.h_state = torch.zeros(
            batch_size, self.d_inner, self.d_state, 
            device=device, dtype=torch.float32
        )
    
    def forward(self, x: torch.Tensor, stateful: bool = True) -> torch.Tensor:
        """
        Forward pass with optional state persistence.
        
        Args:
            x: [batch, seq_len, d_model]
            stateful: If True, maintain state between calls
            
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        device = x.device
        
        # Initialize state if needed
        if self.h_state is None or self.h_state.size(0) != batch:
            self.reset_state(batch, device)
        
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
        
        # Selective scan with state
        y = self._selective_scan(x_conv, dt, A, B, C, self.D, stateful)
        
        # Apply gate and output projection
        y = y * F.silu(z)
        output = self.out_proj(y)
        
        return output
    
    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        stateful: bool = True
    ) -> torch.Tensor:
        """
        Selective scan with optional state persistence.
        
        For MC_RTT: This is where the "trajectory memory" lives.
        The hidden state h accumulates information about the movement history.
        """
        batch, seq_len, d_inner = x.shape
        
        # Use persistent state or start fresh
        if stateful and self.h_state is not None:
            h = self.h_state.clone()
        else:
            h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            dt_t = dt[:, t, :]
            B_t = B[:, t, :]
            C_t = C[:, t, :]
            
            # Discretize with numerical stability
            dt_A = dt_t.unsqueeze(-1) * A.unsqueeze(0)
            dt_A = torch.clamp(dt_A, min=-20, max=0)  # Prevent overflow
            A_bar = torch.exp(dt_A)
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
            
            # State update: h' = A*h + B*x
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            
            # Clamp hidden state to prevent explosion
            h = torch.clamp(h, min=-100, max=100)
            
            # Output: y = C*h + D*x
            y_t = torch.einsum("bdn,bn->bd", h, C_t) + D * x_t
            outputs.append(y_t)
        
        # Save state for next call (if stateful)
        if stateful:
            # Clamp before saving
            h_clamped = torch.clamp(h, min=-100, max=100)
            self.h_state = h_clamped.detach()  # Detach to avoid gradient through time beyond window
        
        return torch.stack(outputs, dim=1)


class MambaBlockStateful(nn.Module):
    """Stateful Mamba block with residual connection and layer norm."""
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s6 = S6LayerStateful(d_model, d_state=d_state, expand=expand)
        self.dropout = nn.Dropout(dropout)
    
    def reset_state(self, batch_size: int, device: torch.device):
        self.s6.reset_state(batch_size, device)
    
    def forward(self, x: torch.Tensor, stateful: bool = True) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.s6(x, stateful=stateful)
        x = self.dropout(x)
        return x + residual


class StatefulMambaEncoder(nn.Module):
    """
    Stateful Mamba encoder for continuous trajectory tracking.
    
    Key difference from Exp 13:
    - Hidden state persists across batches (no reset between batches)
    - Acts as a "Neural Kalman Filter" for trajectory estimation
    """
    
    def __init__(
        self,
        n_channels: int = 130,  # MC_RTT has 130 units
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
        
        # Stateful Mamba layers
        self.layers = nn.ModuleList([
            MambaBlockStateful(d_model, d_state, expand, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim)
        )
    
    def reset_state(self, batch_size: int, device: torch.device):
        """Reset all layer states."""
        for layer in self.layers:
            layer.reset_state(batch_size, device)
    
    def forward(self, x: torch.Tensor, stateful: bool = True) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, n_channels] spike windows
            stateful: Maintain state between calls
            
        Returns:
            z: [batch, output_dim] encoded representation (from last timestep)
        """
        B, T, C = x.shape
        
        # Input projection + positional embedding
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        
        # Process through Mamba layers (stateful)
        for layer in self.layers:
            x = layer(x, stateful=stateful)
        
        # Output: take last timestep
        x = self.ln_final(x)
        z = self.output_proj(x[:, -1, :])
        
        return z


# ============================================================
# Velocity Decoder (No VQ for initial experiment)
# ============================================================

class MambaVelocityDecoder(nn.Module):
    """
    Stateful Mamba model for velocity decoding.
    
    For MC_RTT: No VQ initially - focus on proving the stateful
    Mamba hypothesis before adding discretization.
    """
    
    def __init__(
        self,
        n_channels: int = 130,
        window_size: int = 80,  # 2 seconds at 40Hz
        d_model: int = 256,
        d_state: int = 16,
        num_layers: int = 6,
        embedding_dim: int = 128,
        output_dim: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.window_size = window_size
        
        # Encoder
        self.encoder = StatefulMambaEncoder(
            n_channels=n_channels,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            output_dim=embedding_dim,
            dropout=dropout
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
    
    def reset_state(self, batch_size: int, device: torch.device):
        """Reset encoder state (call at start of each session/epoch)."""
        self.encoder.reset_state(batch_size, device)
    
    def forward(
        self,
        x: torch.Tensor,
        stateful: bool = True,
        targets: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, n_channels] spike windows
            stateful: Maintain state between calls
            targets: Optional velocity targets for loss computation
        """
        # Encode
        z = self.encoder(x, stateful=stateful)
        
        # Decode
        velocity_pred = self.decoder(z)
        
        output = {
            'velocity_pred': velocity_pred,
            'z': z
        }
        
        if targets is not None:
            loss = F.mse_loss(velocity_pred, targets)
            output['loss'] = loss
        
        return output


# ============================================================
# Dataset for Sequential (Non-Shuffled) Training
# ============================================================

class SequentialWindowDataset(Dataset):
    """
    Dataset for sequential window processing.
    
    CRITICAL: This dataset maintains temporal order.
    Use with DataLoader(shuffle=False) for stateful training.
    """
    
    def __init__(
        self,
        spike_counts: np.ndarray,
        velocities: np.ndarray,
        window_size: int = 80,
        normalize: bool = True
    ):
        self.window_size = window_size
        
        # Normalize spike counts
        if normalize:
            self.spike_mean = spike_counts.mean(axis=0, keepdims=True)
            self.spike_std = spike_counts.std(axis=0, keepdims=True) + 1e-6
            spike_counts = (spike_counts - self.spike_mean) / self.spike_std
            
            # Normalize velocities
            self.vel_mean = velocities.mean(axis=0, keepdims=True)
            self.vel_std = velocities.std(axis=0, keepdims=True) + 1e-6
            velocities = (velocities - self.vel_mean) / self.vel_std
        
        n = len(spike_counts) - window_size + 1
        
        # Pre-compute all windows
        self.windows = np.stack([spike_counts[i:i+window_size] for i in range(n)])
        
        # Velocity at the END of each window
        self.velocities = velocities[window_size-1:window_size-1+n]
        
        # Store normalization params for denormalization
        self.normalized = normalize
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return {
            'window': torch.tensor(self.windows[idx], dtype=torch.float32),
            'velocity': torch.tensor(self.velocities[idx], dtype=torch.float32),
            'idx': idx  # For debugging temporal order
        }


# ============================================================
# Stateful Training Loop
# ============================================================

def train_stateful_mamba(
    model: MambaVelocityDecoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 3e-4,
    reset_state_every: int = 1  # Reset state every N epochs (1 = each epoch)
):
    """
    Stateful training loop.
    
    CRITICAL DIFFERENCES:
    1. DataLoader shuffle=False (temporal order preserved)
    2. Model state persists within epoch
    3. State reset at epoch boundaries
    """
    
    print("\n" + "="*60)
    print("Training Stateful Mamba on MC_RTT")
    print("="*60)
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  State reset: Every {reset_state_every} epoch(s)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_r2 = -float('inf')
    best_state = None
    history = {'train_loss': [], 'val_r2': [], 'val_r2_vx': [], 'val_r2_vy': []}
    
    for epoch in range(1, epochs + 1):
        # ========================================
        # Training (Stateful)
        # ========================================
        model.train()
        train_losses = []
        
        # Reset state at start of each epoch
        if epoch % reset_state_every == 1 or reset_state_every == 1:
            # Get batch size from first batch
            first_batch = next(iter(train_loader))
            batch_size = first_batch['window'].size(0)
            model.reset_state(batch_size, device)
        
        for batch_idx, batch in enumerate(train_loader):
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            # Handle varying batch sizes (last batch might be smaller)
            if window.size(0) != batch_size:
                model.reset_state(window.size(0), device)
                batch_size = window.size(0)
            
            optimizer.zero_grad()
            output = model(window, stateful=True, targets=velocity)
            loss = output['loss']
            
            # Skip if loss is NaN
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at batch {batch_idx}, skipping...")
                model.reset_state(window.size(0), device)
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # More aggressive clipping
            optimizer.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # ========================================
        # Validation (Stateless for fair eval)
        # ========================================
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            # Reset state for validation
            first_batch = next(iter(val_loader))
            model.reset_state(first_batch['window'].size(0), device)
            
            for batch in val_loader:
                window = batch['window'].to(device)
                
                # Handle varying batch sizes
                if window.size(0) != first_batch['window'].size(0):
                    model.reset_state(window.size(0), device)
                
                # Use stateful=False for validation to avoid state leakage
                output = model(window, stateful=False)
                val_preds.append(output['velocity_pred'].cpu())
                val_targets.append(batch['velocity'])
        
        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        
        # Check for NaN and handle gracefully
        if np.isnan(val_preds).any() or np.isnan(val_targets).any():
            print(f"  WARNING: NaN detected in epoch {epoch}, resetting state...")
            val_r2, val_r2_vx, val_r2_vy = -1.0, -1.0, -1.0
            # Reset model state to recover
            model.reset_state(64, device)
        else:
            val_r2 = r2_score(val_targets, val_preds)
            val_r2_vx = r2_score(val_targets[:, 0], val_preds[:, 0])
            val_r2_vy = r2_score(val_targets[:, 1], val_preds[:, 1])
        
        # Track best
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Log
        avg_loss = np.mean(train_losses)
        history['train_loss'].append(avg_loss)
        history['val_r2'].append(val_r2)
        history['val_r2_vx'].append(val_r2_vx)
        history['val_r2_vy'].append(val_r2_vy)
        
        if epoch % 10 == 0 or epoch == 1:
            status = "🎯" if val_r2 >= 0.70 else ("📈" if val_r2 >= 0.60 else "")
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, R²={val_r2:.4f} "
                  f"(vx={val_r2_vx:.4f}, vy={val_r2_vy:.4f}) {status}")
    
    # Restore best model
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n  Training complete. Best R² = {best_val_r2:.4f}")
    
    return best_val_r2, history


def run_experiment():
    """Run the Stateful Mamba on MC_RTT experiment."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 25: Stateful Mamba on MC_RTT ('The Navigation Filter')")
    print("="*70)
    print("\nKey Hypothesis:")
    print("  • MC_RTT = Continuous navigation (context IS trajectory)")
    print("  • Mamba + Stateful training = Neural Kalman Filter")
    print("  • What failed on MC_Maze will SHINE on MC_RTT")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # ========================================
    # Load MC_RTT data
    # ========================================
    print("\nLoading MC_RTT dataset...")
    spike_counts, velocities = load_mc_rtt_from_nwb(str(DATA_PATH))
    
    n_channels = spike_counts.shape[1]
    print(f"  Channels: {n_channels}")
    print(f"  Duration: {len(spike_counts) / 40:.1f}s ({len(spike_counts)} bins)")
    
    # ========================================
    # Experiment configurations
    # ========================================
    configs = [
        {
            'name': 'Mamba-6L Stateful (2s window)',
            'window_size': 80,
            'num_layers': 6,
            'd_model': 256,
            'd_state': 16,
            'epochs': 100
        },
        {
            'name': 'Mamba-4L Stateful (2s window)',
            'window_size': 80,
            'num_layers': 4,
            'd_model': 256,
            'd_state': 16,
            'epochs': 100
        },
        {
            'name': 'Mamba-6L Stateful (1s window)',
            'window_size': 40,
            'num_layers': 6,
            'd_model': 256,
            'd_state': 16,
            'epochs': 100
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
        dataset = SequentialWindowDataset(spike_counts, velocities, window_size)
        
        # IMPORTANT: Sequential split (not random!) to maintain temporal structure
        n_total = len(dataset)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val
        
        # Use first 70% for training, next 15% for validation, last 15% for test
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n_total))
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        # CRITICAL: shuffle=False for stateful training
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        print(f"  Window size: {window_size} bins ({window_size * 25}ms)")
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        print(f"  Batch order: SEQUENTIAL (no shuffle)")
        
        # Create model
        model = MambaVelocityDecoder(
            n_channels=n_channels,
            window_size=window_size,
            d_model=cfg['d_model'],
            d_state=cfg['d_state'],
            num_layers=cfg['num_layers'],
            embedding_dim=128,
            dropout=0.1
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
        
        # Train
        best_r2, history = train_stateful_mamba(
            model, train_loader, val_loader, device,
            epochs=cfg['epochs']
        )
        
        elapsed = time.time() - start_time
        
        # ========================================
        # Test evaluation
        # ========================================
        model.eval()
        test_preds, test_targets = [], []
        
        with torch.no_grad():
            first_batch = next(iter(test_loader))
            model.reset_state(first_batch['window'].size(0), device)
            
            for batch in test_loader:
                window = batch['window'].to(device)
                if window.size(0) != first_batch['window'].size(0):
                    model.reset_state(window.size(0), device)
                
                output = model(window, stateful=False)
                test_preds.append(output['velocity_pred'].cpu())
                test_targets.append(batch['velocity'])
        
        test_preds = torch.cat(test_preds).numpy()
        test_targets = torch.cat(test_targets).numpy()
        
        test_r2 = r2_score(test_targets, test_preds)
        test_r2_vx = r2_score(test_targets[:, 0], test_preds[:, 0])
        test_r2_vy = r2_score(test_targets[:, 1], test_preds[:, 1])
        
        results.append({
            'name': cfg['name'],
            'window_size': window_size,
            'val_r2': best_r2,
            'test_r2': test_r2,
            'test_r2_vx': test_r2_vx,
            'test_r2_vy': test_r2_vy,
            'params': param_count,
            'time': elapsed
        })
        
        status = "🎯" if test_r2 >= 0.70 else ("📈" if test_r2 >= 0.60 else "")
        print(f"\n  Test Result: R²={test_r2:.4f} (vx={test_r2_vx:.4f}, vy={test_r2_vy:.4f}) {status}")
        print(f"  Time: {elapsed/60:.1f} min")
        
        # Save if good
        if test_r2 >= 0.60:
            save_path = Path(__file__).parent / 'models' / 'exp25_mamba_mcrtt.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'test_r2': test_r2,
                'val_r2': best_r2,
                'history': history
            }, save_path)
            print(f"  Model saved to {save_path}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("EXPERIMENT 25 RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Model':<35} {'Window':>8} {'Val R²':>8} {'Test R²':>8} {'vx':>8} {'vy':>8}")
    print("-"*80)
    
    for r in sorted(results, key=lambda x: x['test_r2'], reverse=True):
        status = "🎯" if r['test_r2'] >= 0.70 else ("📈" if r['test_r2'] >= 0.60 else "")
        print(f"{r['name']:<35} {r['window_size']:>8} {r['val_r2']:>7.4f} {r['test_r2']:>7.4f} "
              f"{r['test_r2_vx']:>7.4f} {r['test_r2_vy']:>7.4f} {status}")
    
    print("\n" + "-"*70)
    print("Context: This is the FIRST experiment on MC_RTT")
    print("Hypothesis: Stateful Mamba = Neural Kalman Filter for trajectory")
    print("="*70)
    
    best = max(results, key=lambda x: x['test_r2'])
    if best['test_r2'] >= 0.70:
        print(f"\n✅ SUCCESS: {best['name']} achieved R² = {best['test_r2']:.4f}")
        print("\nKey insight: Stateful Mamba + sequential training")
        print("transforms context from 'noise' (MC_Maze) to 'trajectory' (MC_RTT)")
    else:
        print(f"\n📊 Best: {best['name']} R² = {best['test_r2']:.4f}")
        print("\nNext steps if needed:")
        print("  • Add bidirectional context (if causal constraint can be relaxed)")
        print("  • Try longer windows (3s = 120 bins)")
        print("  • Add auxiliary losses (spike reconstruction)")
    
    return results


if __name__ == "__main__":
    run_experiment()
