"""
Experiment 25b: Mamba on MC_RTT - Proper Implementation
Based on official Mamba code from state-spaces/mamba

FIXES from v1:
- Proper dt initialization (inverse softplus per official code)
- Correct softplus application order (after dt_proj, not before)
- Layer normalization for numerical stability
- Stateless design (simpler, more stable for initial testing)

TARGET: R² > 0.70 (establish baseline on new dataset)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from einops import rearrange, repeat
import math
import time
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

sys.path.insert(0, str(Path(__file__).parent))

DATA_PATH = Path(__file__).parent.parent / "data" / "mc_rtt.nwb"


# ============================================================
# MC_RTT Data Loader
# ============================================================

def load_mc_rtt_from_nwb(file_path: str, bin_size_ms: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load spike counts and kinematics from MC_RTT NWB file."""
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
        
        units = nwb.units
        n_channels = len(units.id[:])
        print(f"  Found {n_channels} neural units")
        
        behavior = nwb.processing.get('behavior')
        finger_vel_container = behavior.data_interfaces.get('finger_vel')
        finger_vel = finger_vel_container.data[:]
        rate = finger_vel_container.rate
        n_samples = len(finger_vel)
        duration = n_samples / rate
        
        print(f"  Raw data: {n_samples} samples at {rate}Hz = {duration:.1f}s")
        
        bin_samples = int(bin_size_ms * rate / 1000)
        n_bins = n_samples // bin_samples
        
        print(f"  Binning: {bin_samples} samples/bin → {n_bins} bins at {1000/bin_size_ms:.0f}Hz")
        
        # Bin spikes
        spike_counts = np.zeros((n_bins, n_channels), dtype=np.float32)
        
        for unit_idx in range(n_channels):
            spike_times = units.get_unit_spike_times(unit_idx)
            if spike_times is not None and len(spike_times) > 0:
                bin_indices = (spike_times * 1000 / bin_size_ms).astype(np.int32)
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                for bin_idx in bin_indices:
                    spike_counts[bin_idx, unit_idx] += 1
        
        # Bin velocities
        velocities = np.zeros((n_bins, 2), dtype=np.float32)
        for i in range(n_bins):
            start_idx = i * bin_samples
            end_idx = start_idx + bin_samples
            velocities[i] = np.nanmean(finger_vel[start_idx:end_idx], axis=0)  # Use nanmean to handle NaN
        
        # Check for NaN in velocities and interpolate
        nan_mask = np.isnan(velocities).any(axis=1)
        n_nan = nan_mask.sum()
        if n_nan > 0:
            print(f"  WARNING: {n_nan} bins have NaN velocities, interpolating...")
            # Linear interpolation for NaN values
            for dim in range(2):
                vel_dim = velocities[:, dim]
                nan_idx = np.isnan(vel_dim)
                if nan_idx.any():
                    valid_idx = ~nan_idx
                    x_valid = np.where(valid_idx)[0]
                    x_nan = np.where(nan_idx)[0]
                    velocities[x_nan, dim] = np.interp(x_nan, x_valid, vel_dim[valid_idx])
        
        # Verify no NaN remaining
        assert not np.isnan(velocities).any(), "NaN still present after interpolation!"
        
        print(f"  Loaded: spike_counts {spike_counts.shape}, velocities {velocities.shape}")
        
        return spike_counts, velocities


# ============================================================
# Proper Mamba Implementation (Based on Official Code)
# ============================================================

class MambaLayer(nn.Module):
    """
    Proper Mamba (S6) Layer following the official implementation.
    
    Key differences from naive implementation:
    1. dt_proj.bias initialized with inverse softplus
    2. softplus applied after adding bias (handled by dt_proj)
    3. Proper initialization of dt_proj.weight
    4. A is always NEGATIVE for stability
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
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
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # 1D convolution
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
        
        # ==== PROPER INITIALIZATION (from official Mamba) ====
        
        # Initialize dt_proj.weight
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt_proj.bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        
        # Inverse of softplus: x = log(exp(y) - 1) = y + log(1 - exp(-y)) 
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # S4D real initialization for A
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape
        
        # Project input
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)
        
        # 1D convolution
        x_conv = rearrange(x_inner, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        x_conv = rearrange(x_conv, "b d l -> b l d")
        x_conv = F.silu(x_conv)
        
        # S6: compute dt, B, C
        x_dbl = self.x_proj(x_conv)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Project dt and apply softplus
        dt = self.dt_proj(dt)  # This adds the properly initialized bias
        dt = F.softplus(dt)    # Ensures dt > 0
        
        # A is always negative for stability
        A = -torch.exp(self.A_log.float())
        
        # Selective scan
        y = self._selective_scan(x_conv, dt, A, B, C)
        
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
    ) -> torch.Tensor:
        """
        Selective scan with proper numerical handling.
        
        Args:
            x: [batch, seq_len, d_inner]
            dt: [batch, seq_len, d_inner] - discretization step (positive)
            A: [d_inner, d_state] - state matrix (NEGATIVE)
            B: [batch, seq_len, d_state] - input matrix
            C: [batch, seq_len, d_state] - output matrix
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]    # [batch, d_inner]
            dt_t = dt[:, t, :]  # [batch, d_inner]
            B_t = B[:, t, :]    # [batch, d_state]
            C_t = C[:, t, :]    # [batch, d_state]
            
            # Discretize: A_bar = exp(dt * A)
            # dt > 0 (from softplus), A < 0, so dt * A < 0
            # Therefore exp(dt * A) is in (0, 1) - STABLE
            dt_A = torch.einsum("bd,dn->bdn", dt_t, A)  # [batch, d_inner, d_state]
            A_bar = torch.exp(dt_A)  # Values in (0, 1)
            
            # B_bar = dt * B (simplified ZOH discretization)
            B_bar = torch.einsum("bd,bn->bdn", dt_t, B_t)  # [batch, d_inner, d_state]
            
            # State update: h = A_bar * h + B_bar * x
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)
            
            # Output: y = C * h + D * x
            y_t = torch.einsum("bdn,bn->bd", h, C_t) + self.D * x_t
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    """Mamba block with residual connection and layer norm."""
    
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaLayer(d_model, d_state=d_state, expand=expand)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class MambaEncoder(nn.Module):
    """Mamba encoder for spike sequence processing."""
    
    def __init__(
        self,
        n_channels: int = 130,
        d_model: int = 256,
        d_state: int = 16,
        num_layers: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        output_dim: int = 128,
        max_len: int = 100
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.d_model = d_model
        
        # Input projection with layer norm for stability
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, d_model),
            nn.LayerNorm(d_model)
        )
        
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
            x: [batch, seq_len, n_channels]
        Returns:
            z: [batch, output_dim]
        """
        B, T, C = x.shape
        
        # Input projection + positional embedding
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :T, :]
        
        # Process through Mamba layers
        for layer in self.layers:
            x = layer(x)
        
        # Output: take last timestep
        x = self.ln_final(x)
        z = self.output_proj(x[:, -1, :])
        
        return z


class MambaVelocityDecoder(nn.Module):
    """Mamba model for velocity decoding."""
    
    def __init__(
        self,
        n_channels: int = 130,
        window_size: int = 80,
        d_model: int = 256,
        d_state: int = 16,
        num_layers: int = 4,
        embedding_dim: int = 128,
        output_dim: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.window_size = window_size
        
        self.encoder = MambaEncoder(
            n_channels=n_channels,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
            output_dim=embedding_dim,
            dropout=dropout
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict:
        z = self.encoder(x)
        velocity_pred = self.decoder(z)
        
        output = {'velocity_pred': velocity_pred, 'z': z}
        
        if targets is not None:
            loss = F.mse_loss(velocity_pred, targets)
            output['loss'] = loss
        
        return output


# ============================================================
# Dataset
# ============================================================

class WindowDataset(Dataset):
    """Dataset for window-based processing."""
    
    def __init__(
        self,
        spike_counts: np.ndarray,
        velocities: np.ndarray,
        window_size: int = 80,
        normalize: bool = True
    ):
        self.window_size = window_size
        
        if normalize:
            # Normalize per channel
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
        self.velocities = velocities[window_size-1:window_size-1+n]
        self.normalized = normalize
    
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

def train_model(
    model: MambaVelocityDecoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3
):
    """Training loop with proper gradient handling."""
    
    print("\n" + "="*60)
    print("Training Mamba on MC_RTT")
    print("="*60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_r2 = -float('inf')
    best_state = None
    history = {'train_loss': [], 'val_r2': []}
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            window = batch['window'].to(device)
            velocity = batch['velocity'].to(device)
            
            optimizer.zero_grad()
            output = model(window, targets=velocity)
            loss = output['loss']
            
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at epoch {epoch}, skipping batch")
                continue
            
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
        
        if np.isnan(val_preds).any():
            print(f"  WARNING: NaN in predictions at epoch {epoch}")
            val_r2 = -1.0
        else:
            val_r2 = r2_score(val_targets, val_preds)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        avg_loss = np.mean(train_losses) if train_losses else float('nan')
        history['train_loss'].append(avg_loss)
        history['val_r2'].append(val_r2)
        
        if epoch % 10 == 0 or epoch == 1:
            status = "🎯" if val_r2 >= 0.70 else ("📈" if val_r2 >= 0.60 else "")
            print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, R²={val_r2:.4f} {status}")
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n  Training complete. Best R² = {best_val_r2:.4f}")
    
    return best_val_r2, history


def run_experiment():
    """Run the experiment."""
    
    print("\n" + "="*70)
    print("EXPERIMENT 25b: Mamba on MC_RTT (Proper Implementation)")
    print("="*70)
    print("\nKey fixes in v2:")
    print("  • Proper Mamba initialization (official implementation)")
    print("  • Correct dt/softplus handling (bias init with inverse softplus)")
    print("  • A is always negative: exp(dt * A) in (0,1) = stable")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading MC_RTT dataset...")
    spike_counts, velocities = load_mc_rtt_from_nwb(str(DATA_PATH))
    
    n_channels = spike_counts.shape[1]
    print(f"  Channels: {n_channels}")
    print(f"  Duration: {len(spike_counts) / 40:.1f}s ({len(spike_counts)} bins)")
    
    # Configurations
    configs = [
        {
            'name': 'Mamba-4L (2s window)',
            'window_size': 80,
            'num_layers': 4,
            'd_model': 256,
            'd_state': 16,
            'epochs': 100,
            'lr': 1e-3
        },
        {
            'name': 'Mamba-6L (2s window)',
            'window_size': 80,
            'num_layers': 6,
            'd_model': 256,
            'd_state': 16,
            'epochs': 100,
            'lr': 5e-4
        },
        {
            'name': 'Mamba-4L (1s window)',
            'window_size': 40,
            'num_layers': 4,
            'd_model': 256,
            'd_state': 16,
            'epochs': 100,
            'lr': 1e-3
        },
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Configuration: {cfg['name']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        window_size = cfg['window_size']
        dataset = WindowDataset(spike_counts, velocities, window_size)
        
        # Sequential split
        n_total = len(dataset)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n_total))
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        # Shuffle OK for stateless model
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        
        print(f"  Window: {window_size} bins ({window_size * 25}ms)")
        print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
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
        best_r2, history = train_model(
            model, train_loader, val_loader, device,
            epochs=cfg['epochs'], lr=cfg['lr']
        )
        
        elapsed = time.time() - start_time
        
        # Test evaluation
        model.eval()
        test_preds, test_targets = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                window = batch['window'].to(device)
                output = model(window)
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
        print(f"\n  Test: R²={test_r2:.4f} (vx={test_r2_vx:.4f}, vy={test_r2_vy:.4f}) {status}")
        print(f"  Time: {elapsed/60:.1f} min")
        
        if test_r2 >= 0.50:
            save_path = Path(__file__).parent / 'models' / 'exp25b_mamba_mcrtt.pt'
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'test_r2': test_r2,
                'history': history
            }, save_path)
            print(f"  Model saved to {save_path}")
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT 25b RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Model':<30} {'Window':>8} {'Val R²':>8} {'Test R²':>8} {'vx':>8} {'vy':>8}")
    print("-"*75)
    
    for r in sorted(results, key=lambda x: x['test_r2'], reverse=True):
        status = "🎯" if r['test_r2'] >= 0.70 else ("📈" if r['test_r2'] >= 0.60 else "")
        print(f"{r['name']:<30} {r['window_size']:>8} {r['val_r2']:>7.4f} {r['test_r2']:>7.4f} "
              f"{r['test_r2_vx']:>7.4f} {r['test_r2_vy']:>7.4f} {status}")
    
    best = max(results, key=lambda x: x['test_r2'])
    if best['test_r2'] >= 0.70:
        print(f"\n✅ SUCCESS: {best['name']} achieved R² = {best['test_r2']:.4f}")
    else:
        print(f"\n📊 Best: {best['name']} R² = {best['test_r2']:.4f}")
        print("\nMC_RTT baseline established. Next steps:")
        print("  • Compare with LSTM baseline")
        print("  • Try Transformer for comparison")
        print("  • Add auxiliary reconstruction loss")
    
    return results


if __name__ == "__main__":
    run_experiment()
