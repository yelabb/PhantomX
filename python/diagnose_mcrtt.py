"""
Diagnostic script for MC_RTT data and Mamba numerical stability.
Run this BEFORE exp25 to understand the data and catch issues early.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

DATA_PATH = Path(__file__).parent.parent / "data" / "mc_rtt.nwb"


def diagnose_data():
    """Check MC_RTT data for issues."""
    print("="*60)
    print("DIAGNOSTIC: MC_RTT Data Quality Check")
    print("="*60)
    
    from pynwb import NWBHDF5IO
    
    with NWBHDF5IO(str(DATA_PATH), mode='r', load_namespaces=True) as io:
        nwb = io.read()
        
        # Units info
        units = nwb.units
        n_channels = len(units.id[:])
        print(f"\n1. Neural Units: {n_channels}")
        
        # Check spike times for each unit
        spike_counts = []
        for unit_idx in range(n_channels):
            spike_times = units.get_unit_spike_times(unit_idx)
            spike_counts.append(len(spike_times) if spike_times is not None else 0)
        
        spike_counts = np.array(spike_counts)
        print(f"   Spikes per unit: min={spike_counts.min()}, max={spike_counts.max()}, mean={spike_counts.mean():.1f}")
        print(f"   Units with 0 spikes: {(spike_counts == 0).sum()}")
        
        # Behavior data
        behavior = nwb.processing.get('behavior')
        
        # Finger velocity
        finger_vel = behavior.data_interfaces.get('finger_vel')
        vel_data = finger_vel.data[:]
        rate = finger_vel.rate
        
        print(f"\n2. Finger Velocity:")
        print(f"   Shape: {vel_data.shape}")
        print(f"   Rate: {rate} Hz")
        print(f"   Duration: {len(vel_data) / rate:.1f} seconds")
        print(f"   Range: [{vel_data.min():.4f}, {vel_data.max():.4f}]")
        print(f"   Mean: {vel_data.mean():.4f}")
        print(f"   Std: {vel_data.std():.4f}")
        print(f"   NaN count: {np.isnan(vel_data).sum()}")
        print(f"   Inf count: {np.isinf(vel_data).sum()}")
        
        # Check for outliers
        vel_abs_max = np.abs(vel_data).max()
        outliers = np.abs(vel_data) > 10 * vel_data.std()
        print(f"   Outliers (>10σ): {outliers.sum()}")
        
    return vel_data


def diagnose_binned_data():
    """Check binned data for issues."""
    print("\n" + "="*60)
    print("DIAGNOSTIC: Binned Data Check (25ms bins)")
    print("="*60)
    
    from pynwb import NWBHDF5IO
    
    bin_size_ms = 25.0
    
    with NWBHDF5IO(str(DATA_PATH), mode='r', load_namespaces=True) as io:
        nwb = io.read()
        
        units = nwb.units
        n_channels = len(units.id[:])
        
        behavior = nwb.processing.get('behavior')
        finger_vel = behavior.data_interfaces.get('finger_vel')
        vel_data = finger_vel.data[:]
        rate = finger_vel.rate
        
        # Bin parameters
        bin_samples = int(bin_size_ms * rate / 1000)
        n_bins = len(vel_data) // bin_samples
        
        print(f"\n1. Binning:")
        print(f"   Bin size: {bin_size_ms}ms = {bin_samples} samples")
        print(f"   Total bins: {n_bins}")
        
        # Bin spikes
        spike_counts = np.zeros((n_bins, n_channels), dtype=np.float32)
        
        for unit_idx in range(n_channels):
            spike_times = units.get_unit_spike_times(unit_idx)
            if spike_times is not None and len(spike_times) > 0:
                bin_indices = (spike_times * 1000 / bin_size_ms).astype(np.int32)
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                for bin_idx in bin_indices:
                    spike_counts[bin_idx, unit_idx] += 1
        
        print(f"\n2. Spike Counts:")
        print(f"   Shape: {spike_counts.shape}")
        print(f"   Range: [{spike_counts.min():.0f}, {spike_counts.max():.0f}]")
        print(f"   Mean: {spike_counts.mean():.4f}")
        print(f"   Std: {spike_counts.std():.4f}")
        print(f"   Sparsity: {(spike_counts == 0).mean() * 100:.1f}%")
        
        # Bin velocities
        velocities = np.zeros((n_bins, 2), dtype=np.float32)
        for i in range(n_bins):
            start_idx = i * bin_samples
            end_idx = start_idx + bin_samples
            velocities[i] = vel_data[start_idx:end_idx].mean(axis=0)
        
        print(f"\n3. Binned Velocities:")
        print(f"   Shape: {velocities.shape}")
        print(f"   Range: [{velocities.min():.4f}, {velocities.max():.4f}]")
        print(f"   Mean: {velocities.mean():.4f}")
        print(f"   Std: {velocities.std():.4f}")
        print(f"   NaN count: {np.isnan(velocities).sum()}")
        
        # Normalize and check
        spike_mean = spike_counts.mean(axis=0, keepdims=True)
        spike_std = spike_counts.std(axis=0, keepdims=True) + 1e-6
        spike_norm = (spike_counts - spike_mean) / spike_std
        
        vel_mean = velocities.mean(axis=0, keepdims=True)
        vel_std = velocities.std(axis=0, keepdims=True) + 1e-6
        vel_norm = (velocities - vel_mean) / vel_std
        
        print(f"\n4. After Normalization:")
        print(f"   Spikes - mean: {spike_norm.mean():.6f}, std: {spike_norm.std():.4f}")
        print(f"   Spikes - range: [{spike_norm.min():.4f}, {spike_norm.max():.4f}]")
        print(f"   Velocity - mean: {vel_norm.mean():.6f}, std: {vel_norm.std():.4f}")
        print(f"   Velocity - range: [{vel_norm.min():.4f}, {vel_norm.max():.4f}]")
        
        # Check for extreme values that could cause issues
        spike_extreme = np.abs(spike_norm) > 10
        vel_extreme = np.abs(vel_norm) > 10
        print(f"\n5. Extreme Values (|x| > 10):")
        print(f"   Spikes: {spike_extreme.sum()}")
        print(f"   Velocities: {vel_extreme.sum()}")
        
    return spike_counts, velocities


def diagnose_mamba_forward():
    """Test Mamba forward pass for numerical issues."""
    print("\n" + "="*60)
    print("DIAGNOSTIC: Mamba Forward Pass Stability")
    print("="*60)
    
    from einops import repeat
    import math
    
    # Get some real data
    spike_counts, velocities = diagnose_binned_data()
    
    # Normalize
    spike_mean = spike_counts.mean(axis=0, keepdims=True)
    spike_std = spike_counts.std(axis=0, keepdims=True) + 1e-6
    spike_norm = (spike_counts - spike_mean) / spike_std
    
    # Create a small test batch
    window_size = 80
    batch_size = 4
    n_channels = spike_counts.shape[1]
    
    # Take first few windows
    windows = np.stack([spike_norm[i:i+window_size] for i in range(batch_size)])
    x = torch.tensor(windows, dtype=torch.float32)
    
    print(f"\n1. Test Input:")
    print(f"   Shape: {x.shape}")
    print(f"   Range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"   NaN: {torch.isnan(x).sum()}")
    
    # Test simple linear projection
    d_model = 256
    proj = nn.Linear(n_channels, d_model)
    
    with torch.no_grad():
        x_proj = proj(x)
        print(f"\n2. After Input Projection:")
        print(f"   Shape: {x_proj.shape}")
        print(f"   Range: [{x_proj.min():.4f}, {x_proj.max():.4f}]")
        print(f"   NaN: {torch.isnan(x_proj).sum()}")
    
    # Test the SSM A matrix initialization
    d_state = 16
    d_inner = d_model * 2  # expand = 2
    
    A = repeat(
        torch.arange(1, d_state + 1, dtype=torch.float32),
        "n -> d n",
        d=d_inner
    )
    A_log = torch.log(A)
    A_neg = -torch.exp(A_log)  # This should be negative
    
    print(f"\n3. SSM A Matrix:")
    print(f"   A (before log): [{A.min():.4f}, {A.max():.4f}]")
    print(f"   A_log: [{A_log.min():.4f}, {A_log.max():.4f}]")
    print(f"   -exp(A_log): [{A_neg.min():.4f}, {A_neg.max():.4f}]")
    
    # Test discretization stability
    # dt should be in [dt_min, dt_max] after softplus
    dt_min, dt_max = 0.001, 0.1
    
    # Simulate dt values
    dt_raw = torch.randn(batch_size, window_size, d_inner)
    dt = F.softplus(dt_raw)
    
    print(f"\n4. Discretization (dt):")
    print(f"   dt range after softplus: [{dt.min():.6f}, {dt.max():.4f}]")
    
    # dt * A should not be too large (causes exp overflow)
    dt_A = dt.unsqueeze(-1) * A_neg.unsqueeze(0).unsqueeze(0)
    print(f"   dt * A range: [{dt_A.min():.4f}, {dt_A.max():.4f}]")
    
    # exp(dt * A) - this is A_bar
    A_bar = torch.exp(dt_A)
    print(f"   exp(dt*A) range: [{A_bar.min():.6f}, {A_bar.max():.4f}]")
    print(f"   exp(dt*A) NaN: {torch.isnan(A_bar).sum()}")
    
    # The issue: if dt is large and A is positive, exp(dt*A) explodes
    # A should always be NEGATIVE for stability
    
    # Simulate hidden state evolution
    print(f"\n5. Hidden State Evolution (80 steps):")
    h = torch.zeros(batch_size, d_inner, d_state)
    
    for t in range(window_size):
        # Simplified update: h = A_bar * h + B_bar * x
        A_bar_t = A_bar[:, t, :, :]  # [batch, d_inner, d_state]
        
        # Just multiply by A_bar repeatedly to see if it explodes
        h = A_bar_t * h + 0.01  # Add small constant instead of B*x
        
        if t % 20 == 0 or t == window_size - 1:
            h_max = h.abs().max().item()
            h_nan = torch.isnan(h).sum().item()
            print(f"   Step {t:3d}: |h|_max = {h_max:.4f}, NaN = {h_nan}")
    
    print("\n6. DIAGNOSIS:")
    if torch.isnan(h).any():
        print("   ❌ Hidden state exploded to NaN!")
        print("   → Check dt initialization and A matrix signs")
    elif h.abs().max() > 1000:
        print("   ⚠️ Hidden state growing large, may cause issues")
        print("   → Consider state normalization or smaller dt")
    else:
        print("   ✅ Hidden state appears stable")


def test_simple_baseline():
    """Test a simple MLP baseline to establish data quality."""
    print("\n" + "="*60)
    print("DIAGNOSTIC: Simple MLP Baseline (sanity check)")
    print("="*60)
    
    from sklearn.metrics import r2_score
    
    # Load data
    spike_counts, velocities = diagnose_binned_data()
    
    # Normalize
    spike_mean = spike_counts.mean(axis=0, keepdims=True)
    spike_std = spike_counts.std(axis=0, keepdims=True) + 1e-6
    spike_norm = (spike_counts - spike_mean) / spike_std
    
    vel_mean = velocities.mean(axis=0, keepdims=True)
    vel_std = velocities.std(axis=0, keepdims=True) + 1e-6
    vel_norm = (velocities - vel_mean) / vel_std
    
    # Simple 1-step prediction
    n_train = int(0.7 * len(spike_norm))
    X_train = torch.tensor(spike_norm[:n_train], dtype=torch.float32)
    y_train = torch.tensor(vel_norm[:n_train], dtype=torch.float32)
    X_val = torch.tensor(spike_norm[n_train:], dtype=torch.float32)
    y_val = torch.tensor(vel_norm[n_train:], dtype=torch.float32)
    
    print(f"\n1. Data shapes:")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # Simple 2-layer MLP
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 2)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\n2. Training simple MLP (10 epochs)...")
    
    batch_size = 256
    for epoch in range(10):
        model.train()
        losses = []
        
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).numpy()
            val_r2 = r2_score(y_val.numpy(), val_pred)
        
        if epoch % 2 == 0 or epoch == 9:
            print(f"   Epoch {epoch+1}: loss={np.mean(losses):.4f}, R²={val_r2:.4f}")
    
    print(f"\n3. BASELINE RESULT:")
    print(f"   Simple MLP (no temporal context): R² = {val_r2:.4f}")
    print(f"   This establishes the lower bound for MC_RTT")
    
    return val_r2


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MC_RTT + MAMBA DIAGNOSTIC SUITE")
    print("="*70)
    
    # Run all diagnostics
    diagnose_data()
    diagnose_binned_data()
    diagnose_mamba_forward()
    baseline_r2 = test_simple_baseline()
    
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    print(f"\nSimple MLP baseline on MC_RTT: R² = {baseline_r2:.4f}")
    print("\nIf Mamba is getting NaN, the issue is likely:")
    print("  1. dt (discretization step) too large → exp(dt*A) explodes")
    print("  2. Hidden state accumulating over long sequences")
    print("  3. Need to use proper Mamba initialization from paper")
