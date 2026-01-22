"""
MC_RTT DataModule

Motor cortex Random Target Tracking task.
Continuous tracking - fundamentally different from MC_Maze.
"""

from pathlib import Path
from typing import Tuple
import numpy as np

from .base import BaseDataModule
from . import register_datamodule


@register_datamodule('mc_rtt')
class MCRTTDataModule(BaseDataModule):
    """
    MC_RTT DataModule.
    
    Continuous random target tracking task.
    130 neural units, continuous data (no discrete trials).
    
    Key differences from MC_Maze:
    - Uses finger_vel instead of hand_vel
    - Continuous tracking, no discrete reaches
    - Longer context important (trajectory integration)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override defaults for continuous data
        if 'shuffle' not in kwargs and kwargs.get('cfg') is None:
            self.shuffle = False  # Sequential for stateful models
    
    @property
    def name(self) -> str:
        return "mc_rtt"
    
    def _get_expected_n_channels(self) -> int:
        """Expected channel count (actual may differ based on data file)."""
        return 130  # Can vary; actual determined from data
    
    @property
    def output_dim(self) -> int:
        return 2  # Finger velocity (vx, vy)
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load MC_RTT data from NWB file."""
        filepath = self.data_dir / "mc_rtt.nwb"
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"MC_RTT data not found at {filepath}. "
                "Download from DANDI Archive or run `python upload_data.py`"
            )
        
        try:
            from pynwb import NWBHDF5IO
        except ImportError:
            raise ImportError("PyNWB required. Install with: pip install pynwb")
        
        print(f"Loading MC_RTT from: {filepath}")
        
        bin_size_ms = 25.0  # 40 Hz
        
        with NWBHDF5IO(str(filepath), mode='r', load_namespaces=True) as io:
            nwb = io.read()
            
            # Get neural units
            units = nwb.units
            n_units = len(units.id[:])
            
            # Get behavior - MC_RTT uses finger_vel
            behavior = nwb.processing.get('behavior')
            finger_vel = behavior.data_interfaces.get('finger_vel')
            
            velocity = finger_vel.data[:]  # [T, 2]
            rate = finger_vel.rate  # 1000 Hz
            n_samples = len(velocity)
            
            # Bin size in samples
            bin_samples = int(bin_size_ms * rate / 1000)
            n_bins = n_samples // bin_samples
            
            print(f"  Raw: {n_samples} samples at {rate}Hz = {n_samples/rate:.1f}s")
            print(f"  Binning: {bin_samples} samples/bin → {n_bins} bins at {1000/bin_size_ms:.0f}Hz")
            
            # Bin spikes
            spike_counts = np.zeros((n_bins, n_units), dtype=np.float32)
            
            for unit_idx in range(n_units):
                spike_times = units.get_unit_spike_times(unit_idx)
                if spike_times is not None and len(spike_times) > 0:
                    bin_indices = (spike_times * 1000 / bin_size_ms).astype(np.int32)
                    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                    for idx in bin_indices:
                        spike_counts[idx, unit_idx] += 1
            
            # Bin velocities
            binned_vel = np.zeros((n_bins, 2), dtype=np.float32)
            for i in range(n_bins):
                start_idx = i * bin_samples
                end_idx = start_idx + bin_samples
                binned_vel[i] = velocity[start_idx:end_idx].mean(axis=0)
        
        print(f"  Loaded: {spike_counts.shape[0]} bins, {spike_counts.shape[1]} channels")
        mean_rate = spike_counts.mean() * (1000 / bin_size_ms)
        print(f"  Mean firing rate: {mean_rate:.2f} spikes/s")
        
        return spike_counts, binned_vel
